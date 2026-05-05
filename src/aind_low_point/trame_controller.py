"""Trame + PyVista controller for probe manipulation.

Parallel implementation alongside the K3D / ipywidgets controller.
Calls store.dispatch() directly — PlanStore is the shared abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pyvista as pv
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3

import numpy as np
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from aind_mri_utils.arc_angles import arc_angles_to_affine

from aind_low_point.assets import AssetCatalog
from aind_low_point.ccf_ontology import CCFOntology
from aind_low_point.ccf_overlay import CCFOverlayManager
from aind_low_point.collisions import CollisionHandler
from aind_low_point.commands import (
    AssignProbeArc,
    SetArcAngle,
    SetProbeKind,
    SetProbePastTarget,
    SetProbeLocalAngles,
    SetProbeOffsetsRA,
    SetProbeTarget,
)
from aind_low_point.core import Material
from aind_low_point.planning import ProbePose
from aind_low_point.rendering import OverlayResolver, OverlaySpec, RendererAdapter
from aind_low_point.runtime import _depth_along_probe_axis, detect_shank_tips_local
from aind_low_point.state_change import PlanStore


# Overlay colour + priority for over-insertion warnings. Collisions are
# at priority 30, so they still win when both overlays apply to the same
# probe (a colliding probe will appear collision-red even if
# additionally over-inserted).
OVERINSERTION_OVERLAY = OverlaySpec(
    color=0xFF8800,  # orange
    alpha=0.7,
    source="overinsertion",
    priority=25,
)


# Predefined visibility / opacity groups exposed in the UI. Each group is
# matched against a NodeInstance's ``tags`` set: a node is "in" the group
# if any of its tags appears here. Probes are excluded — they are the
# planning subjects and always visible.
VISIBILITY_GROUPS: list[tuple[str, str, set[str], set[str]]] = [
    # (state-key, display label, tags-to-include, tags-to-exclude).
    # Groups are independent — a node ends up in every group whose
    # include set intersects its tags AND whose exclude set doesn't.
    # The exclude column is what keeps the implant slider distinct
    # from the rest of the fixtures (the implant has both "implant"
    # and "fixture" tags, so excluding "implant" from the broader
    # group is how we split it out without renaming any tags).
    ("brain", "Brain outline", {"brain"}, set()),
    ("structures", "CCF regions", {"structure"}, set()),
    ("implant", "Implant", {"implant"}, set()),
    ("fixtures", "Other fixtures", {"fixture", "headframe"}, {"implant"}),
]


def _ccf_color_for_target(
    catalog: AssetCatalog, target_key: str | None
) -> str | None:
    """Return the CCF color_hex for *target_key* if its source asset
    carries CCF metadata (set by AtlasMeshPackSpecModel.expand). Returns
    None for targets that aren't derived from a CCF region."""
    if target_key is None:
        return None
    target_spec = catalog.targets.get(target_key)
    if target_spec is None or target_spec.source_key is None:
        return None
    source_spec = catalog.assets.get(target_spec.source_key)
    if source_spec is None:
        return None
    acronym = source_spec.metadata.get("ccf_acronym")
    if not acronym:
        return None
    structure = CCFOntology.from_bundled().find_by_acronym(acronym)
    return structure.color_hex if structure else None


@dataclass
class TrameController:
    store: PlanStore
    assets: AssetCatalog
    plotter: pv.Plotter
    render_adapter: RendererAdapter
    collision_handler: CollisionHandler
    overlays_resolver: OverlayResolver
    couple_ml: bool = False
    ccf_overlay: CCFOverlayManager | None = field(default=None)
    on_save: Callable[[], None] | None = None
    on_export_plan: Callable[[], None] | None = None
    # Per-probe-asset shank tip positions in local mm — populated lazily
    # the first time a probe is checked for over-insertion. Independent
    # of pose, so this only depends on the asset's mesh.
    _shank_tips_cache: dict = field(default_factory=dict, repr=False)
    # Set of node ids currently flagged as over-inserted. Used to detect
    # transitions so we only repaint probes whose status flipped.
    _prev_overinserted: set = field(default_factory=set, repr=False)

    def build_app(self, server=None):
        """Build trame app with Vuetify3 UI + PyVista 3D view.

        Returns the trame server (call ``server.start()`` to launch).
        """
        server = server or get_server()
        state, ctrl = server.state, server.controller

        self.render_adapter.overlays = self.overlays_resolver

        self._init_state(state)
        self._wire_handlers(state)
        self._build_layout(server, ctrl)

        return server

    def _init_state(self, state) -> None:
        """Populate trame reactive state from domain."""
        probe_names = sorted(self.store.state.probes.keys())
        arc_ids = sorted(self.store.state.kinematics.arc_angles.keys())
        target_names = sorted(self.assets.targets.keys())
        # Available probe kinds from the catalog (everything keyed
        # "probe:<kind>"). The kind value stored in plan.probes is just
        # "<kind>"; the renderer looks the asset up as "probe:<kind>".
        probe_kinds = sorted(
            k.split(":", 1)[1]
            for k in self.assets.assets
            if k.startswith("probe:")
        )

        with state:
            state.probes = probe_names
            state.probe = probe_names[0] if probe_names else None
            state.arcs = arc_ids
            state.arc = arc_ids[0] if arc_ids else None
            state.targets = target_names
            state.target = target_names[0] if target_names else None
            state.probe_kinds = probe_kinds
            state.probe_kind = ""  # populated by _load_probe_state

            # Read-only geometric readouts for the currently selected probe.
            state.probe_tip_str = "—"
            state.probe_depth_str = "—"
            state.probe_overinsertion_str = "—"

            # Visibility / opacity per asset group. Initial values are taken
            # from each group's first node so the UI matches what's drawn.
            for skey, _label, include, exclude in VISIBILITY_GROUPS:
                members = self._nodes_with_any_tag(include, exclude)
                if not members:
                    setattr(state, f"{skey}_visible", True)
                    setattr(state, f"{skey}_opacity", 1.0)
                    continue
                first = members[0]
                base_mat = first.material_override or self.assets.get_spec(
                    first.asset_key
                ).default_material
                setattr(state, f"{skey}_visible", bool(base_mat.visible))
                setattr(state, f"{skey}_opacity", float(base_mat.opacity))

            state.offset_r = 0.0
            state.offset_a = 0.0
            state.depth = 0.0
            state.ap_tilt = 0.0
            state.ml_tilt = 0.0
            state.spin = 0

        self._load_probe_state(state)
        # Initial CCF-based colouring for every probe whose target is a
        # CCF-derived region. Collect all affected node IDs and issue a
        # single repaint_materials call instead of one per probe.
        colored_nids: list[str] = []
        for name in probe_names:
            plan = self.store.state.probes.get(name)
            if plan is None or plan.target_key is None:
                continue
            color = _ccf_color_for_target(self.assets, plan.target_key)
            if color is None:
                continue
            nid = f"probe:{name}"
            node = self.render_adapter.scene.nodes.get(nid)
            if node is None:
                continue
            base = node.material_override
            if base is None:
                base = self.assets.get_spec(node.asset_key).default_material
            node.material_override = base.replace(color_hex_str=color)
            colored_nids.append(nid)
        if colored_nids:
            self.render_adapter.repaint_materials(colored_nids)
        self.render_adapter.backend.flush()

        # Subscribe to the plan store so the readouts update on every
        # dispatch involving the currently-selected probe.
        self._readout_state = state  # captured for the closure
        self.store.subscribe(self._on_plan_change_for_readouts)
        # Run an initial over-insertion pass for every probe so the
        # overlay reflects the loaded plan from the moment the GUI
        # comes up (rather than waiting for the first dispatch).
        self._refresh_overinsertion_overlay(list(probe_names))

        # CCF overlay state
        if self.ccf_overlay is not None:
            state.ccf_search_query = ""
            state.ccf_search_results = []
            state.ccf_selected_region = None
            state.ccf_visible_regions = []
            state.ccf_global_opacity = self.ccf_overlay.global_opacity

    def _wire_handlers(self, state) -> None:
        """Register trame reactive handlers."""

        @state.change("probe")
        def on_probe_change(**kwargs):
            self._load_probe_state(state)

        @state.change("offset_r", "offset_a")
        def on_offsets(**kwargs):
            if not state.probe:
                return
            self.store.dispatch(
                SetProbeOffsetsRA(
                    name=state.probe,
                    R_mm=float(state.offset_r),
                    A_mm=float(state.offset_a),
                )
            )

        @state.change("ap_tilt")
        def on_ap(**kwargs):
            if not state.probe:
                return
            self._on_ap_change(state, state.probe, float(state.ap_tilt))

        @state.change("ml_tilt")
        def on_ml(**kwargs):
            if not state.probe:
                return
            self._on_ml_change(state, state.probe, float(state.ml_tilt))

        @state.change("depth")
        def on_depth(**kwargs):
            if not state.probe:
                return
            self.store.dispatch(
                SetProbePastTarget(
                    name=state.probe,
                    past_target_mm=float(state.depth),
                )
            )

        @state.change("spin")
        def on_spin(**kwargs):
            if not state.probe:
                return
            self.store.dispatch(
                SetProbeLocalAngles(name=state.probe, spin=float(state.spin))
            )

        @state.change("arc")
        def on_arc_assign(**kwargs):
            self._on_arc_assign(state)

        @state.change("probe_kind")
        def on_kind_change(**kwargs):
            if not state.probe or not state.probe_kind:
                return
            self._on_probe_kind_change(state.probe, str(state.probe_kind))

        for skey, _label, include, exclude in VISIBILITY_GROUPS:
            self._wire_visibility_handlers(state, skey, include, exclude)

        if self.ccf_overlay is not None:
            self._wire_ccf_handlers(state, self.ccf_overlay)

    def _wire_ccf_handlers(self, state, ccf: CCFOverlayManager) -> None:
        """Register trame handlers for CCF overlay controls."""

        @state.change("ccf_search_query")
        def on_ccf_search(**kwargs):
            q = state.ccf_search_query or ""
            if not q:
                state.ccf_search_results = []
                return
            state.ccf_search_results = ccf.ontology.autocomplete_items(q, limit=50)

        @state.change("ccf_selected_region")
        def on_ccf_select(**kwargs):
            label_id = state.ccf_selected_region
            if label_id is None:
                return
            ccf.show(label_id)
            self._sync_ccf_visible(state)

        @state.change("ccf_global_opacity")
        def on_ccf_opacity(**kwargs):
            ccf.set_global_opacity(float(state.ccf_global_opacity))

    def _sync_ccf_visible(self, state) -> None:
        """Sync trame state with currently visible CCF regions."""
        if self.ccf_overlay is None:
            return
        state.ccf_visible_regions = [
            {
                "label_id": r.label_id,
                "acronym": r.structure.acronym,
                "name": r.structure.name,
                "color": r.color,
            }
            for r in self.ccf_overlay.visible_regions()
        ]

    def _on_ap_change(self, state, probe: str, ap: float) -> None:
        plan = self.store.state.probes.get(probe)
        if plan and plan.arc_id and plan.bind_ap_to_arc:
            self.store.dispatch(SetArcAngle(arc_id=plan.arc_id, ap_deg=ap))
        else:
            self.store.dispatch(SetProbeLocalAngles(name=probe, ap_local=ap))

    def _on_ml_change(self, state, probe: str, ml: float) -> None:
        plan = self.store.state.probes.get(probe)
        if not plan:
            return
        if plan.calibrated and probe in self.store.state.calibrations:
            state.ml_tilt = float(plan.ml_local)
            return
        if self.couple_ml:
            arc_id = plan.arc_id
            for name, other in self.store.state.probes.items():
                if other.calibrated and name in self.store.state.calibrations:
                    continue
                if other.arc_id == arc_id:
                    self.store.dispatch(SetProbeLocalAngles(name=name, ml_local=ml))
        else:
            self.store.dispatch(SetProbeLocalAngles(name=probe, ml_local=ml))

    def _on_arc_assign(self, state) -> None:
        if not state.probe or not state.arc:
            return
        plan = self.store.state.probes.get(state.probe)
        if not plan or plan.arc_id == state.arc:
            return
        self.store.dispatch(
            AssignProbeArc(
                name=state.probe,
                arc_id=state.arc,
                bind_ap_to_arc=True,
            )
        )
        new_angle = float(self.store.state.kinematics.arc_angles.get(state.arc, 0.0))
        state.ap_tilt = new_angle

    def _nodes_with_any_tag(
        self,
        include: set[str],
        exclude: set[str] = frozenset(),
    ):
        """Scene nodes whose tag set intersects *include* and is disjoint
        from *exclude* (used by visibility groups)."""
        return [
            n
            for n in self.render_adapter.scene.nodes.values()
            if (n.tags & include) and not (n.tags & exclude)
        ]

    def _apply_material_to_nodes(
        self,
        node_ids,
        *,
        opacity: float | None = None,
        visible: bool | None = None,
    ) -> None:
        """Mutate ``material_override`` on a set of nodes, preserving every
        field except the ones explicitly overridden. Issues a single
        ``repaint_materials`` call for the affected nodes."""
        affected: list[str] = []
        for nid in node_ids:
            node = self.render_adapter.scene.nodes.get(nid)
            if node is None:
                continue
            base = node.material_override
            if base is None:
                base = self.assets.get_spec(node.asset_key).default_material
            overrides = {}
            if opacity is not None:
                overrides["opacity"] = float(opacity)
            if visible is not None:
                overrides["visible"] = bool(visible)
            node.material_override = base.replace(**overrides)
            affected.append(nid)
        if affected:
            self.render_adapter.repaint_materials(affected)

    def _wire_visibility_handlers(
        self, state, skey: str, include: set[str], exclude: set[str]
    ) -> None:
        """Register reactive handlers for the (skey)_visible and
        (skey)_opacity state variables — applies changes to every scene
        node whose tags intersect *include* and don't touch *exclude*."""
        vis_var = f"{skey}_visible"
        op_var = f"{skey}_opacity"

        @state.change(vis_var)
        def _on_visible(**_):
            members = self._nodes_with_any_tag(include, exclude)
            self._apply_material_to_nodes(
                [n.key for n in members],
                visible=bool(getattr(state, vis_var)),
            )

        @state.change(op_var)
        def _on_opacity(**_):
            members = self._nodes_with_any_tag(include, exclude)
            self._apply_material_to_nodes(
                [n.key for n in members],
                opacity=float(getattr(state, op_var)),
            )

    def _compute_probe_readouts(
        self, probe_name: str
    ) -> tuple[str, str, str]:
        """Return ``(tip_RAS_str, depth_str, overinsertion_str)`` for the
        named probe — pre-formatted for direct display.

        ``overinsertion_str`` is ``"⚠ <n>/<N> shanks"`` if any shank tip
        has 2+ brain-surface intersections along the +probe-z ray (the
        tip has gone through to the back side of the brain), ``"OK"``
        if all shanks have ≤1 intersection, or ``"—"`` when no brain
        mesh is loaded.
        """
        plan = self.store.state.probes.get(probe_name)
        if plan is None:
            return "—", "—", "—"
        pose = ProbePose.from_planning_state(self.store.state, probe_name)
        tip_lps = np.asarray(pose.tip, dtype=np.float64)
        tip_ras = convert_coordinate_system(tip_lps, "LPS", "RAS")
        tip_str = f"{tip_ras[0]:+.2f}, {tip_ras[1]:+.2f}, {tip_ras[2]:+.2f} mm"

        depth_str = "—"
        overins_str = "—"
        brain_spec = self.assets.assets.get("brain")
        if brain_spec is not None and brain_spec.mesh is not None:
            R = arc_angles_to_affine(pose.ap, pose.ml, pose.spin)
            probe_axis = R @ np.array([0.0, 0.0, 1.0])
            brain_mesh = brain_spec.mesh.raw
            depth = _depth_along_probe_axis(tip_lps, probe_axis, brain_mesh)
            if depth is not None:
                depth_str = f"{depth:.2f} mm"
            n_over, n_total = self._count_overinserted_shanks(
                probe_name, pose, R, brain_mesh
            )
            if n_total == 0:
                overins_str = "—"
            elif n_over == 0:
                overins_str = "OK"
            else:
                overins_str = f"⚠ {n_over}/{n_total} shanks"
        return tip_str, depth_str, overins_str

    def _shank_tips_local(self, asset_key: str) -> np.ndarray:
        """Lazy per-asset shank-tip detection cache. Logs the count once
        on first access so a mis-detection on an exotic mesh is visible
        immediately."""
        if asset_key in self._shank_tips_cache:
            return self._shank_tips_cache[asset_key]
        spec = self.assets.assets.get(asset_key)
        if spec is None or spec.mesh is None:
            tips = np.zeros((1, 3), dtype=np.float64)
        else:
            tips = detect_shank_tips_local(spec.mesh.raw)
        self._shank_tips_cache[asset_key] = tips
        print(f"  {asset_key}: detected {len(tips)} shank tip(s)")
        return tips

    def _count_overinserted_shanks(
        self,
        probe_name: str,
        pose: ProbePose,
        R: np.ndarray,
        brain_mesh,
    ) -> tuple[int, int]:
        """Return ``(n_overinserted, n_total)`` — how many of the probe's
        shanks have 2+ brain-surface intersections along the +probe-z
        ray vs the total shank count.

        ``probe:X`` scene node's ``asset_key`` determines which probe
        mesh (and therefore which shank pattern) we're checking.
        """
        nid = f"probe:{probe_name}"
        node = self.render_adapter.scene.nodes.get(nid)
        if node is None:
            return 0, 0
        local_tips = self._shank_tips_local(node.asset_key)
        if len(local_tips) == 0:
            return 0, 0
        # local → world: world_tip = R @ local_tip + pose.tip. The probe
        # mesh's local origin is at shank-0's tip (canonical "centeredOn"
        # naming), and pose.tip is the world position of that origin.
        world_tips = local_tips @ R.T + np.asarray(pose.tip, dtype=np.float64)
        probe_axis = R @ np.array([0.0, 0.0, 1.0])
        n_over = 0
        for tip_w in world_tips:
            try:
                locs, _, _ = brain_mesh.ray.intersects_location(
                    ray_origins=tip_w[None, :],
                    ray_directions=probe_axis[None, :],
                )
            except Exception:
                continue
            if len(locs) >= 2:
                n_over += 1
        return n_over, len(local_tips)

    def _refresh_readouts(self, state, probe_name: str) -> None:
        tip_str, depth_str, overins_str = self._compute_probe_readouts(probe_name)
        with state:
            state.probe_tip_str = tip_str
            state.probe_depth_str = depth_str
            state.probe_overinsertion_str = overins_str

    def _refresh_overinsertion_overlay(self, changed_ids) -> None:
        """For every probe in *changed_ids*, recompute over-insertion and
        update the shared overlay state. Repaints only the probes whose
        over-insertion status flipped."""
        brain_spec = self.assets.assets.get("brain")
        if brain_spec is None or brain_spec.mesh is None:
            return
        brain_mesh = brain_spec.mesh.raw
        flips: list[str] = []
        for probe_name in changed_ids:
            if probe_name not in self.store.state.probes:
                continue
            pose = ProbePose.from_planning_state(self.store.state, probe_name)
            R = arc_angles_to_affine(pose.ap, pose.ml, pose.spin)
            n_over, n_total = self._count_overinserted_shanks(
                probe_name, pose, R, brain_mesh
            )
            nid = f"probe:{probe_name}"
            was = nid in self._prev_overinserted
            is_now = n_over > 0 and n_total > 0
            if is_now and not was:
                self._prev_overinserted.add(nid)
                flips.append(nid)
            elif was and not is_now:
                self._prev_overinserted.discard(nid)
                flips.append(nid)
        if not flips:
            return
        overlays_state = self.overlays_resolver.overlays
        overlays_state.clear_source("overinsertion")
        if self._prev_overinserted:
            overlays_state.set_for_source(
                list(self._prev_overinserted), OVERINSERTION_OVERLAY
            )
        self.render_adapter.repaint_materials(flips)

    def _on_plan_change_for_readouts(self, plan, changed_ids) -> None:
        """PlanStore subscriber: refresh readouts + over-insertion
        overlays when probe poses change. The overlay update covers all
        changed probes (an arc change can move several at once); the
        textual readout only tracks the active probe shown in the UI."""
        state = getattr(self, "_readout_state", None)
        if state is None:
            return
        self._refresh_overinsertion_overlay(changed_ids)
        if state.probe and state.probe in changed_ids:
            self._refresh_readouts(state, state.probe)

    def _apply_target_based_color(self, probe_name: str) -> None:
        """Set ``probe:<name>`` node's material_override to the CCF color of
        its current target (no-op if the target isn't CCF-derived)."""
        plan = self.store.state.probes.get(probe_name)
        if plan is None or plan.target_key is None:
            return
        color = _ccf_color_for_target(self.assets, plan.target_key)
        if color is None:
            return
        nid = f"probe:{probe_name}"
        node = self.render_adapter.scene.nodes.get(nid)
        if node is None:
            return
        # Inherit other material fields (opacity, point_size, …) from the
        # current override or the asset's default material.
        base = node.material_override
        if base is None:
            base = self.assets.get_spec(node.asset_key).default_material
        node.material_override = base.replace(color_hex_str=color)
        self.render_adapter.repaint_materials([nid])

    def _on_probe_kind_change(self, probe_name: str, new_kind: str) -> None:
        """Swap the probe's mesh by changing its ``kind``.

        Updates the scene node's ``asset_key`` to the new ``probe:<kind>``,
        drops the renderer / collision handles for the old mesh, dispatches
        SetProbeKind to update the planning state (which fires RenderHandler
        to re-create the renderer node), and re-registers the collision
        object with the new BVH.
        """
        new_asset_key = f"probe:{new_kind}"
        if new_asset_key not in self.assets.assets:
            return  # unknown probe type — defensive
        plan = self.store.state.probes.get(probe_name)
        if plan is None or plan.kind == new_kind:
            return
        nid = f"probe:{probe_name}"
        scene = self.render_adapter.scene
        node = scene.nodes.get(nid)
        if node is None:
            return

        # Mutate scene node so the renderer / collision adapter see the
        # new geometry on their next pass.
        node.asset_key = new_asset_key

        # Drop existing handles so they get recreated with the new mesh.
        self.render_adapter.backend.remove([nid])
        self.collision_handler.adapter.remove_nodes([nid])

        # Update planning state — fires RenderHandler which calls
        # sync_nodes → _upsert_node → has_node=False → create_mesh with the
        # new geometry.
        self.store.dispatch(SetProbeKind(name=probe_name, kind=new_kind))

        # Re-register collision object with the new BVH (sync handles
        # missing-node case by creating).
        self.collision_handler.adapter.on_store_change(
            self.store.state, [probe_name]
        )

        # Re-apply target-based colour to the new node.
        self._apply_target_based_color(probe_name)
        self.render_adapter.backend.flush()

    def _build_layout(self, server, ctrl) -> None:
        """Build the Vuetify3 + PyVista layout."""

        def on_set_target():
            state = server.state
            if not state.probe or not state.target:
                return
            self.store.dispatch(
                SetProbeTarget(name=state.probe, target_key=state.target)
            )
            self.store.dispatch(
                SetProbePastTarget(name=state.probe, past_target_mm=0.0)
            )
            state.depth = 0.0
            # Re-colour the probe to match the new target's CCF region.
            self._apply_target_based_color(state.probe)

        with SinglePageLayout(server) as layout:
            layout.title.set_text("Probe Planner")

            with layout.content:
                with vuetify3.VContainer(fluid=True, classes="fill-height"):
                    with vuetify3.VRow(classes="fill-height"):
                        self._build_controls(on_set_target)
                        with vuetify3.VCol(cols=9, classes="fill-height"):
                            view = plotter_ui(self.plotter, mode="client")
                            ctrl.view_update = view.update

    def _build_controls(self, on_set_target) -> None:
        """Build the left-column control widgets."""
        with vuetify3.VCol(cols=3):
            vuetify3.VSelect(
                v_model=("probe",),
                items=("probes",),
                label="Probe",
                hide_details=True,
                density="compact",
            )
            vuetify3.VSelect(
                v_model=("arc",),
                items=("arcs",),
                label="Arc",
                hide_details=True,
                density="compact",
            )
            vuetify3.VSelect(
                v_model=("probe_kind",),
                items=("probe_kinds",),
                label="Probe type",
                hide_details=True,
                density="compact",
            )
            vuetify3.VDivider(classes="my-2")
            self._slider_row("offset_r", "R (mm)", -7.5, 7.5, 0.05)
            self._slider_row("offset_a", "A (mm)", -7.5, 7.5, 0.05)
            self._slider_row("depth", "Depth (mm)", -10, 10, 0.1)
            vuetify3.VDivider(classes="my-2")
            self._slider_row("ap_tilt", "AP tilt (°)", -60, 60, 0.5)
            self._slider_row("ml_tilt", "ML tilt (°)", -60, 60, 0.5)
            self._slider_row("spin", "Spin (°)", -180, 180, 1)
            vuetify3.VDivider(classes="my-2")
            vuetify3.VSelect(
                v_model=("target",),
                items=("targets",),
                label="Target",
                hide_details=True,
                density="compact",
            )
            vuetify3.VBtn(
                "Set target",
                color="primary",
                click=on_set_target,
                classes="mt-2",
            )
            vuetify3.VDivider(classes="my-2")
            self._build_readouts()
            vuetify3.VDivider(classes="my-2")
            self._build_display_controls()
            vuetify3.VDivider(classes="my-2")
            if self.on_save is not None:
                vuetify3.VBtn(
                    "Save YAML",
                    color="success",
                    click=self.on_save,
                    classes="mt-2",
                    block=True,
                )
            if self.on_export_plan is not None:
                vuetify3.VBtn(
                    "Export plan",
                    color="primary",
                    click=self.on_export_plan,
                    classes="mt-2",
                    block=True,
                )

            # CCF overlay controls
            if self.ccf_overlay is not None:
                self._build_ccf_controls()

    def _slider_row(
        self,
        model: str,
        label: str,
        min_val: float,
        max_val: float,
        step: float,
    ) -> None:
        """Slider + editable number field bound to the same state variable."""
        with vuetify3.VRow(align="center", no_gutters=True, dense=True):
            with vuetify3.VCol():
                vuetify3.VSlider(
                    v_model=(model, 0),
                    min=min_val,
                    max=max_val,
                    step=step,
                    label=label,
                    hide_details=True,
                )
            with vuetify3.VCol(cols="auto"):
                vuetify3.VTextField(
                    v_model=(model, 0),
                    type="number",
                    step=step,
                    density="compact",
                    hide_details=True,
                    style="width:80px",
                )

    def _build_readouts(self) -> None:
        """Read-only geometric summary for the currently selected probe."""
        with vuetify3.VRow(no_gutters=True, dense=True):
            with vuetify3.VCol(cols=4):
                vuetify3.VLabel("Tip (RAS)")
            with vuetify3.VCol():
                vuetify3.VLabel("{{ probe_tip_str }}")
        with vuetify3.VRow(no_gutters=True, dense=True):
            with vuetify3.VCol(cols=4):
                vuetify3.VLabel("Depth (brain)")
            with vuetify3.VCol():
                vuetify3.VLabel("{{ probe_depth_str }}")
        with vuetify3.VRow(no_gutters=True, dense=True):
            with vuetify3.VCol(cols=4):
                vuetify3.VLabel("Over-inserted")
            with vuetify3.VCol():
                vuetify3.VLabel("{{ probe_overinsertion_str }}")

    def _build_display_controls(self) -> None:
        """Per-group visibility toggles + opacity sliders for the major
        asset categories. Probes are always visible."""
        vuetify3.VLabel("Display")
        for skey, label, _include, _exclude in VISIBILITY_GROUPS:
            with vuetify3.VRow(align="center", no_gutters=True, dense=True):
                with vuetify3.VCol(cols="auto"):
                    vuetify3.VSwitch(
                        v_model=(f"{skey}_visible",),
                        label=label,
                        hide_details=True,
                        density="compact",
                        inset=True,
                    )
                with vuetify3.VCol():
                    vuetify3.VSlider(
                        v_model=(f"{skey}_opacity",),
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        hide_details=True,
                        density="compact",
                    )

    def _build_ccf_controls(self) -> None:
        """Build the CCF region overlay UI widgets."""
        vuetify3.VDivider(classes="my-2")
        vuetify3.VLabel("CCF Regions")
        vuetify3.VAutocomplete(
            v_model=("ccf_selected_region",),
            items=("ccf_search_results", []),
            label="Search brain region",
            hide_details=True,
            density="compact",
            clearable=True,
            classes="mt-1",
            update_search=("ccf_search_query = $event"),
        )
        vuetify3.VSlider(
            v_model=("ccf_global_opacity", 0.3),
            min=0,
            max=1,
            step=0.05,
            label="CCF opacity",
            hide_details=True,
            classes="mt-2",
        )

    def _load_probe_state(self, state) -> None:
        """Sync trame state from the currently selected probe."""
        if not state.probe:
            return
        plan = self.store.state.probes.get(state.probe)
        if not plan:
            return

        r_mm, a_mm = plan.offsets_RA
        ap_tilt = (
            float(self.store.state.kinematics.arc_angles.get(plan.arc_id, 0.0))
            if plan.arc_id and plan.bind_ap_to_arc
            else float(plan.ap_local)
        )
        with state:
            state.offset_r = float(r_mm)
            state.offset_a = float(a_mm)
            state.depth = float(plan.past_target_mm)
            state.ap_tilt = ap_tilt
            state.ml_tilt = float(plan.ml_local)
            state.spin = int(round(float(plan.spin)))
            if plan.arc_id:
                state.arc = plan.arc_id
            if plan.target_key:
                state.target = plan.target_key
            state.probe_kind = plan.kind
        self._refresh_readouts(state, state.probe)
