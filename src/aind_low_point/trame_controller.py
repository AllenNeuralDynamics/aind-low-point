"""Trame + PyVista controller for probe manipulation.

Parallel implementation alongside the K3D / ipywidgets controller.
Calls store.dispatch() directly — PlanStore is the shared abstraction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pyvista as pv
import trimesh
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from aind_mri_utils.arc_angles import arc_angles_to_affine
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import client, vuetify3

from aind_low_point.assets import AssetCatalog
from aind_low_point.ccf_ontology import CCFOntology
from aind_low_point.ccf_overlay import CCFOverlayManager
from aind_low_point.collisions import CollisionHandler
from aind_low_point.commands import (
    AssignProbeArc,
    SetArcAngle,
    SetProbeCalibrated,
    SetProbeKind,
    SetProbeLocalAngles,
    SetProbeOffsetsRA,
    SetProbePastTarget,
    SetProbePositionBearingShank,
    SetProbeTarget,
)
from aind_low_point.core import MeshTransformable
from aind_low_point.planning import PoseResolver, ProbePose, kinematic_violations
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
# Kinematic-violation overlay: pairs of probes whose AP arcs are <16°
# apart, or pairs of probes on the same arc with ML <16° apart, can't
# physically be set up on the rig. Lower priority than over-insertion
# (the geometry is recoverable by retargeting; this isn't).
KINEMATIC_OVERLAY = OverlaySpec(
    color=0xFFD400,  # amber yellow
    alpha=0.7,
    source="kinematic",
    priority=20,
)


# Predefined visibility / opacity groups exposed in the UI. Each group is
# matched against a NodeInstance's ``tags`` set: a node is "in" the group
# if any of its tags appears here. Probes are excluded — they are the
# planning subjects and always visible.
# Gaming-style keyboard shortcuts. Mirrors the K3D controller's
# convention (WASD = R/A offsets, IJKL = tilts, UO = spin) and adds
# RF for depth + Tab/Shift+Tab to cycle probes + ? for help. Modifier
# multipliers match the K3D controller too: Shift = coarse 10×, Ctrl
# = fine 0.2×.
KB_ACTIONS: list[tuple[str, str, str]] = [
    # (key, action_id, display label)
    ("w", "a_inc", "+A offset"),
    ("s", "a_dec", "−A offset"),
    ("a", "r_dec", "−R offset"),
    ("d", "r_inc", "+R offset"),
    ("ArrowUp", "a_inc", "+A offset"),
    ("ArrowDown", "a_dec", "−A offset"),
    ("ArrowLeft", "r_dec", "−R offset"),
    ("ArrowRight", "r_inc", "+R offset"),
    # ``r`` and ``f`` are left for VTK.js's defaults (reset camera /
    # fly to point); depth uses the "elevator" convention with
    # ``e`` = extend deeper, ``q`` = retract shallower.
    ("e", "depth_inc", "Deeper (+depth)"),
    ("q", "depth_dec", "Shallower (−depth)"),
    ("i", "ap_inc", "AP tilt up"),
    ("k", "ap_dec", "AP tilt down"),
    ("j", "ml_dec", "ML tilt left"),
    ("l", "ml_inc", "ML tilt right"),
    ("u", "spin_dec", "Spin −"),
    ("o", "spin_inc", "Spin +"),
    ("Tab", "next_probe", "Next probe"),
    ("1", "speed_slow", "Slow speed"),
    ("2", "speed_normal", "Normal speed"),
    ("3", "speed_fast", "Fast speed"),
    ("c", "recenter", "Recenter on brain"),
    ("t", "focus_target", "Focus on target"),
    ("?", "help", "Toggle help"),
]
# Persistent speed-mode multipliers. Stack with Shift/Ctrl one-shot
# multipliers (Shift = ×10 coarse, Ctrl = ×0.2 fine), so the effective
# step is base × mode × modifier.
KB_SPEED_MULTIPLIER: dict[str, float] = {
    "slow": 0.5,
    "normal": 1.0,
    "fast": 5.0,
}
KB_SPEED_LABEL: dict[str, str] = {
    "slow": "Slow",
    "normal": "Normal",
    "fast": "Fast",
}
# Deduplicate keys for the JS-side fan-out (Tab and ArrowKeys need
# preventDefault; we send the lowercase key + modifier flags, the
# server picks the action.)
KB_KEYS_TO_INTERCEPT = sorted({k for k, *_ in KB_ACTIONS})


VISIBILITY_GROUPS: list[tuple[str, str, set[str], set[str]]] = [
    # (state-key, display label, tags-to-include, tags-to-exclude).
    # Groups are independent — a node ends up in every group whose
    # include set intersects its tags AND whose exclude set doesn't.
    # The exclude column is what keeps the implant slider distinct
    # from the rest of the fixtures (the implant has both "implant"
    # and "fixture" tags, so excluding "implant" from the broader
    # group is how we split it out without renaming any tags).
    #
    # Probes go first since they're what the user is positioning;
    # opacity edits persist via the node's material_override (the
    # renderer reads override before the asset's base material), so
    # subsequent probe pose updates don't reset the slider.
    ("probes", "Probes", {"probe"}, set()),
    ("brain", "Brain outline", {"brain"}, set()),
    ("structures", "CCF regions", {"structure"}, set()),
    ("implant", "Implant", {"implant"}, set()),
    ("fixtures", "Other fixtures", {"fixture", "headframe"}, {"implant"}),
]


def _as_float(value) -> float | None:
    """Coerce a trame state value to float, or None if blank/invalid.

    VTextField companions on numeric sliders briefly hold ``""`` while
    the user is editing; ``float("")`` raises. Handlers should skip
    dispatch on None and wait for the next change event.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ccf_color_for_target(catalog: AssetCatalog, target_key: str | None) -> str | None:
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
    on_save_plan: Callable[[], None] | None = None
    on_load_plan: Callable[[], None] | None = None
    # Per-key step sizes for keyboard shortcuts (multiplied by Shift=10×
    # for coarse, Ctrl=0.2× for fine — matches the K3D controller).
    kb_step_offset_mm: float = 0.05
    kb_step_depth_mm: float = 0.1
    kb_step_tilt_deg: float = 0.5
    kb_step_spin_deg: float = 1.0
    # Per-probe-asset shank tip positions in local mm — populated lazily
    # the first time a probe is checked for over-insertion. Independent
    # of pose, so this only depends on the asset's mesh.
    _shank_tips_cache: dict = field(default_factory=dict, repr=False)
    # Sets of node ids currently flagged by each warning overlay. Used
    # to detect transitions so we only repaint probes whose status
    # actually flipped.
    _prev_overinserted: set = field(default_factory=set, repr=False)
    _prev_kinematic: set = field(default_factory=set, repr=False)
    # Cached world-frame brain mesh used by ray-cast based depth and
    # over-insertion checks. The catalog's ``brain_spec.mesh.raw`` is in
    # the asset's pre-scene-node frame (e.g. an MRI NRRD's file frame);
    # the scene node may carry a ``transform: headframe_to_lps`` that
    # isn't applied to ``.raw``. We need the world-LPS mesh so the
    # ray cast lines up with the probe tips. Populated lazily on first
    # access; the brain's scene-node transform is static at runtime.
    _brain_world_mesh: object | None = field(default=None, repr=False)
    _brain_world_resolved: bool = field(default=False, repr=False)
    # Held so camera-only updates (recenter, focus-on-target) can call
    # ``ctrl.view_update`` to push the new camera state to the browser.
    # The renderer's normal flush only runs when actors change.
    _ctrl: object | None = field(default=None, repr=False)
    _last_click_time: float = field(default=0.0, repr=False)
    _scene_loaded_once: bool = field(default=False, repr=False)

    def build_app(self, server=None):
        """Build trame app with Vuetify3 UI + PyVista 3D view.

        Returns the trame server (call ``server.start()`` to launch).
        """
        server = server or get_server()
        state, ctrl = server.state, server.controller
        self._ctrl = ctrl

        self.render_adapter.overlays = self.overlays_resolver

        self._init_state(state)
        self._wire_handlers(state)
        # Set edge highlight on the initial probe BEFORE the view widget
        # is created, so the first scene serialization picks it up. Skip
        # the view-update flush here — ctrl.view_update isn't assigned
        # until _build_layout runs.
        nid = f"probe:{state.probe}" if state.probe else None
        self.render_adapter.backend.highlight(nid)
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
            k.split(":", 1)[1] for k in self.assets.assets if k.startswith("probe:")
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
            # Number of shanks for the current probe (1 for single-shank
            # kinds, 4 for quadbase). Drives the shank-selector
            # dropdown's items list and visibility.
            state.probe_shank_options = [1]
            state.probe_position_bearing_shank = 1

            # Keyboard fan-out:
            # ``state.kb_event`` is set by the JS keydown listener to a
            # fresh dict on every keypress; the server-side change
            # handler reads (key, shift, ctrl) and dispatches.
            state.kb_event = {"key": "", "shift": False, "ctrl": False, "t": 0}
            state.kb_help_open = False
            state.kb_speed = "normal"  # slow / normal / fast — see KB_SPEED_MULTIPLIER

            # Current tab in the left control column. Pose-editing is
            # the default since it's what most slider/keyboard work
            # touches; switching to Readouts / Display / Files surfaces
            # less-frequently-used controls without making the column
            # scroll-only.
            state.ctrl_tab = "pose"

            # Read-only geometric readouts for the currently selected probe.
            state.probe_tip_str = "—"
            state.probe_depth_str = "—"
            state.probe_overinsertion_str = "—"
            state.probe_kinematic_str = "—"
            # Collision readouts. ``probe_collision_str`` shows what the
            # currently selected probe is touching (local view); the
            # scene-wide counter on the help dialog / status bar lives
            # in ``scene_collision_str``.
            state.probe_collision_str = "—"
            state.scene_collision_str = "—"
            # Plan-file upload (drives the "Load plan" file picker).
            # Trame's VFileInput writes a dict ``{name, size, content,
            # type}`` once a file is selected; the change handler reads
            # the bytes, validates, applies, then resets to None.
            state.plan_file = None
            state.plan_load_status = ""

            # Visibility / opacity per asset group. Initial values are taken
            # from each group's first node so the UI matches what's drawn.
            for skey, _label, include, exclude in VISIBILITY_GROUPS:
                members = self._nodes_with_any_tag(include, exclude)
                if not members:
                    setattr(state, f"{skey}_visible", True)
                    setattr(state, f"{skey}_opacity", 1.0)
                    continue
                first = members[0]
                base_mat = (
                    first.material_override
                    or self.assets.get_spec(first.asset_key).default_material
                )
                setattr(state, f"{skey}_visible", bool(base_mat.visible))
                setattr(state, f"{skey}_opacity", float(base_mat.opacity))

            state.offset_r = 0.0
            state.offset_a = 0.0
            state.depth = 0.0
            state.ap_tilt = 0.0
            state.ml_tilt = 0.0
            state.spin = 0

            # Calibration / NewScale UI state. ``probe_has_calibration``
            # gates the entire NewScale section; ``probe_calibrated``
            # mirrors ``plan.calibrated`` and drives the toggle. Inputs
            # are pre-zeroed; readouts update on probe/plan changes.
            state.probe_has_calibration = False
            state.probe_calibrated = False
            state.probe_newscale_apply_x = 0.0
            state.probe_newscale_apply_y = 0.0
            state.probe_newscale_apply_z = 0.0
            state.probe_newscale_readout_x = "—"
            state.probe_newscale_readout_y = "—"
            state.probe_newscale_readout_z = "—"

        # Centering is built into the kinematic chain: any probe with
        # ``past_target_mm = 0`` and ``offsets_RA = (0, 0)`` already has
        # the recording-array center on the target. No init-time fix-up
        # needed.

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
        # Chain a readout-refresh onto the collision handler's existing
        # on_state_changed callback so collision-driven readouts update
        # without us polling. The collision worker thread can call this
        # at any time; ``with state:`` handles the trame side.
        prev_coll_cb = getattr(self.collision_handler, "on_state_changed", None)
        self.collision_handler.on_state_changed = lambda cs, flips, plan: (
            self._on_collision_state_changed(
                cs,
                flips,
                plan,
                chained_callback=prev_coll_cb,
            )
        )
        # Run an initial over-insertion + kinematic pass for every probe
        # so the overlays reflect the loaded plan from the moment the
        # GUI comes up (rather than waiting for the first dispatch).
        self._refresh_overinsertion_overlay(list(probe_names))
        self._refresh_kinematic_overlay()

        # CCF overlay state
        if self.ccf_overlay is not None:
            state.ccf_search_query = ""
            state.ccf_search_results = []
            state.ccf_selected_region = None
            state.ccf_visible_regions = []
            state.ccf_global_opacity = self.ccf_overlay.global_opacity

    def _wire_handlers(self, state) -> None:  # noqa: C901
        """Register trame reactive handlers."""

        @state.change("probe")
        def on_probe_change(**kwargs):
            self._load_probe_state(state)
            self._update_probe_highlight(state)

        @state.change("offset_r", "offset_a")
        def on_offsets(**kwargs):
            if not state.probe:
                return
            r = _as_float(state.offset_r)
            a = _as_float(state.offset_a)
            if r is None or a is None:
                return
            self.store.dispatch(SetProbeOffsetsRA(name=state.probe, R_mm=r, A_mm=a))

        @state.change("ap_tilt")
        def on_ap(**kwargs):
            if not state.probe:
                return
            ap = _as_float(state.ap_tilt)
            if ap is None:
                return
            self._on_ap_change(state, state.probe, ap)

        @state.change("ml_tilt")
        def on_ml(**kwargs):
            if not state.probe:
                return
            ml = _as_float(state.ml_tilt)
            if ml is None:
                return
            self._on_ml_change(state, state.probe, ml)

        @state.change("depth")
        def on_depth(**kwargs):
            if not state.probe:
                return
            depth = _as_float(state.depth)
            if depth is None:
                return
            self.store.dispatch(
                SetProbePastTarget(name=state.probe, past_target_mm=depth)
            )

        @state.change("spin")
        def on_spin(**kwargs):
            if not state.probe:
                return
            spin = _as_float(state.spin)
            if spin is None:
                return
            self.store.dispatch(SetProbeLocalAngles(name=state.probe, spin=spin))

        @state.change("arc")
        def on_arc_assign(**kwargs):
            self._on_arc_assign(state)

        @state.change("probe_kind")
        def on_kind_change(**kwargs):
            if not state.probe or not state.probe_kind:
                return
            self._on_probe_kind_change(state.probe, str(state.probe_kind))

        @state.change("probe_position_bearing_shank")
        def on_position_bearing_shank(**kwargs):
            if not state.probe:
                return
            try:
                idx = int(state.probe_position_bearing_shank)
            except (TypeError, ValueError):
                return
            plan = self.store.state.probes.get(state.probe)
            if plan is None or plan.position_bearing_shank == idx:
                return
            self.store.dispatch(
                SetProbePositionBearingShank(
                    name=state.probe, position_bearing_shank=idx
                )
            )

        @state.change("probe_calibrated")
        def on_probe_calibrated(**_):
            if not state.probe:
                return
            plan = self.store.state.probes.get(state.probe)
            if plan is None:
                return
            want = bool(state.probe_calibrated)
            if plan.calibrated == want:
                return
            # Only allow toggling on when a calibration is actually loaded.
            cal_present = state.probe in self.store.state.calibrations
            if want and not cal_present:
                state.probe_calibrated = False
                return
            self.store.dispatch(SetProbeCalibrated(name=state.probe, calibrated=want))
            # Re-sync slider readouts: flipping calibration on/off changes
            # the resolved AP/ML (locked to calibration vs free).
            self._load_probe_state(state)

        for skey, _label, include, exclude in VISIBILITY_GROUPS:
            self._wire_visibility_handlers(state, skey, include, exclude)

        @state.change("kb_event")
        def _on_kb(**_):
            ev = state.kb_event or {}
            key = ev.get("key") or ""
            if key:
                self._handle_kb(key, bool(ev.get("shift")), bool(ev.get("ctrl")))

        @state.change("plan_file")
        def _on_plan_file(**_):
            # VFileInput writes either a dict (single) or list-of-dicts
            # (multiple files). Each entry has at least
            # ``{name, size, content}`` where ``content`` is bytes.
            payload = state.plan_file
            if payload is None:
                return
            entry = payload[0] if isinstance(payload, list) else payload
            if not entry:
                return
            content = entry.get("content") if isinstance(entry, dict) else None
            name = entry.get("name") if isinstance(entry, dict) else "?"
            if content is None:
                state.plan_load_status = f"⚠ '{name}': no content"
                return
            try:
                self._apply_plan_yaml_bytes(content)
                state.plan_load_status = f"✓ Loaded {name}"
            except Exception as exc:
                state.plan_load_status = f"⚠ {name}: {exc}"
            # Reset so the same file can be picked again without
            # toggling between two files first.
            state.plan_file = None

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
            opacity = _as_float(state.ccf_global_opacity)
            if opacity is None:
                return
            ccf.set_global_opacity(opacity)

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
        if plan is None:
            return
        # When the probe is locked to its calibration, AP comes from
        # find_probe_angle(cal.rotation) — silently ignore slider events
        # so the stored arc/ap_local is preserved for un-toggling later.
        if plan.calibrated and probe in self.store.state.calibrations:
            return
        if plan.arc_id and plan.bind_ap_to_arc:
            self.store.dispatch(SetArcAngle(arc_id=plan.arc_id, ap_deg=ap))
        else:
            self.store.dispatch(SetProbeLocalAngles(name=probe, ap_local=ap))

    def _on_ml_change(self, state, probe: str, ml: float) -> None:
        plan = self.store.state.probes.get(probe)
        if not plan:
            return
        # Same as AP: when calibrated, ML is locked to the calibration;
        # don't mutate stored ml_local.
        if plan.calibrated and probe in self.store.state.calibrations:
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

    def _compute_probe_readouts(self, probe_name: str) -> tuple[str, str, str]:
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
        pose = ProbePose.from_planning_state(
            self.store.state, probe_name, catalog=self.assets
        )
        R = arc_angles_to_affine(pose.ap, pose.ml, pose.spin)
        # Read out the position-bearing shank's tip, not shank-1's.
        # ``pose.tip`` is the world position of the canonical-local
        # origin (= shank-1 in the AIND canonicalization); the named
        # shank's tip is offset by ``R @ shank_tips_local[N-1]``.
        local_tips = self._shank_tips_local(f"probe:{plan.kind}")
        named_idx = max(0, int(plan.position_bearing_shank) - 1)
        named_idx = min(named_idx, max(0, len(local_tips) - 1))
        named_local = (
            np.asarray(local_tips[named_idx], dtype=np.float64)
            if len(local_tips) > 0
            else np.zeros(3, dtype=np.float64)
        )
        named_world_lps = np.asarray(pose.tip, dtype=np.float64) + R @ named_local
        tip_ras = convert_coordinate_system(named_world_lps, "LPS", "RAS")
        tip_str = f"{tip_ras[0]:+.2f}, {tip_ras[1]:+.2f}, {tip_ras[2]:+.2f} mm"

        depth_str = "—"
        overins_str = "—"
        brain_mesh = self._get_brain_world_mesh()
        if brain_mesh is not None:
            probe_axis = R @ np.array([0.0, 0.0, 1.0])
            # Depth is the distance from the named shank's tip down to
            # the nearest brain-surface intersection along the shaft.
            depth = _depth_along_probe_axis(named_world_lps, probe_axis, brain_mesh)
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

    def _get_brain_world_mesh(self):
        """Return the brain mesh in world LPS, or None if no brain asset.

        The catalog's ``brain_spec.mesh.raw`` is in the asset's
        pre-scene-node frame — for an MRI NRRD that's the file's native
        frame, which may sit ~tens of mm off LPS world if the scene
        node carries a ``transform: headframe_to_lps``. Ray casts that
        use ``.raw`` directly compare a probe tip in LPS world against
        a brain mesh in NRRD-file frame → garbage. Apply the scene
        transform once and cache."""
        if self._brain_world_resolved:
            return self._brain_world_mesh
        scene = self.render_adapter.scene
        brain_id = None
        for k, n in scene.nodes.items():
            if n.asset_key == "brain":
                brain_id = k
                break
        if brain_id is None:
            self._brain_world_resolved = True
            return None
        from aind_low_point.scene import resolve_base_geometry

        wrap = resolve_base_geometry(self.assets, scene, brain_id)
        self._brain_world_mesh = wrap.raw if wrap is not None else None
        self._brain_world_resolved = True
        return self._brain_world_mesh

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

    # Note: ``_compute_target_centered_pose`` and the init-time
    # centering helper used to live here. Both are obsolete after the
    # pivot redesign — ``ProbePose.from_planning_state`` now subtracts
    # ``R @ recording_center_local`` from ``tip``, so any
    # ``(past_target_mm=0, offsets_RA=(0, 0))`` pose automatically
    # places the recording-array center on the target. Setting a target
    # is just a target dispatch followed by a state reset of those
    # variables — no manual offset/depth math needed.

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

    def _format_pair_other(self, this_nid: str, pair: tuple[str, str]) -> str:
        """Return the *other* side of a collision pair, formatted for the
        local-collision readout. Probe nodes lose their ``probe:`` prefix
        so the cell reads like a list of probe names; non-probe nodes
        (``implant``, ``headframe``, ...) pass through unchanged."""
        other = pair[1] if pair[0] == this_nid else pair[0]
        if other.startswith("probe:"):
            return other.split(":", 1)[1]
        return other

    def _compute_collision_strs(self, probe_name: str) -> tuple[str, str]:
        """Return ``(this-probe collisions, scene-wide collisions)`` as
        display strings.

        - ``this-probe`` lists the *other* side of each pair the current
          probe participates in, comma-separated. ``"✓"`` when clear.
        - ``scene`` is a counter ``"N pair(s)"`` over all colliding
          objects (or ``"✓"`` when clear) — gives a quick sense of
          whether problems exist elsewhere in the plan that the local
          probe-view doesn't surface.
        """
        coll_state = getattr(self.collision_handler, "state", None)
        if coll_state is None or not coll_state.pairs:
            return ("✓", "✓")
        this_nid = f"probe:{probe_name}" if probe_name else None
        local_others: list[str] = []
        for pair in coll_state.pairs:
            if this_nid is not None and this_nid in pair:
                local_others.append(self._format_pair_other(this_nid, pair))
        local_str = "⚠ " + ", ".join(sorted(set(local_others))) if local_others else "✓"
        n_pairs = len(coll_state.pairs)
        scene_str = f"⚠ {n_pairs} pair{'s' if n_pairs != 1 else ''}"
        return (local_str, scene_str)

    def _refresh_readouts(self, state, probe_name: str) -> None:
        tip_str, depth_str, overins_str = self._compute_probe_readouts(probe_name)
        kin_str = self._kinematic_status_for_probe(probe_name)
        coll_local, coll_scene = self._compute_collision_strs(probe_name)
        has_cal, is_cal, nx, ny, nz = self._compute_newscale_readout(probe_name)
        with state:
            state.probe_tip_str = tip_str
            state.probe_depth_str = depth_str
            state.probe_overinsertion_str = overins_str
            state.probe_kinematic_str = kin_str
            state.probe_collision_str = coll_local
            state.scene_collision_str = coll_scene
            state.probe_has_calibration = has_cal
            state.probe_calibrated = is_cal
            state.probe_newscale_readout_x = nx
            state.probe_newscale_readout_y = ny
            state.probe_newscale_readout_z = nz

    def _apply_newscale_to_probe(self, state) -> None:
        """Convert the NewScale-input ``(x, y, z)`` to a subject-LPS target
        and dispatch commands so the position-bearing shank's tip lands
        there with ``past_target_mm = 0`` and zero offsets.

        Math (per :func:`ProbePose.from_planning_state`):
        ``pose.tip = adjusted_target − R @ pivot_local``,
        position-bearing shank world ≈ ``pose.tip + R @ shank_pb_local``.
        With ``past_target=0`` and offsets zero, setting the target so
        that ``adjusted_target = newscale_lps + R @ (pivot_local −
        shank_pb_local)`` lands shank_pb at ``newscale_lps``.

        Silently no-ops when the probe isn't calibrated.
        """
        from aind_anatomical_utils.coordinate_systems import (
            convert_coordinate_system,
        )

        from aind_low_point.calibration_conversion import newscale_to_lps
        from aind_low_point.optimization.recording import (
            recording_center_local_for_kind,
        )

        if not state.probe:
            return
        plan = self.store.state.probes.get(state.probe)
        if plan is None or not plan.calibrated:
            return
        cal = self.store.state.calibrations.get(state.probe)
        if cal is None:
            return
        try:
            x = float(state.probe_newscale_apply_x)
            y = float(state.probe_newscale_apply_y)
            z = float(state.probe_newscale_apply_z)
        except (TypeError, ValueError):
            return
        newscale_xyz = np.array([x, y, z], dtype=np.float64)
        pb_world_lps = newscale_to_lps(newscale_xyz, cal)

        # Resolve probe rotation R from the (already locked-by-cal) pose.
        pose = ProbePose.from_planning_state(
            self.store.state, state.probe, catalog=self.assets
        )
        R = pose.transform().rotation

        tips_local = self._shank_tips_local(f"probe:{plan.kind}")
        pb_idx = max(0, int(plan.position_bearing_shank) - 1)
        if len(tips_local) > 0:
            pb_local = np.asarray(
                tips_local[min(pb_idx, len(tips_local) - 1)], dtype=np.float64
            )
        else:
            pb_local = np.zeros(3, dtype=np.float64)
        # Pivot in canonical local frame — same source as
        # ProbePose.from_planning_state's tip computation.
        asset_key = f"probe:{plan.kind}"
        spec = self.assets.assets.get(asset_key)
        if spec is not None and spec.pivot_LPS is not None:
            pivot_local = np.asarray(spec.pivot_LPS, dtype=np.float64)
        else:
            pivot_local = recording_center_local_for_kind(plan.kind)

        target_lps = pb_world_lps + R @ (pivot_local - pb_local)
        target_ras = convert_coordinate_system(
            target_lps.reshape(1, 3), "LPS", "RAS"
        ).reshape(3)

        self.store.dispatch(
            SetProbeTarget(
                name=state.probe,
                target_point_RAS=(
                    float(target_ras[0]),
                    float(target_ras[1]),
                    float(target_ras[2]),
                ),
            )
        )
        self.store.dispatch(SetProbePastTarget(name=state.probe, past_target_mm=0.0))
        self.store.dispatch(SetProbeOffsetsRA(name=state.probe, R_mm=0.0, A_mm=0.0))

    def _compute_newscale_readout(
        self, probe_name: str
    ) -> tuple[bool, bool, str, str, str]:
        """Compute calibration availability + NewScale (x, y, z) the rig
        should read for the currently planned pose of ``probe_name``.

        Returns (has_calibration, is_calibrated, nx, ny, nz). The three
        x/y/z strings are formatted with 3 decimal places, or "—" when
        no calibration is available.
        """
        plan = self.store.state.probes.get(probe_name)
        if plan is None:
            return False, False, "—", "—", "—"
        cal = self.store.state.calibrations.get(probe_name)
        has = cal is not None
        is_cal = bool(plan.calibrated and has)
        if not has:
            return False, bool(plan.calibrated), "—", "—", "—"
        try:
            from aind_low_point.calibration_conversion import lps_to_newscale
            from aind_low_point.planning import ProbePose

            pose = ProbePose.from_planning_state(
                self.store.state, probe_name, catalog=self.assets
            )
            tips_local = self._shank_tips_local(f"probe:{plan.kind}")
            pb_idx = max(0, int(plan.position_bearing_shank) - 1)
            if len(tips_local) > 0:
                pb_local = np.asarray(
                    tips_local[min(pb_idx, len(tips_local) - 1)],
                    dtype=np.float64,
                )
            else:
                pb_local = np.zeros(3, dtype=np.float64)
            R = pose.transform().rotation
            pb_world = np.asarray(pose.tip, dtype=np.float64) + R @ pb_local
            ns = lps_to_newscale(pb_world, cal)
            return (True, is_cal, f"{ns[0]:.3f}", f"{ns[1]:.3f}", f"{ns[2]:.3f}")
        except Exception:
            return has, is_cal, "—", "—", "—"

    def _on_collision_state_changed(
        self,
        coll_state,
        flips,
        plan,
        chained_callback=None,
    ) -> None:
        """Wrapper installed over the collision handler's
        ``on_state_changed``. Runs the original overlay-repaint callback
        first, then refreshes our collision readouts. Kept distinct from
        ``_refresh_readouts`` because collision changes fire from a
        worker thread; we forward to the main loop via the readout
        state (which is the trame server state)."""
        if chained_callback is not None:
            chained_callback(coll_state, flips, plan)
        state = self._readout_state
        if state is None:
            return
        probe_name = state.probe
        coll_local, coll_scene = self._compute_collision_strs(probe_name or "")
        with state:
            state.probe_collision_str = coll_local
            state.scene_collision_str = coll_scene

    def _handle_kb(self, key: str, shift: bool, ctrl: bool) -> None:
        """Dispatch a single keyboard shortcut. Step sizes are multiplied
        by 10× when Shift is held (coarse) and 0.2× when Ctrl is held
        (fine), matching the K3D controller's convention."""
        # Find the action for this key. We look up case-insensitively
        # for letter keys but exact-match for special keys (Tab, ?).
        match = next(
            (a for k, a, _ in KB_ACTIONS if k == key or k == key.lower()),
            None,
        )
        if match is None:
            return

        # Mode/help/probe-cycle actions don't depend on step sizes.
        state = self._readout_state
        if match == "help":
            state.kb_help_open = not state.kb_help_open
            return
        if match in ("speed_slow", "speed_normal", "speed_fast"):
            state.kb_speed = match.split("_", 1)[1]
            return
        if match == "recenter":
            self.recenter_view()
            return
        if match == "focus_target":
            self.focus_on_current_target()
            return
        if match == "next_probe":
            probes = sorted(self.store.state.probes.keys())
            if not probes:
                return
            cur = state.probe or probes[0]
            try:
                idx = probes.index(cur)
            except ValueError:
                idx = 0
            step = -1 if shift else 1
            state.probe = probes[(idx + step) % len(probes)]
            return

        if not state.probe:
            return

        # Modifier mul stacks on top of the persistent speed-mode mul.
        mode_mul = KB_SPEED_MULTIPLIER.get(state.kb_speed or "normal", 1.0)
        modifier_mul = 10.0 if shift else 0.2 if ctrl else 1.0
        mul = mode_mul * modifier_mul
        doff = self.kb_step_offset_mm * mul
        ddep = self.kb_step_depth_mm * mul
        dtilt = self.kb_step_tilt_deg * mul
        dspin = self.kb_step_spin_deg * mul

        # Sliders are bound by name to state. Mutate the trame state
        # variable; the existing @state.change handlers fire and
        # dispatch the underlying command. Means we don't reach into
        # the store directly and stay consistent with slider-driven
        # edits.
        deltas = {
            "a_inc": ("offset_a", +doff),
            "a_dec": ("offset_a", -doff),
            "r_inc": ("offset_r", +doff),
            "r_dec": ("offset_r", -doff),
            "depth_inc": ("depth", +ddep),
            "depth_dec": ("depth", -ddep),
            "ap_inc": ("ap_tilt", +dtilt),
            "ap_dec": ("ap_tilt", -dtilt),
            "ml_inc": ("ml_tilt", +dtilt),
            "ml_dec": ("ml_tilt", -dtilt),
            "spin_inc": ("spin", +dspin),
            "spin_dec": ("spin", -dspin),
        }
        var, delta = deltas.get(match, (None, 0.0))
        if var is None:
            return
        cur = float(getattr(state, var) or 0.0)
        setattr(state, var, cur + delta)

    def _kinematic_status_for_probe(self, probe_name: str) -> str:
        """Per-probe textual readout: 'OK' / '⚠ ML vs X[, Y]; AP-arc vs Z'."""
        plan = self.store.state.probes.get(probe_name)
        if plan is None:
            return "—"
        viols = kinematic_violations(self.store.state)
        ml_clashes: set[str] = set()
        for a, b in viols["within_arc_ml"]:
            if probe_name in (a, b):
                ml_clashes.add(b if a == probe_name else a)
        arc_clashes: set[str] = set()
        if plan.arc_id is not None:
            for a, b in viols["arc_ap"]:
                if plan.arc_id in (a, b):
                    arc_clashes.add(b if a == plan.arc_id else a)
        if not ml_clashes and not arc_clashes:
            return "OK"
        msgs: list[str] = []
        if ml_clashes:
            msgs.append(f"ML vs {','.join(sorted(ml_clashes))}")
        if arc_clashes:
            msgs.append(f"AP-arc vs {','.join(sorted(arc_clashes))}")
        return "⚠ " + "; ".join(msgs)

    def _refresh_kinematic_overlay(self) -> None:
        """Recompute kinematic violations and update the shared overlay
        state. Repaints only probes whose membership flipped."""
        viols = kinematic_violations(self.store.state)
        affected: set[str] = set()
        for arc_a, arc_b in viols["arc_ap"]:
            for name, plan in self.store.state.probes.items():
                if plan.arc_id in (arc_a, arc_b):
                    affected.add(f"probe:{name}")
        for a, b in viols["within_arc_ml"]:
            affected.add(f"probe:{a}")
            affected.add(f"probe:{b}")
        flips = list(self._prev_kinematic ^ affected)
        self._prev_kinematic = affected
        overlays_state = self.overlays_resolver.overlays
        overlays_state.clear_source("kinematic")
        if affected:
            overlays_state.set_for_source(list(affected), KINEMATIC_OVERLAY)
        if flips:
            self.render_adapter.repaint_materials(flips)

    def _refresh_overinsertion_overlay(self, changed_ids) -> None:
        """For every probe in *changed_ids*, recompute over-insertion and
        update the shared overlay state. Repaints only the probes whose
        over-insertion status flipped."""
        brain_mesh = self._get_brain_world_mesh()
        if brain_mesh is None:
            return
        flips: list[str] = []
        for probe_name in changed_ids:
            if probe_name not in self.store.state.probes:
                continue
            pose = ProbePose.from_planning_state(
                self.store.state, probe_name, catalog=self.assets
            )
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
        """PlanStore subscriber: refresh readouts + over-insertion +
        kinematic overlays when probe poses change. The overlay updates
        cover all probes (an arc change can ripple across several); the
        textual readout only tracks the active probe shown in the UI."""
        state = getattr(self, "_readout_state", None)
        if state is None:
            return
        self._refresh_overinsertion_overlay(changed_ids)
        self._refresh_kinematic_overlay()
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
        self.collision_handler.adapter.on_store_change(self.store.state, [probe_name])

        # Re-run collision detection now that the BVH for this probe is
        # the new mesh — without this, collision overlay still reflects
        # the OLD geometry until the user moves something. Same for the
        # over-insertion + kinematic overlays (the over-insertion check
        # depends on the new mesh's shank tips). __call__ runs the sync
        # path, fires on_state_changed, and the existing collision
        # callback updates the overlay state for any flips.
        self.collision_handler(self.store.state, [probe_name])
        self._refresh_overinsertion_overlay([probe_name])
        self._refresh_kinematic_overlay()

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
            # Reset depth/offsets to defaults so the kinematic chain's
            # auto-centering takes over (recording-array center lands
            # on the target). The user can dial deviations afterwards.
            self.store.dispatch(
                SetProbePastTarget(name=state.probe, past_target_mm=0.0)
            )
            self.store.dispatch(SetProbeOffsetsRA(name=state.probe, R_mm=0.0, A_mm=0.0))
            state.depth = 0.0
            state.offset_r = 0.0
            state.offset_a = 0.0
            self._apply_target_based_color(state.probe)

        with SinglePageLayout(server) as layout:
            layout.title.set_text("Probe Planner")

            # Install a window-level keydown listener once on mount.
            # We can't bind to SinglePageLayout's root with v-on:keydown
            # (focus has to be on a focusable element for that to fire),
            # so we attach to the document. The listener skips events
            # whose target is an INPUT/TEXTAREA so the user can still
            # type into search fields. Each captured keypress writes a
            # fresh dict to the ``kb_event`` server state via
            # ``trame.state.set(...)``; the ``state.change("kb_event")``
            # handler then dispatches the action. NB: this string lands
            # inside a double-quoted HTML attribute, so every literal
            # inside the JS must use single quotes — embedded ``"``
            # would terminate the attribute and break Vue templating.
            # NB: Vue 3's template compiler does not expose ``document``
            # as a global to v-on inline handlers (it resolves to
            # ``undefined``), but ``window`` is reachable via its proxy
            # context. Route DOM access through ``window.document``.
            #
            # Sentinel name is bumped (``_aind_kb_listener_v3``) so that
            # any stale listener from a previous attempt in the same
            # browser session does not block re-installation. Wrapped
            # in try/catch so the next failure is loud in the browser
            # console rather than silently leaving the keyboard inert.
            #
            # ``stopImmediatePropagation`` (in addition to
            # ``preventDefault``) is required because VTK.js's render
            # window interactor registers its own keydown handler that
            # binds e.g. 'r' to ResetCamera. Without stopping
            # propagation, our handler absorbs the application
            # action and VTK still fires its own — so 'r' both deepens
            # the probe (intended) and resets the camera (a bug). Our
            # listener runs in the capture phase on ``document``, so
            # stopping propagation here halts the event before it
            # reaches descendant elements like the render canvas.
            kb_keys_js = ",".join(f"'{k}'" for k in KB_KEYS_TO_INTERCEPT)
            client.ClientTriggers(
                mounted=(
                    "(function(){"
                    "if (window._aind_kb_listener_v3) return;"
                    "try {"
                    "  const trapped = new Set([" + kb_keys_js + "]);"
                    "  window.document.addEventListener('keydown', function(e){"
                    "    const tag = e.target && e.target.tagName;"
                    "    if (tag === 'INPUT' || tag === 'TEXTAREA' || "
                    "(e.target && e.target.isContentEditable)) return;"
                    "    if (!trapped.has(e.key) && "
                    "!trapped.has(e.key.toLowerCase())) return;"
                    "    trame.state.set('kb_event', {"
                    "      key: e.key, shift: e.shiftKey, "
                    "ctrl: e.ctrlKey, t: Date.now()"
                    "    });"
                    "    e.preventDefault();"
                    "    e.stopImmediatePropagation();"
                    "  }, true);"
                    "  window._aind_kb_listener_v3 = true;"
                    "} catch (err) {"
                    "  window.console && window.console.error("
                    "'aind keyboard listener install failed:', err);"
                    "}"
                    "})();"
                )
            )
            # Silence the benign ``ResizeObserver loop completed with
            # undelivered notifications`` / ``ResizeObserver loop limit
            # exceeded`` error that Vuetify + trame's render-window
            # sizing fires whenever an observed element schedules a
            # second resize in the same frame. It's a W3C-spec
            # informational notice (not a real failure) but bubbles to
            # ``window.error`` in Chrome / Firefox and clutters the
            # console. Suppress that *one* message; let everything else
            # through.
            client.ClientTriggers(
                mounted=(
                    "(function(){"
                    "if (window._aind_resize_obs_filter) return;"
                    "const SILENCED = ["
                    "  'ResizeObserver loop completed with undelivered notifications.',"
                    "  'ResizeObserver loop limit exceeded'"
                    "];"
                    "window.addEventListener('error', function(e){"
                    "  if (e && e.message && SILENCED.some(function(m){ "
                    "return e.message.indexOf(m) !== -1; })) {"
                    "    e.stopImmediatePropagation();"
                    "    e.preventDefault();"
                    "  }"
                    "}, true);"
                    "window._aind_resize_obs_filter = true;"
                    "})();"
                )
            )

            with layout.content:
                with vuetify3.VContainer(fluid=True, classes="fill-height"):
                    with vuetify3.VRow(classes="fill-height"):
                        self._build_controls(on_set_target)
                        with vuetify3.VCol(cols=9, classes="fill-height"):
                            view = plotter_ui(
                                self.plotter,
                                mode="client",
                                picking_modes=("['click']",),
                                click=(self._on_view_click, "[$event]"),
                                after_scene_loaded=self._on_scene_loaded,
                            )
                            ctrl.view_update = view.update

            # Help dialog — bound to state.kb_help_open (toggled by '?').
            self._build_kb_help_dialog()

    def _build_controls(self, on_set_target) -> None:
        """Build the left-column control widgets.

        Organized into four tabs so the column doesn't outgrow the
        viewport:

        - **Pose** — probe/arc/type selection, R/A/depth + AP/ML/spin
          sliders, target picker, *and* the per-probe readouts +
          keyboard-speed mode that you read off while editing.
        - **Display** — scene visibility toggles + CCF overlay.
        - **Files** — save / export / plan-IO buttons.
        """
        with vuetify3.VCol(cols=3, classes="fill-height overflow-y-auto"):
            with vuetify3.VTabs(
                v_model=("ctrl_tab",),
                density="compact",
                grow=True,
            ):
                vuetify3.VTab(text="Pose", value="pose")
                vuetify3.VTab(text="Display", value="display")
                vuetify3.VTab(text="Files", value="files")
            with vuetify3.VTabsWindow(v_model=("ctrl_tab",), classes="mt-2"):
                with vuetify3.VTabsWindowItem(value="pose"):
                    self._build_pose_tab(on_set_target)
                    vuetify3.VDivider(classes="my-2")
                    self._build_readouts()
                    vuetify3.VDivider(classes="my-2")
                    self._build_kb_speed_controls()
                with vuetify3.VTabsWindowItem(value="display"):
                    self._build_display_controls()
                    if self.ccf_overlay is not None:
                        vuetify3.VDivider(classes="my-2")
                        self._build_ccf_controls()
                with vuetify3.VTabsWindowItem(value="files"):
                    self._build_files_tab()

    def _build_pose_tab(self, on_set_target) -> None:
        """Probe/arc/type + sliders + target — the pose-editing flow."""
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
        # Hidden when the probe has only one shank — the dropdown would
        # just show "1" and it'd waste a row.
        vuetify3.VSelect(
            v_model=("probe_position_bearing_shank",),
            items=("probe_shank_options",),
            label="Reported shank",
            hide_details=True,
            density="compact",
            v_show=("probe_shank_options.length > 1",),
            classes="mt-1",
        )
        vuetify3.VDivider(classes="my-2")
        self._slider_row("offset_r", "R (mm)", -7.5, 7.5, 0.05)
        self._slider_row("offset_a", "A (mm)", -7.5, 7.5, 0.05)
        self._slider_row("depth", "Depth (mm)", -10, 10, 0.1)
        vuetify3.VDivider(classes="my-2")
        self._slider_row(
            "ap_tilt", "AP tilt (°)", -60, 60, 0.5, disabled="probe_calibrated"
        )
        self._slider_row(
            "ml_tilt", "ML tilt (°)", -60, 60, 0.5, disabled="probe_calibrated"
        )
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
        self._build_newscale_section()

    def _build_newscale_section(self) -> None:
        """NewScale machine-coords ↔ probe-pose tools.

        The whole section is shown only when a calibration is loaded for
        the current probe. The Apply inputs + button are disabled when
        the probe isn't currently using calibration (so AP/ML aren't
        locked to the cal). Readouts show the inverse conversion of the
        current pose's position-bearing shank tip.
        """
        with vuetify3.VCard(classes="mt-2", flat=True):
            vuetify3.VCardSubtitle("NewScale")
            with vuetify3.VCardText():
                vuetify3.VSwitch(
                    v_model=("probe_calibrated",),
                    label="Use calibration (lock AP/ML)",
                    hide_details=True,
                    density="compact",
                    disabled=("!probe_has_calibration",),
                )
                vuetify3.VLabel(
                    "Calibrated shank = 'Reported shank' above.",
                    classes="text-caption text-medium-emphasis mb-2",
                )
                vuetify3.VDivider(classes="my-2")
                vuetify3.VLabel(
                    "Apply reading (probe-frame mm) → tip",
                    classes="text-caption mb-1",
                )
                with vuetify3.VRow(dense=True, classes="mb-1"):
                    with vuetify3.VCol(cols=4):
                        vuetify3.VTextField(
                            v_model_number=("probe_newscale_apply_x",),
                            label="x",
                            type="number",
                            step=0.01,
                            density="compact",
                            hide_details=True,
                            disabled=("!probe_calibrated",),
                        )
                    with vuetify3.VCol(cols=4):
                        vuetify3.VTextField(
                            v_model_number=("probe_newscale_apply_y",),
                            label="y",
                            type="number",
                            step=0.01,
                            density="compact",
                            hide_details=True,
                            disabled=("!probe_calibrated",),
                        )
                    with vuetify3.VCol(cols=4):
                        vuetify3.VTextField(
                            v_model_number=("probe_newscale_apply_z",),
                            label="z",
                            type="number",
                            step=0.01,
                            density="compact",
                            hide_details=True,
                            disabled=("!probe_calibrated",),
                        )
                vuetify3.VBtn(
                    "Apply NewScale → tip",
                    color="primary",
                    density="compact",
                    disabled=("!probe_calibrated",),
                    click=self._on_apply_newscale_click,
                )
                vuetify3.VDivider(classes="my-2")
                vuetify3.VLabel(
                    "Current pose NewScale readout",
                    classes="text-caption mb-1",
                )
                vuetify3.VLabel(
                    "x: {{ probe_newscale_readout_x }}   "
                    "y: {{ probe_newscale_readout_y }}   "
                    "z: {{ probe_newscale_readout_z }}",
                    classes="text-body-2",
                )

    def _on_apply_newscale_click(self) -> None:
        """Server-side click handler for the Apply NewScale button."""
        if self._readout_state is None:
            return
        self._apply_newscale_to_probe(self._readout_state)
        # Refresh readouts to reflect the new pose.
        self._refresh_readouts(self._readout_state, self._readout_state.probe)

    def _apply_plan_yaml_bytes(self, content) -> None:
        """Parse plan-only YAML bytes from a browser upload and apply.

        Mirrors the path-based ``on_load_plan`` callback from app.py but
        takes bytes directly so it works with VFileInput uploads. Raises
        on parse / validation failures so the change handler can surface
        the error to the user via ``plan_load_status``.
        """
        import yaml

        from aind_low_point.config import PlanningModel
        from aind_low_point.runtime.export import apply_plan_model_to_state

        if isinstance(content, (bytes, bytearray)):
            text = content.decode("utf-8")
        else:
            text = str(content)
        raw = yaml.safe_load(text)
        loaded = PlanningModel.model_validate(raw)
        apply_plan_model_to_state(loaded, self.store)

    def _build_files_tab(self) -> None:
        """Save/export/plan-IO actions — grouped on their own tab so the
        Pose tab stays focused on edits and the row of save buttons
        doesn't crowd the slider stack.

        Verb convention: ``Save`` = re-importable file (this tool can
        reopen it). ``Export`` = derived hand-off (rig technician,
        downstream pipeline; not re-importable). ``Save plan`` and
        ``Load plan`` are deliberately adjacent — they're the
        plan-slice round-trip pair.
        """
        # Save config: server-side write of the full ConfigModel YAML
        # (paths, transforms, asset catalog, scene, options, plan).
        # Re-importable.
        if self.on_save is not None:
            vuetify3.VBtn(
                "Save config",
                color="success",
                click=self.on_save,
                classes="mt-2",
                block=True,
            )
        # Save plan + Load plan: the plan-only YAML round-trip pair.
        # Save: server-side write to the configured plan path. Load: a
        # browser file picker (VFileInput) — uploads the picked YAML to
        # the server, our state.change handler parses + applies. The
        # plan-only YAML has no asset list, so it ports across configs
        # that share probe rosters.
        if self.on_save_plan is not None:
            vuetify3.VBtn(
                "Save plan",
                color="info",
                click=self.on_save_plan,
                classes="mt-2",
                block=True,
            )
        vuetify3.VFileInput(
            v_model=("plan_file",),
            label="Load plan",
            accept=".yml,.yaml",
            show_size=True,
            truncate_length=20,
            density="compact",
            hide_details=True,
            classes="mt-2",
            prepend_icon="mdi-folder-open",
        )
        # Status line for the load (✓ Loaded foo.yml / ⚠ failure msg).
        vuetify3.VLabel(
            "{{ plan_load_status }}",
            v_show=("plan_load_status",),
            classes="mt-1 text-caption",
        )
        # Export poses: derived per-probe geometry hand-off (resolved
        # tip positions, depths, AP/ML/spin numbers). Not re-importable.
        if self.on_export_plan is not None:
            vuetify3.VBtn(
                "Export poses",
                color="primary",
                click=self.on_export_plan,
                classes="mt-2",
                block=True,
            )

    def _slider_row(
        self,
        model: str,
        label: str,
        min_val: float,
        max_val: float,
        step: float,
        disabled: str | None = None,
    ) -> None:
        """Slider + editable number field bound to the same state variable.

        ``disabled`` is an optional Vue expression (e.g. ``"probe_calibrated"``)
        used to gray out both controls when truthy.
        """
        kwargs = {"disabled": (disabled,)} if disabled else {}
        with vuetify3.VRow(align="center", no_gutters=True, dense=True):
            with vuetify3.VCol():
                vuetify3.VSlider(
                    v_model=(model, 0),
                    min=min_val,
                    max=max_val,
                    step=step,
                    label=label,
                    hide_details=True,
                    **kwargs,
                )
            with vuetify3.VCol(cols="auto"):
                vuetify3.VTextField(
                    v_model=(model, 0),
                    type="number",
                    step=step,
                    density="compact",
                    hide_details=True,
                    style="width:80px",
                    **kwargs,
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
        with vuetify3.VRow(no_gutters=True, dense=True):
            with vuetify3.VCol(cols=4):
                vuetify3.VLabel("Kinematic")
            with vuetify3.VCol():
                vuetify3.VLabel("{{ probe_kinematic_str }}")
        # Collisions split into local (this probe) and scene-wide.
        # Local lists the other side of each pair the current probe
        # touches; scene shows a total count so problems elsewhere
        # still surface without you having to switch probes to check.
        with vuetify3.VRow(no_gutters=True, dense=True):
            with vuetify3.VCol(cols=4):
                vuetify3.VLabel("Collide (this)")
            with vuetify3.VCol():
                vuetify3.VLabel("{{ probe_collision_str }}")
        with vuetify3.VRow(no_gutters=True, dense=True):
            with vuetify3.VCol(cols=4):
                vuetify3.VLabel("Collide (all)")
            with vuetify3.VCol():
                vuetify3.VLabel("{{ scene_collision_str }}")

    def _build_display_controls(self) -> None:
        """Per-group visibility toggles + opacity sliders for the major
        asset categories. Probes are always visible."""
        vuetify3.VLabel("Display")
        vuetify3.VBtn(
            "Recenter on brain",
            click=self.recenter_view,
            classes="mt-1 mb-2",
            block=True,
            variant="outlined",
            density="compact",
        )
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

    def _build_kb_speed_controls(self) -> None:
        """Speed-mode toggle for keyboard increments (1/2/3 hotkeys also
        switch this). Multiplies the base step; stacks with Shift/Ctrl
        modifiers."""
        with vuetify3.VRow(no_gutters=True, dense=True, align="center"):
            with vuetify3.VCol(cols=4):
                vuetify3.VLabel("Speed")
            with vuetify3.VCol():
                vuetify3.VBtnToggle(
                    v_model=("kb_speed",),
                    mandatory=True,
                    density="compact",
                    divided=True,
                    children=None,
                    hide_details=True,
                ).add_children(
                    [
                        vuetify3.VBtn("Slow", value="slow", size="small"),
                        vuetify3.VBtn("Norm", value="normal", size="small"),
                        vuetify3.VBtn("Fast", value="fast", size="small"),
                    ]
                )

    def _build_kb_help_dialog(self) -> None:
        """Modal that lists the keyboard shortcuts. Toggle with '?'."""
        # Drop duplicate action ids (e.g. arrow-key aliases of WASD).
        seen_actions: set[str] = set()
        rows: list[tuple[str, str]] = []
        for key, action_id, label in KB_ACTIONS:
            if action_id in seen_actions:
                continue
            seen_actions.add(action_id)
            rows.append((key, label))
        rows.append(("Shift + key", "10× step (coarse, stacks with mode)"))
        rows.append(("Ctrl + key", "0.2× step (fine, stacks with mode)"))

        with vuetify3.VDialog(v_model=("kb_help_open",), max_width="480"):
            with vuetify3.VCard():
                vuetify3.VCardTitle("Keyboard shortcuts")
                with vuetify3.VCardText():
                    with vuetify3.VList(density="compact"):
                        for key, label in rows:
                            with vuetify3.VListItem():
                                with vuetify3.VRow(no_gutters=True, align="center"):
                                    with vuetify3.VCol(cols="3"):
                                        vuetify3.VKbd(key)
                                    with vuetify3.VCol():
                                        vuetify3.VListItemTitle(label)
                with vuetify3.VCardActions():
                    vuetify3.VBtn(
                        "Close",
                        color="primary",
                        click="kb_help_open = false",
                    )

    def _update_probe_highlight(self, state) -> None:
        nid = f"probe:{state.probe}" if state.probe else None
        self.render_adapter.backend.highlight(nid)
        self._flush_view()

    def _on_scene_loaded(self, *args, **kwargs) -> None:
        # The client emits ``afterSceneLoaded`` on every scene push, but
        # we only need to force the initial-highlight delta once — when
        # the client first mounts. Without the off→on flip, the very
        # first scene snapshot reaches the client but vtk.js doesn't
        # apply ``edgeVisibility`` until a property delta arrives.
        if self._scene_loaded_once:
            return
        self._scene_loaded_once = True
        state = getattr(self, "_readout_state", None)
        if state is None or not state.probe:
            return
        nid = f"probe:{state.probe}"
        backend = self.render_adapter.backend
        backend.set_edge_highlight(nid, on=False)
        backend.set_edge_highlight(nid, on=True)
        backend._highlighted = nid
        self._flush_view()

    def _on_view_click(self, event) -> None:
        now = time.monotonic()
        prev = self._last_click_time
        self._last_click_time = now
        if now - prev > 0.4:
            return
        if not event:
            return
        wp = event.get("worldPosition") if isinstance(event, dict) else None
        if not wp or len(wp) < 3:
            return
        world_pt = np.asarray(wp[:3], dtype=float)
        name = self._resolve_probe_at_point(world_pt)
        if name is None:
            return
        state = getattr(self, "_readout_state", None)
        if state is None or state.probe == name:
            return
        with state:
            state.probe = name

    def _resolve_probe_at_point(
        self, world_pt: np.ndarray, *, threshold_mm: float = 0.5
    ) -> str | None:
        resolver = PoseResolver(
            scene=self.render_adapter.scene,
            plan=self.store.state,
            catalog=self.assets,
        )
        best_name: str | None = None
        best_dist = float("inf")
        for nid, node in self.render_adapter.scene.nodes.items():
            if not nid.startswith("probe:") or not node.enabled:
                continue
            geom = self.assets.get_geometry(node.asset_key)
            if not isinstance(geom, MeshTransformable):
                continue
            R, t = resolver.world_rt_for_node(node)
            local_pt = R.T @ (world_pt - t)
            # Some probe OBJs contain degenerate triangles (zero area);
            # trimesh's barycentric-coord computation divides by 0 on
            # those and emits a RuntimeWarning. Output is still correct.
            with np.errstate(invalid="ignore", divide="ignore"):
                _, dists, _ = trimesh.proximity.closest_point(
                    geom.raw, [local_pt]
                )
            d = float(dists[0])
            if d < best_dist:
                best_dist = d
                best_name = nid.removeprefix("probe:")
        if best_dist > threshold_mm:
            return None
        return best_name

    def _flush_view(self) -> None:
        """Push the current plotter state (camera, actors) to the
        browser. Camera-only updates (recenter / focus) need this
        because the renderer's auto-flush only fires on actor changes.

        ``plotter.render()`` triggers VTK's render pass; ``view_update``
        ships the resulting scene state over the trame WebSocket. We
        need both — render alone leaves the browser holding the prior
        snapshot, and view_update alone doesn't roll the camera matrix
        forward.
        """
        try:
            self.plotter.render()
        except Exception:
            pass
        update = getattr(self._ctrl, "view_update", None) if self._ctrl else None
        if callable(update):
            update()

    def recenter_view(self) -> None:
        """Set the camera to a brain-focused isometric pose.

        Camera sits in the **superior–left–anterior** octant relative
        to the brain's centroid, looking back at it; ``view_up`` is
        ``+z`` (Superior) so dragging up tilts toward Inferior. The
        camera distance is then refit to the brain mesh's bounding
        box so the brain fills the view rather than the whole scene
        bounding box (which includes the bulky implant + headframe
        stack and tends to shrink the brain).

        Bound to the ``c`` keyboard shortcut and the "Recenter" button
        in the Display tab. Also called at startup from
        :meth:`apply_default_view`.
        """
        brain_spec = self.assets.assets.get("brain")
        if brain_spec is None or brain_spec.mesh is None:
            self.plotter.reset_camera()
            return
        brain = brain_spec.mesh.raw
        centroid = np.asarray(brain.centroid, dtype=np.float64)
        # trimesh `bounds` is ((xmin, ymin, zmin), (xmax, ymax, zmax)) —
        # different from PyVista's flat (xmin, xmax, ymin, ymax, zmin,
        # zmax). Normalize to the flat form for ``reset_camera`` below.
        bounds_arr = np.asarray(brain.bounds, dtype=np.float64).reshape(2, 3)
        xmin, ymin, zmin = (float(v) for v in bounds_arr[0])
        xmax, ymax, zmax = (float(v) for v in bounds_arr[1])
        diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
        print(
            f"[recenter_view] brain centroid={centroid}, bounds="
            f"x=({xmin:.2f},{xmax:.2f}) y=({ymin:.2f},{ymax:.2f}) "
            f"z=({zmin:.2f},{zmax:.2f}) diag={diag:.2f}"
        )
        # LPS axes: +x = Left, +y = Posterior, +z = Superior.
        # Superior–left–anterior octant ⇒ offset has +x, -y, +z.
        # Equal magnitudes give an isometric perspective.
        offset = np.array([1.0, -1.0, 1.0]) * (diag / np.sqrt(3.0))
        self.plotter.camera.focal_point = tuple(centroid)
        self.plotter.camera.position = tuple(centroid + offset)
        # PyVista renames VTK's ``view_up`` to ``up`` and blocks the
        # original; setting ``view_up`` raises PyVistaAttributeError.
        self.plotter.camera.up = (0.0, 0.0, 1.0)
        # Re-fit camera distance to the brain bounds (not full scene).
        self.plotter.reset_camera(bounds=(xmin, xmax, ymin, ymax, zmin, zmax))
        cam = self.plotter.camera
        cam_pos = tuple(round(v, 2) for v in cam.position)
        print(
            f"[recenter_view] after reset: pos={cam_pos} "
            f"focal={tuple(round(v, 2) for v in cam.focal_point)} "
            f"up={tuple(round(v, 2) for v in cam.up)}"
        )
        self._flush_view()

    def focus_on_current_target(self) -> None:
        """Shift the camera's ``focal_point`` onto the currently selected
        probe's target.

        Only the orbit centre moves — ``position`` stays put — so the
        view direction tilts toward the target and subsequent left-drag
        orbits around it. If you want to also reposition the camera or
        refit zoom, press ``c`` (recenter on brain) afterwards or use
        VTK.js's ``r`` to reset.

        Looks up the target via ``plan_state.target_index`` for catalog
        keys (already in LPS) or converts ``target_point_RAS`` for
        inline targets. No-op silently if no probe is selected or the
        target can't be resolved.
        """
        state = self._readout_state
        probe_name = state.probe
        if not probe_name:
            return
        plan = self.store.state.probes.get(probe_name)
        if plan is None:
            return
        target_lps: np.ndarray | None = None
        if plan.target_key is not None:
            tlps = self.store.state.target_index.get(plan.target_key)
            if tlps is not None:
                arr = np.asarray(tlps, dtype=np.float64).reshape(-1, 3)
                target_lps = arr.mean(axis=0)
        if target_lps is None and plan.target_point_RAS is not None:
            ras = np.asarray(plan.target_point_RAS, dtype=np.float64).reshape(1, 3)
            target_lps = convert_coordinate_system(ras, "RAS", "LPS").reshape(3)
        if target_lps is None:
            return
        self.plotter.camera.focal_point = tuple(float(c) for c in target_lps)
        self._flush_view()

    # Default opacity overrides per scene-tag. Implant is mostly
    # transparent so probes-threading-through-holes are visible; other
    # fixtures (headframe, well, probe guard) sit at 40% transparency
    # so they're still readable as a frame of reference without
    # obscuring the implant + brain.
    _DEFAULT_OPACITY_BY_TAG: tuple[tuple[str, float, frozenset[str]], ...] = (
        ("implant", 0.2, frozenset()),
        ("fixture", 0.6, frozenset({"implant"})),
        ("headframe", 0.6, frozenset({"implant"})),
    )

    def apply_default_opacities(self) -> None:
        """Override per-node opacity for tagged fixtures so the implant
        is mostly transparent and other fixtures sit at 40% transparency.

        Reads tags from the scene graph and pokes the PyVistaBackend
        actor's `prop.opacity` directly. Called once at startup; the
        per-asset config opacity values are the baseline this overrides.
        """
        backend = self.render_adapter.backend
        if not hasattr(backend, "_actors"):
            return
        for node in self.render_adapter.scene.nodes.values():
            tags = node.tags
            for tag, opacity, excluded in self._DEFAULT_OPACITY_BY_TAG:
                if tag in tags and not (tags & excluded):
                    actor = backend._actors.get(node.key)
                    if actor is not None:
                        actor.prop.opacity = float(opacity)
                    break

    def apply_default_view(self) -> None:
        """Apply default camera (brain-focused iso, S-L-A octant) and
        opacity (implant 20%, fixtures 60%) on first paint. Called
        once from :func:`aind_low_point.app.build_trame_app` after the
        renderer's initial build."""
        self.recenter_view()
        self.apply_default_opacities()

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
        # Resolved angles. When the probe is calibrated, AP/ML come from
        # the calibration rotation (find_probe_angle), not from arc/ml_local
        # — match what ProbePose actually renders.
        cal = self.store.state.calibrations.get(state.probe)
        if plan.calibrated and cal is not None:
            from aind_mri_utils.reticle_calibrations import find_probe_angle

            ap_tilt, ml_tilt = find_probe_angle(cal.rotation)
            ap_tilt = float(ap_tilt)
            ml_tilt = float(ml_tilt)
        else:
            ap_tilt = (
                float(self.store.state.kinematics.arc_angles.get(plan.arc_id, 0.0))
                if plan.arc_id and plan.bind_ap_to_arc
                else float(plan.ap_local)
            )
            ml_tilt = float(plan.ml_local)
        n_shanks = max(1, len(self._shank_tips_local(f"probe:{plan.kind}")))
        with state:
            state.offset_r = float(r_mm)
            state.offset_a = float(a_mm)
            state.depth = float(plan.past_target_mm)
            state.ap_tilt = ap_tilt
            state.ml_tilt = ml_tilt
            state.spin = int(round(float(plan.spin)))
            if plan.arc_id:
                state.arc = plan.arc_id
            if plan.target_key:
                state.target = plan.target_key
            state.probe_kind = plan.kind
            state.probe_shank_options = list(range(1, n_shanks + 1))
            state.probe_position_bearing_shank = max(
                1, min(n_shanks, int(plan.position_bearing_shank))
            )
        self._refresh_readouts(state, state.probe)
