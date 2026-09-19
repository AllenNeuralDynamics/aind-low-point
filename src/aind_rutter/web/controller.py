"""Trame + PyVista controller for probe manipulation.

Holds the reactive state and turns a user intent into a planning command,
which it dispatches through ``PlanStore``. The page it builds, the numbers it
shows, the colours it paints and where it points the camera are each a mixin
beside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pyvista as pv
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from trame.app import get_server

from aind_rutter.collision.worker import CollisionHandler
from aind_rutter.domain.catalog import AssetCatalog
from aind_rutter.domain.commands import (
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
from aind_rutter.domain.plan import (
    locked_axes_for,
    probe_asset_key,
)
from aind_rutter.domain.pose import (
    ProbePose,
    resolved_angles,
)
from aind_rutter.render.adapter import RendererAdapter
from aind_rutter.render.overlays import OverlayResolver
from aind_rutter.session.store import PlanStore
from aind_rutter.web.camera import CameraMixin
from aind_rutter.web.ccf_regions import CCFOverlayManager
from aind_rutter.web.keybindings import (
    KB_ACTIONS,
    KB_SPEED_MULTIPLIER,
)
from aind_rutter.web.layout import LayoutMixin
from aind_rutter.web.materials import (
    VISIBILITY_GROUPS,
    MaterialsMixin,
    ccf_color_for_target,
)
from aind_rutter.web.readouts import ReadoutsMixin


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


@dataclass
class TrameController(LayoutMixin, ReadoutsMixin, MaterialsMixin, CameraMixin):
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
            # Which pose controls this probe cannot drive. The runtime owns the
            # rule; the sliders gray themselves from this list.
            state.probe_locked_axes = []
            state.probe_newscale_apply_x = 0.0
            state.probe_newscale_apply_y = 0.0
            state.probe_newscale_apply_z = 0.0
            state.probe_newscale_readout_x = "—"
            state.probe_newscale_readout_y = "—"
            state.probe_newscale_readout_z = "—"

        # Centering is built into the kinematic chain: any probe with
        # ``past_target_mm = 0`` and ``offsets_LP = (0, 0)`` already has
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
            color = ccf_color_for_target(self.assets, plan.target_key)
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
        # Preserve the stored arc/ap_local for un-toggling later.
        if "ap_tilt" in locked_axes_for(self.store.state, probe):
            return
        if plan.arc_id and plan.bind_ap_to_arc:
            self.store.dispatch(SetArcAngle(arc_id=plan.arc_id, ap_deg=ap))
        else:
            self.store.dispatch(SetProbeLocalAngles(name=probe, ap_local=ap))

    def _on_ml_change(self, state, probe: str, ml: float) -> None:
        plan = self.store.state.probes.get(probe)
        if not plan:
            return
        if "ml_tilt" in locked_axes_for(self.store.state, probe):
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

        from aind_rutter.build.calibration import newscale_to_lps
        from aind_rutter.domain.probe_kinds import recording_center_local_for_kind

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

    def _on_probe_kind_change(self, probe_name: str, new_kind: str) -> None:
        """Swap the probe's mesh by changing its ``kind``.

        Dispatching is all it takes to move the geometry: the render and
        collision adapters reconcile each probe node with its plan's kind and
        rebuild any handle they had built from the old mesh. What is left here
        is the part the adapters do not cover — the overlays and the colour,
        which depend on the new mesh's shank tips.
        """
        if probe_asset_key(new_kind) not in self.assets.assets:
            return  # unknown probe type — defensive
        plan = self.store.state.probes.get(probe_name)
        if plan is None or plan.kind == new_kind:
            return

        self.store.dispatch(SetProbeKind(name=probe_name, kind=new_kind))
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

        from aind_rutter.config import PlanningModel
        from aind_rutter.plan_io.replay import apply_plan_model_to_state

        if isinstance(content, (bytes, bytearray)):
            text = content.decode("utf-8")
        else:
            text = str(content)
        raw = yaml.safe_load(text)
        loaded = PlanningModel.model_validate(raw)
        apply_plan_model_to_state(loaded, self.store)

    def _load_probe_state(self, state) -> None:
        """Sync trame state from the currently selected probe."""
        if not state.probe:
            return
        plan = self.store.state.probes.get(state.probe)
        if not plan:
            return

        # The sliders are labelled R and A; the plan holds LPS.
        r_mm, a_mm = -plan.offsets_LP[0], -plan.offsets_LP[1]
        # The same resolution ProbePose renders from, clamped to the rig
        # limits, so the sliders cannot show a pose the probe is not in.
        ap_tilt, ml_tilt, spin = resolved_angles(state.probe, self.store.state)
        n_shanks = max(1, len(self._shank_tips_local(f"probe:{plan.kind}")))
        with state:
            state.offset_r = float(r_mm)
            state.offset_a = float(a_mm)
            state.depth = float(plan.past_target_mm)
            state.ap_tilt = ap_tilt
            state.ml_tilt = ml_tilt
            state.spin = int(round(spin))
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
