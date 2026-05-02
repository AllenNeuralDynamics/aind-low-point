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

from aind_low_point.assets import AssetCatalog
from aind_low_point.ccf_overlay import CCFOverlayManager
from aind_low_point.collisions import CollisionHandler
from aind_low_point.commands import (
    AssignProbeArc,
    SetArcAngle,
    SetProbePastTarget,
    SetProbeLocalAngles,
    SetProbeOffsetsRA,
    SetProbeTarget,
)
from aind_low_point.rendering import OverlayResolver, RendererAdapter
from aind_low_point.state_change import PlanStore


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

        state.probes = probe_names
        state.probe = probe_names[0] if probe_names else None
        state.arcs = arc_ids
        state.arc = arc_ids[0] if arc_ids else None
        state.targets = target_names
        state.target = target_names[0] if target_names else None

        state.offset_r = 0.0
        state.offset_a = 0.0
        state.depth = 0.0
        state.ap_tilt = 0.0
        state.ml_tilt = 0.0
        state.spin = 0

        self._load_probe_state(state)

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
            if self.on_save is not None:
                vuetify3.VBtn(
                    "Save YAML",
                    color="success",
                    click=self.on_save,
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
        state.offset_r = float(r_mm)
        state.offset_a = float(a_mm)
        state.depth = float(plan.past_target_mm)

        if plan.arc_id and plan.bind_ap_to_arc:
            state.ap_tilt = float(
                self.store.state.kinematics.arc_angles.get(plan.arc_id, 0.0)
            )
        else:
            state.ap_tilt = float(plan.ap_local)

        state.ml_tilt = float(plan.ml_local)
        state.spin = int(round(float(plan.spin)))

        if plan.arc_id:
            state.arc = plan.arc_id
        if plan.target_key:
            state.target = plan.target_key
