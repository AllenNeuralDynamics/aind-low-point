"""Trame + PyVista controller for probe manipulation.

Parallel implementation alongside the K3D / ipywidgets controller.
Calls store.dispatch() directly — PlanStore is the shared abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyvista as pv
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3

from aind_low_point.assets import AssetCatalog
from aind_low_point.collisions import CollisionHandler
from aind_low_point.commands import (
    AssignProbeArc,
    SetArcAngle,
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
        state.ap_tilt = 0.0
        state.ml_tilt = 0.0
        state.spin = 0

        self._load_probe_state(state)

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
            self._on_ap_change(state)

        @state.change("ml_tilt")
        def on_ml(**kwargs):
            self._on_ml_change(state)

        @state.change("spin")
        def on_spin(**kwargs):
            if not state.probe:
                return
            self.store.dispatch(
                SetProbeLocalAngles(
                    name=state.probe,
                    spin=float(state.spin),
                )
            )

        @state.change("arc")
        def on_arc_assign(**kwargs):
            self._on_arc_assign(state)

    def _on_ap_change(self, state) -> None:
        if not state.probe:
            return
        plan = self.store.state.probes.get(state.probe)
        if plan and plan.arc_id and plan.bind_ap_to_arc:
            self.store.dispatch(
                SetArcAngle(
                    arc_id=plan.arc_id,
                    ap_deg=float(state.ap_tilt),
                )
            )
        else:
            self.store.dispatch(
                SetProbeLocalAngles(
                    name=state.probe,
                    ap_local=float(state.ap_tilt),
                )
            )

    def _on_ml_change(self, state) -> None:
        if not state.probe:
            return
        plan = self.store.state.probes.get(state.probe)
        if not plan:
            return
        if plan.calibrated and state.probe in self.store.state.calibrations:
            state.ml_tilt = float(plan.ml_local)
            return
        new_ml = float(state.ml_tilt)
        if self.couple_ml:
            arc_id = plan.arc_id
            for name, other in self.store.state.probes.items():
                if other.calibrated and name in self.store.state.calibrations:
                    continue
                if other.arc_id == arc_id:
                    self.store.dispatch(SetProbeLocalAngles(name=name, ml_local=new_ml))
        else:
            self.store.dispatch(SetProbeLocalAngles(name=state.probe, ml_local=new_ml))

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
            vuetify3.VSlider(
                v_model=("offset_r", 0),
                min=-5,
                max=5,
                step=0.05,
                label="R (mm)",
                hide_details=True,
            )
            vuetify3.VSlider(
                v_model=("offset_a", 0),
                min=-5,
                max=5,
                step=0.05,
                label="A (mm)",
                hide_details=True,
            )
            vuetify3.VDivider(classes="my-2")
            vuetify3.VSlider(
                v_model=("ap_tilt", 0),
                min=-60,
                max=60,
                step=0.5,
                label="AP tilt (\u00b0)",
                hide_details=True,
            )
            vuetify3.VSlider(
                v_model=("ml_tilt", 0),
                min=-60,
                max=60,
                step=0.5,
                label="ML tilt (\u00b0)",
                hide_details=True,
            )
            vuetify3.VSlider(
                v_model=("spin", 0),
                min=-180,
                max=180,
                step=1,
                label="Spin (\u00b0)",
                hide_details=True,
            )
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
