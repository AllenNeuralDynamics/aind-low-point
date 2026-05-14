"""The controller that manipulates the plan"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import ipywidgets as widgets
import k3d
from ipyevents import Event
from IPython.display import display

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
class ProbeWidgetController:
    store: PlanStore
    assets: AssetCatalog
    plot: k3d.Plot
    render_adapter: RendererAdapter
    collision_handler: CollisionHandler
    overlays_resolver: OverlayResolver

    # Coupling behavior (AP always coupled via arc; ML can be optionally coupled)
    couple_ml: bool = False

    # UI containers
    controls: widgets.VBox = field(init=False)
    view: widgets.HBox = field(init=False)

    # Core widgets
    probe_dd: widgets.Dropdown = field(init=False)
    arc_label: widgets.HTML = field(init=False)

    # Offsets (R/A in RAS coords)
    offset_r_mm: widgets.FloatSlider = field(init=False)
    offset_a_mm: widgets.FloatSlider = field(init=False)

    # Rotations (about tip)
    ap_tilt_deg: widgets.FloatSlider = field(init=False)
    ml_tilt_deg: widgets.FloatSlider = field(init=False)
    spin_deg: widgets.IntSlider = field(init=False)

    arc_assign_dd: widgets.Dropdown = field(init=False)
    # Target snap (optional convenience)
    target_dd: widgets.Dropdown = field(init=False)
    goto_btn: widgets.Button = field(init=False)

    status_out: widgets.Output = field(init=False)

    # Keyboard helpers (optional)
    kb_panel: widgets.HTML = field(init=False)
    kb_event: Event | None = field(init=False, default=None)
    step_offset_mm: float = 0.05
    step_tilt_deg: float = 0.5
    step_spin_deg: float = 1.0

    def __post_init__(self):
        # enable collision overlays in renderer
        self.render_adapter.overlays = self.overlays_resolver

        self._build_widgets()
        self._wire_events()
        self._populate_initial()

        kb_help = widgets.HTML(
            "<div style='font-size:12px;line-height:1.3'>"
            "<b>Keyboard</b> (click panel): "
            "Offset <code>W/S</code>=A, <code>A/D</code>=R; "
            "Tilt <code>I/K</code>=AP, <code>J/L</code>=ML; "
            "Spin <code>U/O</code>. Shift×10, Ctrl×0.2."
            "</div>"
        )
        self.controls = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.probe_dd,
                        self.arc_label,
                        self.arc_assign_dd,
                        self.target_dd,
                        self.goto_btn,
                    ]
                ),
                kb_help,
                self.kb_panel,
                widgets.HTML("<b>Offsets (mm)</b>"),
                widgets.HBox([self.offset_r_mm, self.offset_a_mm]),
                widgets.HTML("<b>Orientation about tip (°)</b>"),
                widgets.HBox([self.ap_tilt_deg, self.ml_tilt_deg, self.spin_deg]),
                self.status_out,
            ]
        )
        self.view = widgets.HBox([self.controls, self.plot])

    # ---------- helpers ----------
    def _current_probe(self) -> tuple[str | None, Any]:
        pname = self.probe_dd.value
        if not pname:
            return None, None
        return pname, self.store.state.probes.get(pname)

    def _probe_arc_id(self, probe_name: str) -> str | None:
        plan = self.store.state.probes.get(probe_name)
        return plan.arc_id if plan else None

    def _arc_angle(self, arc_id: str) -> float:
        return float(self.store.state.kinematics.arc_angles.get(arc_id, 0.0))

    def _arc_ids(self) -> list[str]:
        return sorted(self.store.state.kinematics.arc_angles.keys())

    def _target_names(self) -> list[str]:
        return sorted(self.assets.targets.keys())

    # ---------- domain pushes ----------
    def _apply_offsets_live(self):
        pname, plan = self._current_probe()
        if not pname:
            return
        self.store.dispatch(
            SetProbeOffsetsRA(
                name=pname,
                R_mm=float(self.offset_r_mm.value),
                A_mm=float(self.offset_a_mm.value),
            )
        )

    def _apply_spin_live(self):
        pname, plan = self._current_probe()
        if not pname:
            return
        self.store.dispatch(
            SetProbeLocalAngles(name=pname, spin=float(self.spin_deg.value))
        )

    def _apply_ml_live(self):
        pname, plan = self._current_probe()
        if not pname or not plan:
            return
        if plan.calibrated and pname in self.store.state.calibrations:
            # locked - revert slider to current value
            self.ml_tilt_deg.value = float(plan.ml_local)
            return
        new_ml = float(self.ml_tilt_deg.value)
        if self.couple_ml:
            # propagate to all non-calibrated probes sharing this arc
            arc_id = self._probe_arc_id(pname)
            for other_name, other in self.store.state.probes.items():
                if other.calibrated and other_name in self.store.state.calibrations:
                    continue
                if other.arc_id == arc_id:
                    self.store.dispatch(
                        SetProbeLocalAngles(name=other_name, ml_local=new_ml)
                    )
        else:
            self.store.dispatch(SetProbeLocalAngles(name=pname, ml_local=new_ml))

    def _apply_ap_via_arc_live(self):
        """AP tilt is coupled by arc."""
        pname, plan = self._current_probe()
        if not pname or not plan:
            return
        arc_id = self._probe_arc_id(pname)
        if arc_id:
            self.store.dispatch(
                SetArcAngle(arc_id=arc_id, ap_deg=float(self.ap_tilt_deg.value))
            )

    # ---------- load UI from domain ----------
    def _load_probe_into_widgets(self, probe_name: str):
        plan = self.store.state.probes.get(probe_name)
        if not plan:
            return

        arc_id = plan.arc_id

        # keep arc dropdown options in sync
        self.arc_assign_dd.options = self._arc_ids()
        if arc_id:
            self.arc_assign_dd.value = arc_id

        arc_label = arc_id or "None"
        self.arc_label.value = f"<b>Arc:</b> {arc_label}"
        if arc_id and plan.bind_ap_to_arc:
            self.arc_label.value += " &nbsp; (<i>AP coupled</i>)"

        # Offsets
        r_mm, a_mm = plan.offsets_RA
        self.offset_r_mm.value = float(r_mm)
        self.offset_a_mm.value = float(a_mm)

        # Angles: AP from arc if bound, else local; ML and spin from local
        if arc_id and plan.bind_ap_to_arc:
            self.ap_tilt_deg.value = self._arc_angle(arc_id)
        else:
            self.ap_tilt_deg.value = float(plan.ap_local)
        self.ml_tilt_deg.value = float(plan.ml_local)
        self.spin_deg.value = int(round(float(plan.spin)))

        # lock AP/ML when calibrated
        is_cal = plan.calibrated and probe_name in self.store.state.calibrations
        self.ap_tilt_deg.disabled = is_cal
        self.ml_tilt_deg.disabled = is_cal

        with self.status_out:
            self.status_out.clear_output(wait=True)
            print(
                f"[UI] Loaded '{probe_name}'  "
                f"(arc={arc_id}, {'calibrated' if is_cal else 'free'})"
            )

    # ---------- UI build & events ----------
    def _build_widgets(self):
        self.probe_dd = widgets.Dropdown(
            options=sorted(self.store.state.probes.keys()),
            description="Probe:",
            layout={"width": "220px"},
        )
        self.arc_label = widgets.HTML(layout={"width": "180px"})

        self.arc_assign_dd = widgets.Dropdown(
            options=self._arc_ids(),
            description="Arc:",
            layout={"width": "140px"},
        )

        # Offset sliders (R/A)
        self.offset_r_mm = widgets.FloatSlider(
            value=0.0,
            min=-5,
            max=5,
            step=0.05,
            description="R (mm)",
            continuous_update=False,
            layout={"width": "220px"},
        )
        self.offset_a_mm = widgets.FloatSlider(
            value=0.0,
            min=-5,
            max=5,
            step=0.05,
            description="A (mm)",
            continuous_update=False,
            layout={"width": "220px"},
        )

        # Orientation about tip
        self.ap_tilt_deg = widgets.FloatSlider(
            value=0.0,
            min=-60,
            max=60,
            step=0.5,
            description="AP tilt (°)",
            continuous_update=False,
            layout={"width": "220px"},
        )
        self.ml_tilt_deg = widgets.FloatSlider(
            value=0.0,
            min=-60,
            max=60,
            step=0.5,
            description="ML tilt (°)",
            continuous_update=False,
            layout={"width": "220px"},
        )
        self.spin_deg = widgets.IntSlider(
            value=0,
            min=-180,
            max=180,
            step=1,
            description="Spin (°)",
            continuous_update=False,
            layout={"width": "220px"},
        )

        # Targets
        self.target_dd = widgets.Dropdown(
            options=self._target_names(),
            description="Target:",
            layout={"width": "220px"},
        )
        self.goto_btn = widgets.Button(description="Set target", button_style="info")

        self.status_out = widgets.Output(
            layout={
                "border": "1px solid lightgray",
                "max_height": "120px",
                "overflow": "auto",
            }
        )

        # Keyboard panel
        self.kb_panel = widgets.HTML(
            value="<div style='border:1px dashed #999;padding:6px;"
            "border-radius:6px;background:#fafafa;'>"
            "<b>Click here</b> to enable keyboard control</div>",
            layout=widgets.Layout(width="320px"),
        )
        if Event is not None:
            self.kb_event = Event(
                source=self.kb_panel,
                watched_events=["keydown"],
                prevent_default_action=True,
            )
        else:
            self.kb_event = None

    def _wire_events(self):
        # Live offsets
        def _on_offset(ch):
            if ch["name"] == "value":
                self._apply_offsets_live()

        for w in (self.offset_r_mm, self.offset_a_mm):
            w.observe(_on_offset, names="value")

        # Live rotations
        self.spin_deg.observe(
            lambda ch: self._apply_spin_live() if ch["name"] == "value" else None,
            names="value",
        )
        self.ml_tilt_deg.observe(
            lambda ch: self._apply_ml_live() if ch["name"] == "value" else None,
            names="value",
        )
        self.ap_tilt_deg.observe(
            lambda ch: self._apply_ap_via_arc_live() if ch["name"] == "value" else None,
            names="value",
        )

        # Probe switch
        self.probe_dd.observe(
            lambda ch: (
                self._load_probe_into_widgets(ch["new"])
                if ch["name"] == "value"
                else None
            ),
            names="value",
        )

        # Set target
        def _set_target(_):
            pname, _ = self._current_probe()
            if not pname or not self.target_dd.value:
                return
            self.store.dispatch(
                SetProbeTarget(name=pname, target_key=self.target_dd.value)
            )
            with self.status_out:
                print(f"[UI] Set target for '{pname}' to '{self.target_dd.value}'")

        self.goto_btn.on_click(_set_target)

        def _on_arc_assign(change):
            if change["name"] != "value" or change["new"] is None:
                return
            pname, plan = self._current_probe()
            if not pname or not plan:
                return
            new_arc_id = change["new"]
            old_arc_id = plan.arc_id
            if new_arc_id == old_arc_id:
                return

            self.store.dispatch(
                AssignProbeArc(name=pname, arc_id=new_arc_id, bind_ap_to_arc=True)
            )

            # Refresh UI labels/sliders to reflect the new arc
            self.arc_label.value = (
                f"<b>Arc:</b> {new_arc_id} &nbsp; (<i>AP coupled</i>)"
            )
            self.ap_tilt_deg.value = self._arc_angle(new_arc_id)

        self.arc_assign_dd.observe(_on_arc_assign, names="value")

        # Keyboard (optional)
        if self.kb_event is not None:
            self.kb_event.on_dom_event(self._on_keyboard_event)

    def _on_keyboard_event(self, event):
        """Handle keyboard shortcuts for probe manipulation."""
        key = (event or {}).get("key", "")
        shift = bool((event or {}).get("shiftKey", False))
        ctrl = bool((event or {}).get("ctrlKey", False))
        mul = 10.0 if shift else 0.2 if ctrl else 1.0
        doff = self.step_offset_mm * mul
        dtilt = self.step_tilt_deg * mul
        dspin = self.step_spin_deg * mul
        handled = True

        # Offsets (R/A)
        if key in ("w", "W", "ArrowUp"):
            self.offset_a_mm.value += doff
        elif key in ("s", "S", "ArrowDown"):
            self.offset_a_mm.value -= doff
        elif key in ("a", "A", "ArrowLeft"):
            self.offset_r_mm.value -= doff
        elif key in ("d", "D", "ArrowRight"):
            self.offset_r_mm.value += doff

        # Rotations (about tip)
        elif key in ("i", "I"):
            self.ap_tilt_deg.value += dtilt
        elif key in ("k", "K"):
            self.ap_tilt_deg.value -= dtilt
        elif key in ("j", "J"):
            self.ml_tilt_deg.value -= dtilt
        elif key in ("l", "L"):
            self.ml_tilt_deg.value += dtilt
        elif key in ("u", "U"):
            self.spin_deg.value -= dspin
        elif key in ("o", "O"):
            self.spin_deg.value += dspin
        else:
            handled = False

        if handled:
            with self.status_out:
                print(f"[KB] {key} (x{mul:g})")

    def _populate_initial(self):
        # Select first probe and load its state
        if self.probe_dd.options:
            self.probe_dd.value = list(self.probe_dd.options)[0]
            self._load_probe_into_widgets(self.probe_dd.value)

    # ---------- public ----------
    def display(self):
        display(self.view)
