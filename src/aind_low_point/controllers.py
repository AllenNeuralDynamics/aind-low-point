"""The controller that manipulates the plan"""

from __future__ import annotations

from dataclasses import dataclass, field

import ipywidgets as widgets
import k3d
import numpy as np
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from ipyevents import Event
from IPython.display import display

from aind_low_point.assets import (
    AssetCatalog,
)
from aind_low_point.collisions import CollisionHandler
from aind_low_point.rendering import OverlayResolver, RendererAdapter
from aind_low_point.state_change import PlanStore


@dataclass
class ProbeWidgetController:
    data: AppData
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

    # Position (RAS)
    pos_ap_ras: widgets.FloatSlider = field(init=False)
    pos_ml_ras: widgets.FloatSlider = field(init=False)
    pos_dv_ras: widgets.FloatSlider = field(init=False)

    # Rotations (about tip)
    ap_tilt_deg: widgets.FloatSlider = field(
        init=False
    )  # AP coupled by arc unless calibrated
    ml_tilt_deg: widgets.FloatSlider = field(
        init=False
    )  # per-probe (or optionally coupled)
    spin_deg: widgets.IntSlider = field(init=False)  # per-probe

    arc_assign_dd: widgets.Dropdown = field(init=False)
    # Target snap (optional convenience)
    target_dd: widgets.Dropdown = field(init=False)
    goto_btn: widgets.Button = field(init=False)

    status_out: widgets.Output = field(init=False)

    # Keyboard helpers (optional)
    kb_panel: widgets.HTML = field(init=False)
    kb_event: Event | None = field(init=False, default=None)
    step_pos_mm: float = 0.05
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
            "<b>Keyboard</b> (click panel): Move <code>W/S</code>=AP ±, <code>A/D</code>=ML ∓, <code>R/F</code>=DV ±; "
            "Tilt <code>I/K</code>=AP ±, <code>J/L</code>=ML ±; Spin <code>U/O</code> ±. "
            "Shift×10, Ctrl×0.2."
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
                widgets.HTML("<b>Position (RAS)</b>"),
                widgets.HBox([self.pos_ap_ras, self.pos_ml_ras, self.pos_dv_ras]),
                widgets.HTML("<b>Orientation about tip (°)</b>"),
                widgets.HBox([self.ap_tilt_deg, self.ml_tilt_deg, self.spin_deg]),
                self.status_out,
            ]
        )
        self.view = widgets.HBox([self.controls, self.plot])

    # ---------- helpers ----------
    def _current_probe(self):
        pname = self.probe_dd.value
        return pname, self.store.state.probes[pname] if pname else (None, None)

    def _probe_arc_id(self, probe_name: str) -> str:
        return self.data.plan.probe_info[probe_name].arc

    def _arc_angle(self, arc_id: str) -> float:
        return float(self.data.plan.arcs[arc_id])

    def _target_names(self) -> list[str]:
        return sorted(self.assets.targets.keys())

    def _target_point_LPS(self, key: str) -> np.ndarray:
        pts = self.assets.targets[key].raw  # (N,3)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            raise ValueError(f"targets['{key}'] must be (N,3)")
        return np.asarray(pts.mean(axis=0), dtype=float)

    # ---------- domain pushes ----------
    def _dispatch_tip_only(self, probe_name: str, tip_lps: np.ndarray):
        probe = self.store.state.probes[probe_name]
        self.store.dispatch(
            SetProbePlanPose(
                name=probe_name,
                ap=probe.pose.ap,  # unchanged
                ml=probe.pose.ml,  # unchanged
                spin=probe.pose.spin,  # unchanged
                tip=tip_lps,
            )
        )

    def _dispatch_pose(
        self,
        probe_name: str,
        ap: float,
        ml: float,
        spin: float,
        tip_lps: np.ndarray | None = None,
    ):
        probe = self.store.state.probes[probe_name]
        self.store.dispatch(
            SetProbePlanPose(
                name=probe_name,
                ap=ap,
                ml=ml,
                spin=spin,
                tip=probe.pose.tip if tip_lps is None else tip_lps,
            )
        )

    # ---------- apply from UI (live) ----------
    def _apply_xyz_live(self):
        pname, probe = self._current_probe()
        if not pname:
            return
        tip_ras = np.array(
            [self.pos_ml_ras.value, self.pos_ap_ras.value, self.pos_dv_ras.value],
            dtype=float,
        )
        tip_lps = convert_coordinate_system(tip_ras, "RAS", "LPS")  # convert RAS to LPS
        self._dispatch_tip_only(pname, tip_lps)

    def _apply_spin_live(self):
        pname, probe = self._current_probe()
        if not pname:
            return
        # spin is always allowed, even if calibrated
        self._dispatch_pose(
            pname, ap=probe.pose.ap, ml=probe.pose.ml, spin=float(self.spin_deg.value)
        )

    def _apply_ml_live(self):
        pname, probe = self._current_probe()
        if not pname:
            return
        if getattr(probe, "calibrated", False):
            # locked
            self.ml_tilt_deg.value = float(probe.pose.ml)
            return
        new_ml = float(self.ml_tilt_deg.value)
        if self.couple_ml:
            # propagate to all non-calibrated probes sharing this arc
            arc_id = self._probe_arc_id(pname)
            for other_name, other in self.store.state.probes.items():
                if getattr(other, "calibrated", False):
                    continue
                if self._probe_arc_id(other_name) == arc_id:
                    self._dispatch_pose(
                        other_name, ap=other.pose.ap, ml=new_ml, spin=other.pose.spin
                    )
        else:
            self._dispatch_pose(
                pname, ap=probe.pose.ap, ml=new_ml, spin=probe.pose.spin
            )

    def _apply_ap_via_arc_live(self):
        """AP tilt is coupled by arc: edit arc angle and propagate to all non-calibrated probes on that arc."""
        pname, probe = self._current_probe()
        if not pname:
            return
        arc_id = self._probe_arc_id(pname)
        self.store.dispatch(
            SetArcAngle(arc_id=arc_id, angle_deg=float(self.ap_tilt_deg.value))
        )

    # ---------- load UI from domain ----------
    def _load_probe_into_widgets(self, probe_name: str):
        p = self.store.state.probes[probe_name]
        arc_id = self._probe_arc_id(probe_name)

        # keep options in sync (in case arcs were added/removed elsewhere)
        self.arc_assign_dd.options = sorted(self.data.plan.arcs.keys())
        self.arc_assign_dd.value = arc_id  # select current arc

        self.arc_label.value = f"<b>Arc:</b> {arc_id} &nbsp; (<i>AP coupled</i>)"

        # XYZ
        tip_ras = convert_coordinate_system(p.pose.tip, "RAS", "LPS")
        self.pos_ml_ras.value = float(tip_ras[0])
        self.pos_ap_ras.value = float(tip_ras[1])
        self.pos_dv_ras.value = float(tip_ras[2])

        # AP from arc; ML, Spin from probe
        self.ap_tilt_deg.value = self._arc_angle(arc_id)
        self.ml_tilt_deg.value = float(p.pose.ml)
        self.spin_deg.value = int(round(float(p.pose.spin)))

        # lock AP/ML when calibrated
        is_cal = getattr(p, "calibrated", False)
        self.ap_tilt_deg.disabled = (
            is_cal  # AP control will still show arc value but be disabled if calibrated
        )
        self.ml_tilt_deg.disabled = is_cal

        with self.status_out:
            self.status_out.clear_output(wait=True)
            print(
                f"[UI] Loaded '{probe_name}'  (arc={arc_id}, {'calibrated' if is_cal else 'free'})"
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
            options=sorted(self.data.plan.arcs.keys()),
            description="Arc:",
            layout={"width": "140px"},
        )

        # Position sliders (RAS)
        self.pos_ap_ras = widgets.FloatSlider(
            value=0.0,
            min=-10,
            max=10,
            step=0.05,
            description="AP (mm)",
            continuous_update=False,
            layout={"width": "220px"},
        )
        self.pos_ml_ras = widgets.FloatSlider(
            value=0.0,
            min=-10,
            max=10,
            step=0.05,
            description="ML (mm)",
            continuous_update=False,
            layout={"width": "220px"},
        )
        self.pos_dv_ras = widgets.FloatSlider(
            value=0.0,
            min=-10,
            max=10,
            step=0.05,
            description="DV (mm)",
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
        self.goto_btn = widgets.Button(description="Go to target", button_style="info")

        self.status_out = widgets.Output(
            layout={
                "border": "1px solid lightgray",
                "max_height": "120px",
                "overflow": "auto",
            }
        )

        # Keyboard panel
        self.kb_panel = widgets.HTML(
            value="<div style='border:1px dashed #999;padding:6px;border-radius:6px;background:#fafafa;'>"
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
        # Live position
        for w in (self.pos_ap_ras, self.pos_ml_ras, self.pos_dv_ras):
            w.observe(
                lambda ch: self._apply_xyz_live() if ch["name"] == "value" else None,
                names="value",
            )

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
            lambda ch: self._load_probe_into_widgets(ch["new"])
            if ch["name"] == "value"
            else None,
            names="value",
        )

        # Go to target (XYZ only)
        def _goto(_):
            if not self.target_dd.value:
                return
            tip_lps = self._target_point_LPS(self.target_dd.value)
            tip_ras = convert_coordinate_system(
                tip_lps, "LPS", "RAS"
            )  # convert LPS to RAS
            self.pos_ml_ras.value = float(tip_ras[0])
            self.pos_ap_ras.value = float(tip_ras[1])
            self.pos_dv_ras.value = float(tip_ras[2])
            self._apply_xyz_live()

        self.goto_btn.on_click(_goto)

        def _on_arc_assign(change):
            if change["name"] != "value" or change["new"] is None:
                return
            pname, probe = self._current_probe()
            if not pname:
                return
            new_arc_id = change["new"]
            old_arc_id = self._probe_arc_id(pname)
            if new_arc_id == old_arc_id:
                return

            # If calibrated, we still change the assignment but don't move AP/ML
            is_cal = getattr(probe, "calibrated", False)

            # Snap the new arc to the probe's current AP for continuity, then (optionally)
            # propagate to this probe (no propagation if calibrated).
            self.store.dispatch(
                AssignProbeArc(
                    probe_name=pname,
                    new_arc_id=new_arc_id,
                    snap_arc_to_current_ap=True,
                    propagate=not is_cal,
                )
            )

            # Refresh UI labels/sliders to reflect the new arc
            self.arc_label.value = (
                f"<b>Arc:</b> {new_arc_id} &nbsp; (<i>AP coupled</i>)"
            )
            # AP slider reflects arc angle (it may be disabled if calibrated)
            self.ap_tilt_deg.value = self._arc_angle(new_arc_id)

        self.arc_assign_dd.observe(_on_arc_assign, names="value")

        # Keyboard (optional)
        if self.kb_event is not None:

            def on_key(event):
                key = (event or {}).get("key", "")
                shift = bool((event or {}).get("shiftKey", False))
                ctrl = bool((event or {}).get("ctrlKey", False))
                mul = 10.0 if shift else 0.2 if ctrl else 1.0
                dpos = self.step_pos_mm * mul
                dtilt = self.step_tilt_deg * mul
                dspin = self.step_spin_deg * mul
                handled = True

                # XYZ (RAS)
                if key in ("w", "W", "ArrowUp"):
                    self.pos_ap_ras.value += dpos
                elif key in ("s", "S", "ArrowDown"):
                    self.pos_ap_ras.value -= dpos
                elif key in ("a", "A", "ArrowLeft"):
                    self.pos_ml_ras.value -= dpos
                elif key in ("d", "D", "ArrowRight"):
                    self.pos_ml_ras.value += dpos
                elif key in ("r", "R"):
                    self.pos_dv_ras.value += dpos
                elif key in ("f", "F"):
                    self.pos_dv_ras.value -= dpos

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

            self.kb_event.on_dom_event(on_key)

    def _populate_initial(self):
        # Select first probe and load its state
        if self.probe_dd.options:
            self.probe_dd.value = list(self.probe_dd.options)[0]
            self._load_probe_into_widgets(self.probe_dd.value)

    # ---------- public ----------
    def display(self):
        display(self.view)
