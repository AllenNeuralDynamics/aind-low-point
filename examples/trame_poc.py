"""PyVista + trame POC for probe visualization.

Tests whether trame provides acceptable real-time interactivity
for mesh manipulation (vs Panel's VTK pane which was too slow).

Usage:
    Standalone: python examples/trame_poc.py
    Jupyter: server.start(exec_mode="task") after running this file
"""

import pyvista as pv
import vtk
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3

# --- Trame server ---
server = get_server()
state, ctrl = server.state, server.controller

# --- Create mesh and plotter ---
probe_shaft = pv.Cylinder(radius=0.1, height=5.0, center=(0, 0, 2.5))
probe_tip = pv.Cone(radius=0.15, height=0.5, center=(0, 0, -0.25), direction=(0, 0, -1))
probe_mesh = probe_shaft + probe_tip

pl = pv.Plotter()
actor = pl.add_mesh(probe_mesh, color="steelblue", opacity=0.9)
pl.add_axes()
pl.camera_position = "iso"

# --- Reactive state ---
state.x = 0.0
state.y = 0.0
state.z = 0.0
state.pitch = 0.0
state.yaw = 0.0


@state.change("x", "y", "z", "pitch", "yaw")
def on_pose_change(**kwargs):
    """Update actor transform when any slider changes."""
    transform = vtk.vtkTransform()
    transform.Translate(state.x, state.y, state.z)
    transform.RotateX(state.pitch)
    transform.RotateY(state.yaw)

    actor.SetUserTransform(transform)
    ctrl.view_update()


# --- Layout ---
with SinglePageLayout(server) as layout:
    layout.title.set_text("Probe POC - PyVista + Trame")

    with layout.content:
        with vuetify3.VContainer(fluid=True, classes="fill-height"):
            with vuetify3.VRow(classes="fill-height"):
                with vuetify3.VCol(cols=3):
                    vuetify3.VSlider(
                        v_model=("x", 0),
                        min=-10,
                        max=10,
                        step=0.1,
                        label="X (mm)",
                        hide_details=True,
                    )
                    vuetify3.VSlider(
                        v_model=("y", 0),
                        min=-10,
                        max=10,
                        step=0.1,
                        label="Y (mm)",
                        hide_details=True,
                    )
                    vuetify3.VSlider(
                        v_model=("z", 0),
                        min=-10,
                        max=10,
                        step=0.1,
                        label="Z (mm)",
                        hide_details=True,
                    )
                    vuetify3.VSlider(
                        v_model=("pitch", 0),
                        min=-60,
                        max=60,
                        step=1,
                        label="Pitch (deg)",
                        hide_details=True,
                    )
                    vuetify3.VSlider(
                        v_model=("yaw", 0),
                        min=-60,
                        max=60,
                        step=1,
                        label="Yaw (deg)",
                        hide_details=True,
                    )
                with vuetify3.VCol(cols=9, classes="fill-height"):
                    # mode="client" → vtk.js renders in browser (WebGL)
                    # Mesh data sent once; only transform deltas on updates
                    view = plotter_ui(pl, mode="client")
                    ctrl.view_update = view.update

if __name__ == "__main__":
    server.start()
