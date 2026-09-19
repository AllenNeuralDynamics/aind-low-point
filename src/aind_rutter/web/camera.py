"""Camera framing, click-to-pick and the probe highlight."""

from __future__ import annotations

import time

import numpy as np
import trimesh

from aind_rutter.domain.enums import Role
from aind_rutter.domain.pose import PoseResolver
from aind_rutter.domain.transforms import MeshTransformable


class CameraMixin:
    """See :class:`~aind_rutter.web.controller.TrameController`."""

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
            if not node.enabled:
                continue
            spec = self.assets.assets.get(node.asset_key)
            if spec is None or spec.role is not Role.PROBE:
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
                _, dists, _ = trimesh.proximity.closest_point(geom.raw, [local_pt])
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

        Frames the brain where the scene puts it, not where its file does: a
        brain whose node carries a ``transform`` is drawn tens of mm from its
        own mesh coordinates, and both the focal point and the bounds refit
        would miss it.
        """
        brain = self._get_brain_world_mesh()
        if brain is None:
            self.plotter.reset_camera()
            return
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
        keys or the plan's inline point, both already LPS, for
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
        if target_lps is None and plan.target_point_LPS is not None:
            target_lps = np.asarray(plan.target_point_LPS, dtype=np.float64)
        if target_lps is None:
            return
        self.plotter.camera.focal_point = tuple(float(c) for c in target_lps)
        self._flush_view()

    def apply_default_view(self) -> None:
        """Apply default camera (brain-focused iso, S-L-A octant) and
        opacity (implant 20%, fixtures 60%) on first paint. Called
        once from :func:`aind_rutter.web.app.build_trame_app` after the
        renderer's initial build."""
        self.recenter_view()
        self.apply_default_opacities()
