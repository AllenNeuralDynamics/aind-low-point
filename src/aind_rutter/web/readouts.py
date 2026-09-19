"""Derived numbers the planner shows: tip, depth, over-insertion,
collisions, kinematic status and the NewScale readout.

Each returns formatted text or a status and writes nothing; the caller
puts the result into the trame state.
"""

from __future__ import annotations

import numpy as np
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from aind_mri_utils.arc_angles import arc_angles_to_affine

from aind_rutter.build.queries import brain_world_mesh
from aind_rutter.domain.plan import (
    kinematic_violations,
    locked_axes_for,
    probe_asset_key,
)
from aind_rutter.domain.pose import (
    ProbePose,
    detect_shank_tips_local,
    named_shank_tip_world,
)
from aind_rutter.plan_io.rig_export import depth_along_probe_axis


class ReadoutsMixin:
    """See :class:`~aind_rutter.web.controller.TrameController`."""

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
        named_world_lps = named_shank_tip_world(
            pose.tip,
            R,
            self._shank_tips_local(probe_asset_key(plan.kind)),
            plan.position_bearing_shank,
        )
        tip_ras = convert_coordinate_system(named_world_lps, "LPS", "RAS")
        tip_str = f"{tip_ras[0]:+.2f}, {tip_ras[1]:+.2f}, {tip_ras[2]:+.2f} mm"

        depth_str = "—"
        overins_str = "—"
        brain_mesh = self._get_brain_world_mesh()
        if brain_mesh is not None:
            probe_axis = R @ np.array([0.0, 0.0, 1.0])
            # Depth is the distance from the named shank's tip down to
            # the nearest brain-surface intersection along the shaft.
            depth = depth_along_probe_axis(named_world_lps, probe_axis, brain_mesh)
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
        self._brain_world_mesh = brain_world_mesh(
            self.assets, self.render_adapter.scene
        )
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
            state.probe_locked_axes = sorted(
                locked_axes_for(self.store.state, probe_name)
            )
            state.probe_newscale_readout_x = nx
            state.probe_newscale_readout_y = ny
            state.probe_newscale_readout_z = nz

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
            from aind_rutter.build.calibration import lps_to_newscale
            from aind_rutter.domain.pose import ProbePose

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
