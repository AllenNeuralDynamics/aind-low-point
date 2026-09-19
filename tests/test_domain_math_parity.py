"""One definition each for the quantities several layers compute.

The survey found the probe pivot derived in four places and the per-probe
variable count declared in five. A second copy is not a style problem: the app
draws one pivot and the optimizer optimizes another, and nothing says so.
"""

from __future__ import annotations

import numpy as np
import pytest

from aind_rutter.domain.probe_kinds import RECORDING_GEOMETRY, pivot_from_shank_tips

TIPS = np.array([[0.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.5, 0.0, 0.0]])


def test_the_per_probe_variable_count_has_one_definition() -> None:
    """`restore` declared a bare 6 of its own; a layout change would have
    silently disagreed with the objective it indexes into."""
    from aind_rutter.optimization.objectives import clearance_metrics, variables
    from aind_rutter.optimization.objectives.phase1 import PHASE1_PER_PROBE_VARS
    from aind_rutter.optimization.pipeline import restore

    counts = {
        variables.PPV,
        clearance_metrics.PPV,
        restore.PPV,
        PHASE1_PER_PROBE_VARS,
    }
    assert counts == {PHASE1_PER_PROBE_VARS}


def test_the_pivot_helper_takes_a_centre_rather_than_looking_one_up() -> None:
    """It used to consult the built-in table when the centre was omitted, so a
    caller that passed only a kind got a different answer for an unregistered
    probe than one that resolved the geometry itself."""
    import inspect

    params = inspect.signature(pivot_from_shank_tips).parameters
    assert list(params) == ["shank_tips_local", "center_mm"]
    assert params["center_mm"].default is inspect.Parameter.empty


@pytest.mark.parametrize("kind", sorted(RECORDING_GEOMETRY))
def test_every_site_derives_the_same_pivot(kind: str) -> None:
    """The four call sites — runtime build, the visibility atlas, and both
    probe-static builders — now compute this one expression."""
    center = RECORDING_GEOMETRY[kind].active_center_mm
    expected = np.array([TIPS[:, 0].mean(), TIPS[:, 1].mean(), center])
    np.testing.assert_allclose(pivot_from_shank_tips(TIPS, center), expected)


def test_the_atlas_and_the_statics_agree() -> None:
    """Both take the centre from the probe's resolved recording geometry."""
    from types import SimpleNamespace

    from aind_rutter.optimization.enumeration.visibility_atlas import (
        _probe_centroid_local,
    )

    geom = RECORDING_GEOMETRY["2.1"]
    probe = SimpleNamespace(kind="2.1", shank_tips_local=TIPS, recording=geom)
    np.testing.assert_allclose(
        _probe_centroid_local(probe),
        pivot_from_shank_tips(TIPS, geom.active_center_mm),
    )


def test_a_configured_pivot_still_wins_everywhere() -> None:
    """`AssetSpec.pivot_LPS` overrides the derivation; that is what D11 fixed."""
    from aind_rutter.optimization.geometry.holes import Hole, HoleSection
    from aind_rutter.optimization.geometry.probes import ProbeStaticInfo
    from aind_rutter.optimization.objectives.probe_static import _build_probe_static

    configured = np.array([0.25, -0.5, 2.0])
    probe = ProbeStaticInfo(
        name="p",
        target_LPS=np.zeros(3),
        kind="2.1",
        shank_tips_local=TIPS,
        pivot_local=configured,
    )
    hole = Hole(
        id=1,
        axis=np.array([0.0, 0.0, 1.0]),
        ref_point=np.zeros(3),
        sections=[
            HoleSection(
                axis=np.array([0.0, 0.0, 1.0]),
                center=np.zeros(3),
                a=0.5,
                b=0.5,
                theta=0.0,
            )
        ],
    )
    from types import SimpleNamespace

    (static,) = _build_probe_static(
        [probe],
        [hole],
        SimpleNamespace(probe_to_hole={"p": 1}),
        SimpleNamespace(probe_to_arc_idx={"p": 0}, arc_centroids_deg=(0.0,)),
    )
    np.testing.assert_allclose(static.pivot_local, configured)


def test_the_reduced_stride_has_one_definition() -> None:
    """It was a bare `3` at 21 sites across five modules.

    A literal is not wrong until the layout changes, at which point a site that
    was missed keeps indexing the old shape and silently reads a neighbouring
    variable rather than failing.
    """
    from aind_rutter.optimization.objectives import layout

    assert layout.REDUCED_PER_PROBE_VARS == 3
    assert layout.PHASE1_PER_PROBE_VARS == 6


def test_the_two_layouts_share_their_first_three_slots() -> None:
    """Which is what lets a reduced solution seed a full one."""
    from aind_rutter.optimization.objectives import layout

    assert (layout.ML, layout.SPIN_COS, layout.SPIN_SIN) == (0, 1, 2)
    assert layout.ML < layout.REDUCED_PER_PROBE_VARS
    assert layout.SPIN_SIN < layout.REDUCED_PER_PROBE_VARS


def test_the_block_helpers_match_the_arithmetic_they_replaced() -> None:
    from aind_rutter.optimization.objectives import layout

    for n_arcs in (0, 1, 3):
        for i in range(4):
            assert layout.reduced_block(n_arcs, i) == n_arcs + 3 * i
            assert layout.full_block(n_arcs, i) == n_arcs + 6 * i
        for n_probes in range(5):
            assert layout.reduced_n_vars(n_arcs, n_probes) == n_arcs + 3 * n_probes
            assert layout.full_n_vars(n_arcs, n_probes) == n_arcs + 6 * n_probes


def test_no_module_spells_the_reduced_stride_out() -> None:
    """The regression this consolidation exists to prevent."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "aind_rutter" / "optimization"
    pattern = re.compile(r"n_arcs \+ 3 ?\* ?[A-Za-z_]")
    offenders = [
        f"{path.name}:{n}"
        for path in root.rglob("*.py")
        if "__pycache__" not in str(path)
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, f"reduced stride written out at {offenders}"


def test_the_named_shank_tip_is_derived_once() -> None:
    """The app's Tip-RAS readout and the rig export both need this.

    They had their own copies, so a disagreement meant the number on screen was
    not the number handed to the rig.
    """
    from aind_rutter.domain.pose import named_shank_tip_world

    rotation = np.eye(3)
    tips = np.array([[0.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.5, 0.0]])
    pose_tip = np.array([1.0, 2.0, 3.0])

    # `position_bearing_shank` is a 1-based position, not an index.
    np.testing.assert_allclose(
        named_shank_tip_world(pose_tip, rotation, tips, 1), pose_tip
    )
    np.testing.assert_allclose(
        named_shank_tip_world(pose_tip, rotation, tips, 3), pose_tip + tips[2]
    )


def test_a_shank_number_past_the_mesh_clamps() -> None:
    """A four-shank number on a one-shank mesh reads the shank that exists."""
    from aind_rutter.domain.pose import named_shank_tip_world

    tips = np.array([[0.0, 0.0, 0.0]])
    tip = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(named_shank_tip_world(tip, np.eye(3), tips, 4), tip)


def test_a_mesh_with_no_tips_reads_back_the_pose_tip() -> None:
    from aind_rutter.domain.pose import named_shank_tip_world

    tip = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        named_shank_tip_world(tip, np.eye(3), np.zeros((0, 3)), 2), tip
    )


def test_the_brain_mesh_falls_back_only_when_asked() -> None:
    """The export can be handed a catalog with no scene, where the brain is
    authored directly in LPS. The app always has a scene, and falling back to
    the untransformed mesh there is the failure the helper exists to prevent:
    a ray cast would compare a world-LPS tip against file coordinates."""
    from types import SimpleNamespace

    from aind_rutter.runtime.scene_geometry import brain_world_mesh

    mesh = SimpleNamespace(raw="raw-mesh")
    catalog = SimpleNamespace(assets={"brain": SimpleNamespace(mesh=mesh)})

    assert brain_world_mesh(catalog, None, fallback_to_raw=True) == "raw-mesh"
    assert brain_world_mesh(catalog, None) is None


def test_the_named_shank_tip_survives_rotation() -> None:
    """The offset is applied in world, so it turns with the probe."""
    from aind_rutter.domain.pose import named_shank_tip_world

    tips = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(
        named_shank_tip_world(np.zeros(3), quarter_turn, tips, 2),
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )


def test_the_layout_helpers_add_no_work_to_a_traced_kernel() -> None:
    """The helpers replaced index arithmetic inside jitted code, so they have to
    trace to the same graph rather than merely the same answer.

    `spin_restore` walks probes with a `lax.fori_loop`, so its probe index is
    traced and every scalar op in that body is real.
    """
    import jax
    import jax.numpy as jnp

    from aind_rutter.optimization.objectives.layout import (
        ML,
        SPIN_COS,
        SPIN_SIN,
        reduced_block,
        reduced_var,
    )

    n_arcs = 3
    y = jnp.arange(n_arcs + 12, dtype=jnp.float32)
    traced = jnp.int32(2)

    def block_then_offset(y, i):
        """What the four multi-slot sites do: one block, then offsets."""
        off = reduced_block(n_arcs, i)
        return y[off], y[off + SPIN_COS], y[off + SPIN_SIN]

    def multi_slot_literal(y, i):
        off = n_arcs + 3 * i
        return y[off], y[off + 1], y[off + 2]

    assert str(jax.make_jaxpr(block_then_offset)(y, traced)) == str(
        jax.make_jaxpr(multi_slot_literal)(y, traced)
    )

    def single_slot(y, i):
        """What `spin_restore` does for `ml`. `reduced_var` skips the `+ 0`,
        which XLA does not fold."""
        return y[reduced_var(n_arcs, i, ML)]

    def single_slot_literal(y, i):
        return y[n_arcs + 3 * i]

    assert str(jax.make_jaxpr(single_slot)(y, traced)) == str(
        jax.make_jaxpr(single_slot_literal)(y, traced)
    )
