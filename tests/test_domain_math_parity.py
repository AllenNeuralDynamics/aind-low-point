"""One definition each for the quantities several layers compute.

The survey found the probe pivot derived in four places and the per-probe
variable count declared in five. A second copy is not a style problem: the app
draws one pivot and the optimizer optimizes another, and nothing says so.
"""

from __future__ import annotations

import numpy as np
import pytest

from aind_rutter.optimization.geometry.recording import (
    RECORDING_GEOMETRY,
    pivot_from_shank_tips,
)

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
