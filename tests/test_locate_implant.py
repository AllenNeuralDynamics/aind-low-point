"""``rutter-locate-implant`` fits an implant's move from hole readings."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from aind_rutter.plan_io.locate_implant import (
    Bore,
    HoleReading,
    ImplantFit,
    Placed,
    fit_implant,
    line_gaps,
    located_config,
    misfit,
    observe,
    parse_reading,
)

DOWN = np.array([0.0, 0.0, -1.0])


def _bores() -> dict[int, Bore]:
    tops = {1: (2.0, -2.0, 0.0), 2: (-1.5, -1.0, 0.2), 3: (0.5, 2.5, -0.3)}
    tilt = np.array([0.2, -0.1, 1.0]) / np.linalg.norm([0.2, -0.1, 1.0])
    return {k: Bore(k, tilt, np.array(t), -0.4) for k, t in tops.items()}


SLANT = np.array([0.6, 0.0, -0.8])  # 37 deg from DOWN


def _reading(
    bore: Bore, mode: str, s: float = -0.2, down: np.ndarray = DOWN, code: str = "c"
) -> Placed:
    """A tip in ``bore`` at ``s`` along its axis, or 1 mm above its middle
    with the probe aimed along ``down`` at it."""
    if mode == "in":
        tip = bore.top + s * bore.axis
    else:
        tip = bore.middle - 1.0 * down
    return Placed(HoleReading(code, bore.id, (0.0, 0.0, 0.0), mode), code, tip, down)  # type: ignore[arg-type]


def test_readings_parse_and_scale() -> None:
    r = parse_reading("50720:2:5964.5,3006.5,12402.5", "in", 1e-3)
    assert (r.probe, r.hole, r.mode) == ("50720", 2, "in")
    assert np.allclose(r.newscale_mm, (5.9645, 3.0065, 12.4025))
    with pytest.raises(ValueError, match="PROBE:HOLE"):
        parse_reading("50720:2", "in", 1.0)


def test_a_tip_in_the_channel_agrees_and_one_beside_it_does_not() -> None:
    bore = _bores()[1]
    assert np.allclose(misfit(_reading(bore, "in"), bore), 0.0)
    below = _reading(bore, "in", s=-0.5)
    assert misfit(below, bore)[-1] == pytest.approx(0.1)
    assert np.allclose(misfit(_reading(bore, "pointing"), bore), 0.0)


def test_one_hole_fits_the_translation_across_its_bore() -> None:
    bores = _bores()
    shift = np.array([0.3, -0.4, 0.0])
    moved = bores[2].moved(np.eye(3), shift)
    fit = fit_implant([_reading(moved, "in")], bores)
    across = lambda v: v - (v @ moved.axis) * moved.axis  # noqa: E731
    assert np.allclose(across(fit.translation), across(shift), atol=1e-6)
    assert np.allclose(
        misfit(_reading(moved, "in"), bores[2].moved(np.eye(3), fit.translation)),
        0.0,
        atol=1e-6,
    )


def test_three_holes_recover_a_rotation_and_translation() -> None:
    bores = _bores()
    R = Rotation.from_euler("xyz", [2.0, -1.5, 3.0], degrees=True).as_matrix()
    t = np.array([0.25, -0.35, -0.1])
    readings = [
        _reading(bores[1].moved(R, t), "in"),
        _reading(bores[2].moved(R, t), "pointing"),
        _reading(bores[3].moved(R, t), "in", s=-0.1),
        _reading(bores[1].moved(R, t), "pointing"),
        _reading(bores[3].moved(R, t), "pointing"),
    ]
    fit = fit_implant(readings, bores, np.eye(3))
    assert fit.angle_deg == pytest.approx(
        np.degrees(Rotation.from_matrix(R).magnitude()), abs=0.05
    )
    for b in bores.values():
        assert np.allclose(fit.apply(b.middle), R @ b.middle + t, atol=5e-3)


def test_one_line_of_sight_leaves_the_shift_along_it_free() -> None:
    bores = _bores()
    one_probe = [_reading(b, "pointing") for b in bores.values()]
    obs = observe(one_probe, bores, aim_error_mm=0.05)
    assert len(obs.free_shifts) == 1
    assert abs(obs.free_shifts[0] @ DOWN) == pytest.approx(1.0)
    second = _reading(bores[2], "pointing", down=SLANT, code="d")
    assert observe([*one_probe, second], bores, 0.05).free_shifts == []


def test_auto_fits_only_the_rotations_the_readings_see() -> None:
    bores = _bores()
    one_probe = [_reading(b, "pointing") for b in bores.values()]
    seen = observe(one_probe, bores, 0.05).fitted_axes("auto")
    assert len(seen) == 1 and abs(seen[0] @ DOWN) > 0.99
    both = one_probe + [
        _reading(b, "pointing", down=SLANT, code="d") for b in bores.values()
    ]
    assert len(observe(both, bores, 0.01).fitted_axes("auto")) == 3
    assert len(observe([one_probe[0]], bores, 0.05).fitted_axes("auto")) == 0


def test_two_probes_at_one_hole_report_how_far_their_lines_miss() -> None:
    bore = _bores()[1]
    a = _reading(bore, "pointing")
    b = _reading(bore, "pointing", down=SLANT, code="d")
    [(_, _, angle, gap)] = line_gaps([a, b])
    assert angle == pytest.approx(36.87, abs=0.01) and gap == pytest.approx(
        0.0, abs=1e-9
    )
    off = Placed(b.reading, "d", b.tip + np.array([0.0, 0.1, 0.0]), SLANT)
    assert line_gaps([a, off])[0][3] == pytest.approx(0.1)


def test_the_move_is_appended_and_places_the_implant() -> None:
    written = (
        "version: 1\n"
        "transforms:\n"
        "  implant_to_lps:\n"
        "    kind: sitk_file\n"
        "    path: ${paths.mri}/fit.h5\n"
        "assets:\n"
        "- key: brain\n"
        "  src: b.nii.gz\n"
        "- key: implant\n"
        "  src: i.step\n"
        "plan: {}\n"
    )
    R = Rotation.from_euler("z", 3.0, degrees=True).as_matrix()
    fit = ImplantFit(R, np.array([0.1, -0.2, 0.0]), fit_rotation=True)
    raw = yaml.safe_load(located_config(written, fit, "implant"))
    steps = raw["transforms"]["implant_to_lps"]["sequence"]
    assert steps[0] == {"kind": "sitk_file", "path": "${paths.mri}/fit.h5"}
    assert [s["kind"] for s in steps[1:]] == ["rotate_euler_deg", "translate_mm"]
    assert raw["assets"][1] == {
        "key": "implant",
        "transform": "implant_to_lps",
        "src": "i.step",
    }
    with pytest.raises(ValueError, match="no asset"):
        located_config(written, fit, "implant-2")


def test_the_written_steps_compile_to_the_fitted_move() -> None:
    from aind_rutter.build.transforms import compile_recipe_to_chain
    from aind_rutter.config import TransformRecipeModel

    R = Rotation.from_euler("xyz", [1.0, -2.0, 4.0], degrees=True).as_matrix()
    fit = ImplantFit(R, np.array([0.3, 0.1, -0.2]), fit_rotation=True)
    chain = compile_recipe_to_chain(TransformRecipeModel(sequence=fit.as_ops()))
    pts = np.array([[1.0, 2.0, 3.0], [-0.5, 0.2, 0.0]])
    assert np.allclose(chain.apply_to(pts), fit.apply(pts), atol=1e-6)
