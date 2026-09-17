"""`build_trame_app` on a synthetic subject, with no display.

The web frontend had no test at all, so a change to the render, collision or
store wiring surfaced only when someone launched the app. Building the server is
enough to exercise all of it: the plotter, both adapters, the collision worker
and the controller's reactive state are wired during the build, and only
``server.start()`` needs a browser.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import pyvista as pv
import yaml

from aind_rutter.config import ConfigModel
from tests.synthetic_subject import SyntheticSubject, write_subject

APPLIED_SPIN_DEG = 137.0


@pytest.fixture(scope="module", autouse=True)
def offscreen() -> Iterator[None]:
    """Render to a buffer: the test machine may have no X display."""
    previous = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    try:
        yield
    finally:
        pv.OFF_SCREEN = previous


@pytest.fixture(scope="module")
def subject(tmp_path_factory: pytest.TempPathFactory) -> SyntheticSubject:
    return write_subject(tmp_path_factory.mktemp("subject"))


def _build(subject: SyntheticSubject, **kwargs):
    from aind_rutter.app import build_trame_app

    return build_trame_app(ConfigModel.from_yaml(subject.config), **kwargs)


def test_app_builds_and_publishes_the_plan_to_its_state(
    subject: SyntheticSubject,
) -> None:
    state = _build(subject).state
    assert state.probes == ["P1", "P2"]
    assert state.probe == "P1"
    assert state.arcs == ["a", "b"]
    assert state.targets == ["target:brain"]
    assert state.probe_kinds == ["2.1", "quadbase-alpha"]


def test_selected_probe_readouts_follow_the_plan(subject: SyntheticSubject) -> None:
    state = _build(subject).state
    # P1 is selected first: arc a at +10 deg, no spin, no offsets.
    assert state.ap_tilt == pytest.approx(10.0)
    assert state.spin == 0
    assert state.depth == pytest.approx(0.0)
    assert state.offset_r == pytest.approx(0.0)
    assert state.offset_a == pytest.approx(0.0)


def test_a_plan_file_can_be_applied_at_startup(
    subject: SyntheticSubject, tmp_path: Path
) -> None:
    config = ConfigModel.from_yaml(subject.config)
    plan = config.plan.model_copy(deep=True)
    plan.probes["P1"].spin = APPLIED_SPIN_DEG
    plan_path = tmp_path / "plan.yml"
    plan_path.write_text(yaml.safe_dump(plan.model_dump(mode="json")))

    state = _build(subject, plan_path=plan_path, apply_plan_on_start=True).state
    assert state.spin == int(APPLIED_SPIN_DEG)


def test_a_missing_plan_file_does_not_stop_the_build(
    subject: SyntheticSubject, tmp_path: Path
) -> None:
    state = _build(
        subject, plan_path=tmp_path / "absent.yml", apply_plan_on_start=True
    ).state
    assert state.probes == ["P1", "P2"]
