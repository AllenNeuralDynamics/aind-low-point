"""The user guide's worked example has to load.

`docs/source/configuration.rst` is what someone writing a config for a new
subject reads. Its Complete Example did not validate — an asset template carried
a `transform`, which templates do not accept, and a transform recipe was missing
its `sequence:` wrapper — so following the documentation produced a config the
loader refused.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from aind_rutter.config import ConfigModel

DOC = Path(__file__).resolve().parents[1] / "docs" / "source" / "configuration.rst"


def _example(name: str) -> str:
    """The YAML of the doc section headed `name`, dedented out of its block."""
    text = DOC.read_text()
    marker = f"{name}\n{'-' * len(name)}\n"
    assert marker in text, f"{name}: section is gone from the guide"
    block = text.split(marker, 1)[1].split(".. code-block:: yaml\n", 1)[1]
    lines: list[str] = []
    for line in block.splitlines():
        if line.strip() and not line.startswith("    "):
            break
        lines.append(line[4:] if line.startswith("    ") else line)
    return "\n".join(lines)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """Somewhere the example's transform file actually exists.

    A transform's `path` is existence-checked; an asset's `src` is not.
    """
    (tmp_path / "transforms").mkdir()
    (tmp_path / "transforms" / "headframe.h5").write_bytes(b"")
    return tmp_path


def test_the_complete_example_validates(data_root: Path) -> None:
    doc = yaml.safe_load(_example("Complete Example"))
    doc["paths"]["data_root"] = str(data_root)
    resolved = OmegaConf.to_container(OmegaConf.create(doc), resolve=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a near-miss tag would fail here too
        cfg = ConfigModel.model_validate(resolved)

    assert [str(a.key) for a in cfg.assets] == [
        "structure:PL",
        "structure:MD",
        "structure:CLA",
        "probe:2.1",
    ]
    assert sorted(str(t.key) for t in cfg.targets) == [
        "target:CLA",
        "target:MD",
        "target:PL",
    ]


def test_the_example_declares_what_the_loader_now_requires(data_root: Path) -> None:
    """`mr_signal` on every spec, and a probe kind with recording geometry."""
    doc = yaml.safe_load(_example("Complete Example"))
    doc["paths"]["data_root"] = str(data_root)
    cfg = ConfigModel.model_validate(
        OmegaConf.to_container(OmegaConf.create(doc), resolve=True)
    )
    assert all(s.mr_signal is not None for s in [*cfg.assets, *cfg.targets])
    assert {d.kind for d in cfg.plan.probes.values()} == {"2.1"}


def test_the_minimal_example_validates() -> None:
    """The first config the guide shows."""
    cfg = ConfigModel.model_validate(yaml.safe_load(_example("Overview")))
    assert cfg.assets
