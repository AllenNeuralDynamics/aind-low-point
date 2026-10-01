"""Propose a set's allocations, written in the form the spec states them.

A proposal is not used until it is pasted into the spec: an allocation there is
a decision someone made, and a rerun of the proposer cannot change it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from aind_rutter.series.configs import set_config
from aind_rutter.series.spec import ProbeSpec, Score, SeriesSpec

if TYPE_CHECKING:
    from aind_rutter.optimization.pipeline.propose import Proposals

PROPOSED_DIR = "proposed"


def proposed_path(run_dir: Path, subject: str, set_name: str) -> Path:
    return run_dir / subject / PROPOSED_DIR / f"{set_name}.yml"


def propose_set(
    spec: SeriesSpec, run_dir: str | Path, subject: str, set_name: str
) -> Proposals:
    """Proposals for one set on one subject's anatomy and implant."""
    # The pipeline loads jax, which this package otherwise does without.
    from aind_rutter.optimization.pipeline.propose import ProposeOptions, propose

    subject_spec = spec.subjects[subject]
    kind = next(iter(spec.inventory))
    # Any probes that name every region will do: only the regions are read.
    probes = [ProbeSpec(kind=kind, regions=(r,)) for r in spec.sets[set_name].regions]
    base = yaml.safe_load(subject_spec.base_config.read_text())
    config = proposed_path(Path(run_dir), subject, set_name).with_suffix(".regions.yml")
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        yaml.safe_dump(
            set_config(spec, subject, set_name, probes, base), sort_keys=False
        )
    )
    return propose(
        config,
        subject_spec.holes,
        spec.inventory,
        ProposeOptions(
            radius_mm=spec.solve_for(set_name).recorded_radius_mm,
            retro=spec.sets[set_name].score is Score.RETRO,
            **spec.propose.model_dump(),
        ),
    )


def _ras(point_LPS: Any) -> list[float]:
    x, y, z = (float(v) for v in point_LPS)
    return [round(-x, 4), round(-y, 4), round(z, 4)]


def proposed_allocations(result: Proposals) -> dict[str, Any]:
    """The proposals as spec allocations, with what each is predicted to record.

    Named by probe count, ``6p``, and ``6p-2`` on for the next best at a count.
    """
    allocations: dict[str, Any] = {}
    predicted: dict[str, Any] = {}
    seen: dict[int, int] = {}
    for proposal in result.proposals:
        n = len(proposal.probes)
        seen[n] = seen.get(n, 0) + 1
        name = f"{n}p" if seen[n] == 1 else f"{n}p-{seen[n]}"
        probes = []
        for probe in proposal.probes:
            stated: dict[str, Any] = {
                "kind": probe.kind,
                "regions": list(probe.regions),
            }
            if probe.points_LPS:
                stated["points_RAS"] = {
                    region: _ras(point) for region, point in probe.points_LPS.items()
                }
            probes.append(stated)
        allocations[name] = probes
        predicted[name] = {
            "mean": round(proposal.mean_share, 4),
            "regions": {
                region: round(share, 4)
                for probe in proposal.probes
                for region, share in probe.share.items()
            },
            "bores": {"_".join(p.regions): p.hole for p in proposal.probes},
        }
    return {"allocations": allocations, "predicted": predicted}


def proposal_report(set_name: str, result: Proposals, written: dict[str, Any]) -> str:
    lines = [
        f"{set_name}: share of each region's {result.cloud} a free line records",
        "  alone: "
        + "; ".join(
            f"{region} "
            + ", ".join(f"{share:.2f} on {kind}" for kind, share in by_kind.items())
            for region, by_kind in result.alone.items()
        ),
    ]
    for name, probes in written["allocations"].items():
        predicted = written["predicted"][name]
        lines.append(
            f"  {name}: mean {predicted['mean']:.3f}  "
            + ", ".join(
                "+".join(
                    f"{region} {predicted['regions'][region]:.2f}"
                    for region in probe["regions"]
                )
                + f" ({probe['kind']})"
                for probe in probes
            )
        )
    return "\n".join(lines)


def write_proposals(
    spec: SeriesSpec,
    run_dir: str | Path,
    *,
    subjects: Sequence[str] | None = None,
    sets: Sequence[str] | None = None,
    report: Callable[[str], None] = print,
) -> list[Path]:
    """Write ``<run>/<subject>/proposed/<set>.yml`` for each set asked for."""
    run_dir = Path(run_dir)
    written = []
    for subject in subjects or spec.subjects:
        for set_name in spec.sets_of(subject):
            if sets and set_name not in sets:
                continue
            result = propose_set(spec, run_dir, subject, set_name)
            data = proposed_allocations(result)
            path = proposed_path(run_dir, subject, set_name)
            header = (
                f"# {spec.name}: proposed allocations of set {set_name} on subject "
                f"{subject}.\n# To use one, paste it under sets.{set_name}."
                f"allocations in the spec.\n# predicted: the share of each region's "
                f"{result.cloud} within "
                f"{spec.solve_for(set_name).recorded_radius_mm:g} mm of a line no "
                f"fixture or other probe has moved.\n"
            )
            path.write_text(
                header
                + yaml.safe_dump(
                    {"alone": result.alone, **data},
                    sort_keys=False,
                    default_flow_style=None,
                    width=100,
                )
            )
            report(proposal_report(set_name, result, data))
            report(f"  written to {path}")
            written.append(path)
    return written
