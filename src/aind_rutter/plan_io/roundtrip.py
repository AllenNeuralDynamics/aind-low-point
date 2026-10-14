"""Writing a planning state back out as a plan config."""

from __future__ import annotations

import re

import numpy as np
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system

from aind_rutter.config import (
    CatalogTargetRefModel,
    ConfigModel,
    InlineTargetRefModel,
    NodeTargetRefModel,
    PlanningModel,
    ProbeDeclModel,
)
from aind_rutter.domain.plan import PlanningState, ProbePlan


def _reconstruct_target_ref(
    probe: ProbePlan,
    original_probes: dict[str, ProbeDeclModel],
    probe_name: str,
) -> "CatalogTargetRefModel | NodeTargetRefModel | InlineTargetRefModel":
    """Reconstruct a TargetRef from a ProbePlan.

    Priority:
    1. If target_point_LPS is set → InlineTargetRefModel (converted to RAS)
    2. If target_key matches the original → reuse original TargetRef (preserves kind)
    3. Otherwise → CatalogTargetRefModel
    """
    if probe.target_point_LPS is not None:
        ras = convert_coordinate_system(
            np.asarray(probe.target_point_LPS, float).reshape(1, 3), "LPS", "RAS"
        ).reshape(3)
        return InlineTargetRefModel(point_RAS=[float(v) for v in ras])
    orig = original_probes.get(probe_name)
    if (
        orig is not None
        and hasattr(orig.target, "key")
        and probe.target_key == orig.target.key
    ):
        return orig.target
    if probe.target_key is None:
        return CatalogTargetRefModel(key="")
    return CatalogTargetRefModel(key=probe.target_key)


def planning_state_to_plan_model(
    state: PlanningState,
    original: PlanningModel,
) -> PlanningModel:
    """Convert a mutated PlanningState back to a PlanningModel.

    Parameters
    ----------
    state
        The runtime planning state (possibly mutated by commands).
    original
        The original PlanningModel from the config (used to preserve
        calibrations, reticles, and target ref kinds).

    Returns
    -------
    PlanningModel
        A new PlanningModel reflecting the current state.
    """
    probes: dict[str, ProbeDeclModel] = {}
    for name, plan in state.probes.items():
        target_ref = _reconstruct_target_ref(plan, original.probes, name)
        orig_decl = original.probes.get(name)
        probes[name] = ProbeDeclModel(
            kind=plan.kind,
            arc=plan.arc_id,
            slider_ml=plan.ml_local,
            spin=plan.spin,
            ap_local=plan.ap_local,
            bind_ap_to_arc=plan.bind_ap_to_arc,
            target=target_ref,
            # No command edits these, so the declaration is still what holds.
            also_record=orig_decl.also_record if orig_decl else [],
            regions=orig_decl.regions if orig_decl else [],
            past_target_mm=plan.past_target_mm,
            offsets_RA=[-plan.offsets_LP[0], -plan.offsets_LP[1]],
            position_bearing_shank=plan.position_bearing_shank,
            calibrated=plan.calibrated,
            auto_scene=orig_decl.auto_scene if orig_decl else True,
            scene_tags=orig_decl.scene_tags if orig_decl else ["probe", "dynamic"],
        )

    return PlanningModel(
        arcs=dict(state.kinematics.arc_angles),
        subject_from_rig=original.subject_from_rig,
        rig=original.rig,
        probes=probes,
        reticles=original.reticles,
        calibrations=original.calibrations,
    )


def save_plan_to_config(
    state: PlanningState,
    original_config: ConfigModel,
) -> ConfigModel:
    """Produce a new ConfigModel with the plan section updated from state.

    Everything except ``plan`` is preserved from the original config.
    The returned model can be serialized to YAML via
    ``model.model_dump(mode="json")``.

    Parameters
    ----------
    state
        The runtime planning state.
    original_config
        The original ConfigModel (used as the base for non-plan sections).

    Returns
    -------
    ConfigModel
        A new ConfigModel ready for serialization.
    """
    new_plan = planning_state_to_plan_model(state, original_config.plan)
    data = original_config.model_dump(mode="json")
    data["plan"] = new_plan.model_dump(mode="json")

    # Strip auto-generated scene nodes so the validator can re-generate
    # them for the (possibly changed) set of probes / assets.
    explicit_keys = original_config.scene._explicit_node_keys
    if explicit_keys is not None:
        data["scene"]["nodes"] = [
            n for n in data["scene"]["nodes"] if n["key"] in explicit_keys
        ]

    return ConfigModel.model_validate(data)


def replace_plan_block(written: str, plan: PlanningModel) -> str:
    """``written`` with its top-level ``plan:`` block replaced by ``plan``.

    Every other line is kept as written, so comments, ``paths:`` and each
    ``${…}`` reference survive; a config dumped from the model would carry
    this machine's folders and every template already expanded.
    """
    return replace_top_level_block(written, "plan", plan.model_dump(mode="json"))


def replace_top_level_block(written: str, key: str, value: object) -> str:
    """``written`` with its top-level ``key:`` block replaced by ``value``,
    every other line kept as written. A column-0 comment directly above the
    next key belongs to that key."""
    import yaml

    lines = written.splitlines(keepends=True)
    block = yaml.safe_dump(
        {key: value},
        default_flow_style=False,
        sort_keys=False,
    )
    start = next(
        (i for i, line in enumerate(lines) if re.match(rf"{re.escape(key)}\s*:", line)),
        None,
    )
    if start is None:
        sep = "" if not lines or lines[-1].endswith("\n") else "\n"
        out = written + sep + block
    else:
        end = start + 1
        while end < len(lines) and not _starts_top_level(lines[end]):
            end += 1
        while end > start + 1 and _belongs_to_next(lines[end - 1]):
            end -= 1
        out = "".join(lines[:start]) + block + "".join(lines[end:])

    expected = yaml.safe_load(written) or {}
    expected[key] = value
    try:
        spliced = yaml.safe_load(out)
    except yaml.YAMLError:
        spliced = None
    if spliced != expected:
        raise ValueError(
            f"the config's {key} block could not be replaced in place; its "
            "top level is not one block-style key per line"
        )
    return out


def _starts_top_level(line: str) -> bool:
    return bool(line.strip()) and not line[0].isspace() and not line.startswith("#")


def _belongs_to_next(line: str) -> bool:
    return not line.strip() or line.startswith("#")
