"""Writing a planning state back out as a plan config."""

from __future__ import annotations

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
    1. If target_point_RAS is set → InlineTargetRefModel
    2. If target_key matches the original → reuse original TargetRef (preserves kind)
    3. Otherwise → CatalogTargetRefModel
    """
    if probe.target_point_RAS is not None:
        return InlineTargetRefModel(point_RAS=list(probe.target_point_RAS))
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
            past_target_mm=plan.past_target_mm,
            offsets_RA=list(plan.offsets_RA),
            position_bearing_shank=plan.position_bearing_shank,
            calibrated=plan.calibrated,
            auto_scene=orig_decl.auto_scene if orig_decl else True,
            scene_tags=orig_decl.scene_tags if orig_decl else ["probe", "dynamic"],
        )

    return PlanningModel(
        arcs=dict(state.kinematics.arc_angles),
        subject_from_rig=original.subject_from_rig,
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
