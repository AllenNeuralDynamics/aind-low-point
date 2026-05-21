"""RuntimeBundle + build_runtime_from_config orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import trimesh
from aind_mri_utils.reticle_calibrations import find_probe_angle

from aind_low_point.assets import AssetCatalog, AssetSpec, TargetSpec
from aind_low_point.common import Capability, Kind
from aind_low_point.config import (
    AssetSpecModel,
    BaseSpecModel,
    CollisionPolicyModel,
    ConfigModel,
    HeadMountModel,
    MaterialModel,
    ResourceModel,
    TargetSpecModel,
    _merge_dict_shallow,
)
from aind_low_point.core import (
    AffineTransform,
    Float3,
    Material,
    MeshTransformable,
    PointsTransformable,
)
from aind_low_point.planning import Kinematics, PlanningState, ProbePlan
from aind_low_point.runtime.calibration import _get_calibration_rt
from aind_low_point.runtime.canonicalize import (
    CanonicalizationRuntime,
    _apply_canonicalization_mesh,
    _apply_canonicalization_points,
    _resolve_canon_model_to_runtime,
    _resolve_scene_node_transform,
)
from aind_low_point.runtime.chem_shift import ChemShiftContext, _should_apply_chem
from aind_low_point.runtime.loaders import (
    GeometryOut,
    load_geometry,
)
from aind_low_point.runtime.reducers import _REDUCER_REGISTRY
from aind_low_point.runtime.transforms import compile_all_transforms
from aind_low_point.scene import NodeInstance, Scene, resolve_base_geometry


def _build_subject_from_rig(model: HeadMountModel) -> AffineTransform:
    """Build an AffineTransform rotation from a HeadMountModel.

    The model carries an axis-angle in subject-LPS basis. When the angle
    is zero (default), returns identity.
    """
    angle_deg = float(model.angle_deg)
    if abs(angle_deg) < 1e-12:
        return AffineTransform.identity()
    axis = np.asarray(model.axis_LPS, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return AffineTransform.identity()
    axis = axis / n
    angle_rad = np.deg2rad(angle_deg)
    # Rodrigues' formula. R = I + sin(θ)K + (1-cos(θ))K² where K is the
    # cross-product matrix of `axis`.
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    R = (
        np.eye(3, dtype=np.float64)
        + np.sin(angle_rad) * K
        + (1.0 - np.cos(angle_rad)) * (K @ K)
    )
    return AffineTransform(rotation=R)


def _compile_collision_labels(labels_in_use: Iterable[str]) -> dict[str, int]:
    """
    Assign each collision label a bit. Bit 0 is reserved for 'NONE' (unused).
    """
    labels = [lab for lab in dict.fromkeys(labels_in_use) if lab]  # unique, drop falsy
    mapping: dict[str, int] = {}
    for i, lab in enumerate(labels, start=1):  # start bits at 1
        mapping[lab] = 1 << i
    return mapping


def _material_from_model(m: MaterialModel) -> Material:
    return Material(
        name=m.name,
        color_hex_str=m.color,
        opacity=m.opacity,
        wireframe=m.wireframe,
        visible=m.visible,
        point_size=m.point_size,
    )


def resolve_material_for_spec(
    spec_like,  # AssetSpecModel or TargetSpecModel (post-template-merge)
    registry: dict[str, MaterialModel],  # config.materials
) -> "Material":
    base = registry.get(spec_like.material_ref) if spec_like.material_ref else None
    base_d = base.model_dump(exclude_unset=True) if base else None
    over_d = (
        spec_like.material.model_dump(exclude_unset=True)
        if spec_like.material
        else None
    )
    merged = _merge_dict_shallow(base_d, over_d)
    mm = MaterialModel(**(merged or {}))
    return _material_from_model(mm)


@dataclass(frozen=True)
class CollisionLabelIndex:
    label_to_bit: dict[str, int]
    bit_to_label: dict[int, str]


@dataclass(frozen=True)
class RuntimeBundle:
    # Catalog after loading/canonicalizing
    asset_catalog: AssetCatalog  # runtime AssetSpec (with Mesh/PointsTransformable)
    targets_pts: dict[str, np.ndarray]  # key -> (N,3) points in LPS mm
    # Scene ready to render
    scene: Scene
    # Collision label bits (for adapters)
    collision_labels: CollisionLabelIndex
    plan_state: PlanningState


def _capabilities_from_list(lst) -> Capability:
    val = Capability(0)
    for c in lst or []:
        if isinstance(c, Capability):
            val |= c
        elif isinstance(c, int):
            val |= Capability(c)
        elif isinstance(c, str):
            val |= Capability[c.upper()]
        else:
            val |= Capability(int(c))
    return val


def _collision_bits(
    policy: CollisionPolicyModel, label_to_bit: dict[str, int]
) -> tuple[int, int]:
    group_bits = label_to_bit.get(policy.group or "", 0)
    mask_bits = 0
    for lab in policy.mask:
        mask_bits |= label_to_bit.get(lab, 0)
    return group_bits, mask_bits


def _base_spec_kwargs_from_model(
    m: BaseSpecModel,
    label_to_bit: dict[str, int],
    material_models: dict[str, MaterialModel] = {},
) -> dict[str, Any]:
    group_bits, mask_bits = _collision_bits(m.collision, label_to_bit)
    material = resolve_material_for_spec(m, material_models)
    return dict(
        key=m.key,
        kind=m.kind.value,  # "mesh" | "points" | "lines"
        role=m.role,  # keep enum if your runtime type expects it; else use m.role.value
        default_material=material,
        metadata=dict(m.metadata),
        tags=set(m.tags),
        caps=_capabilities_from_list(m.caps),
        collidable_group=group_bits,
        collidable_mask=mask_bits,
        pivot_LPS=np.array(m.pivot_LPS, float) if m.pivot_LPS else None,
        bbox_hint=np.array(m.bbox_hint, float) if m.bbox_hint else None,
    )


def _load_geo(
    spec: BaseSpecModel,
    chem: ChemShiftContext,
    canon: Optional[CanonicalizationRuntime],
) -> Optional[GeometryOut]:
    if not (spec.src and spec.loader):
        return None

    loader_kwargs = spec.loader_kwargs or {}
    geo = load_geometry(Path(spec.src), loader=spec.loader, **loader_kwargs)

    if isinstance(geo, trimesh.Trimesh):
        if canon:
            geo = _apply_canonicalization_mesh(
                geo,
                canon.source_space,
                canon.scale_to_mm,
                canon.transform_file_to_canonical,
            )
        if _should_apply_chem(spec, chem):
            tf = chem.pt_transform_for_ppm(spec.chem_shift_ppm)
            shifted = tf.apply_to(geo.vertices)
            geo = trimesh.Trimesh(vertices=shifted, faces=geo.faces, process=False)
        return geo

    if isinstance(geo, np.ndarray):
        if canon:
            geo = _apply_canonicalization_points(
                geo,
                canon.source_space,
                canon.scale_to_mm,
                canon.transform_file_to_canonical,
            )
        if _should_apply_chem(spec, chem):
            tf = chem.pt_transform_for_ppm(spec.chem_shift_ppm)
            geo = tf.apply_to(geo)
        return geo

    if isinstance(geo, dict):
        # Named point collection (e.g. fcsv). Already in LPS; skip
        # canonicalization. Chem shift still governed by config policy.
        if _should_apply_chem(spec, chem):
            tf = chem.pt_transform_for_ppm(spec.chem_shift_ppm)
            geo = {k: tf.apply_to(v) for k, v in geo.items()}
        return geo

    raise TypeError(f"Unexpected geometry type from loader: {type(geo)}")


# --- asset builder ------------------------------------------------------------


def _default_probe_pivot_local(
    a: AssetSpecModel,
    geo: GeometryOut,
) -> Optional[np.ndarray]:
    """Auto-derive ``pivot_LPS`` for a probe asset: ``(centroid_x, centroid_y,
    active_center_mm)`` where the centroid is computed from the canonicalized
    mesh's shank tips (any direction) and ``active_center_mm`` is looked up
    from :data:`RECORDING_GEOMETRY` for the kind suffix.

    Returns ``None`` if the asset is not a probe, the kind isn't registered,
    or the mesh has no detectable shank tips.
    """
    from aind_low_point.optimization.recording import RECORDING_GEOMETRY
    from aind_low_point.runtime.shanks import detect_shank_tips_local

    if not isinstance(a.key, str) or not a.key.startswith("probe:"):
        return None
    if not isinstance(geo, trimesh.Trimesh):
        return None
    kind = a.key.split(":", 1)[1]
    geom = RECORDING_GEOMETRY.get(kind)
    if geom is None:
        return None
    tips = detect_shank_tips_local(geo)
    if tips.shape[0] == 0:
        return None
    return np.array(
        [
            float(tips[:, 0].mean()),
            float(tips[:, 1].mean()),
            float(geom.active_center_mm),
        ],
        dtype=np.float64,
    )


def build_asset_spec(
    a: AssetSpecModel,
    base_kwargs: dict[str, Any],
    chem: ChemShiftContext,
    canon: Optional[CanonicalizationRuntime],
) -> AssetSpec:
    from aind_low_point.optimization.headstages import build_headstage_hull

    mesh_tf: MeshTransformable | None = None
    pts_tf: PointsTransformable | None = None

    geo = _load_geo(a, chem, canon)
    if a.kind == Kind.MESH:
        if not isinstance(geo, trimesh.Trimesh):
            raise TypeError(f"Asset '{a.key}' loader returned points but kind=MESH")
        mesh_tf = MeshTransformable(geo)
    elif a.kind == Kind.POINTS:
        if isinstance(geo, trimesh.Trimesh):
            raise TypeError(f"Asset '{a.key}' loader returned mesh but kind=POINTS")
        pts_tf = PointsTransformable(geo)
    elif a.kind == Kind.LINES:
        raise NotImplementedError("kind='lines' not implemented in loader")

    # Auto-compute kinematic pivot for probe assets (unless the user
    # set ``pivot_LPS`` explicitly in config). Pivot lives in the
    # canonicalized local frame; per-frame planning poses then place
    # this point at the user-selected target.
    if base_kwargs.get("pivot_LPS") is None:
        default_pivot = _default_probe_pivot_local(a, geo)
        if default_pivot is not None:
            base_kwargs = {**base_kwargs, "pivot_LPS": default_pivot}

    # Per-kind headstage convex hull for the placement optimizer's
    # clearance constraint. Computed from the canonical mesh's "body"
    # region (above the shanks); pipettes / degenerate fixtures return
    # ``None`` and are skipped from pairwise clearance checks.
    headstage_hull = None
    if isinstance(a.key, str) and a.key.startswith("probe:") and mesh_tf is not None:
        headstage_hull = build_headstage_hull(mesh_tf.raw)

    return AssetSpec(
        **base_kwargs,
        source_path=Path(a.src) if a.src else None,
        loader=a.loader,
        mesh=mesh_tf,
        points=pts_tf,
        headstage_hull=headstage_hull,
    )


def build_target_spec(
    t: TargetSpecModel,
    runtime_assets: dict[str, AssetSpec],
    base_kwargs: dict[str, Any],
    chem: ChemShiftContext,
    canon: Optional[CanonicalizationRuntime] = None,
    reducer_registry: dict[str, Callable[..., np.ndarray]] = _REDUCER_REGISTRY,
    resource_registry: dict[str, GeometryOut] = {},
) -> tuple[TargetSpec, np.ndarray]:
    # Targets must be non-collidable by default; enforce here (even if config forgot).
    base_kwargs["caps"] = Capability.RENDERABLE
    base_kwargs["collidable_group"] = 0
    base_kwargs["collidable_mask"] = 0

    # Resolve points (explicit file or derived by reducer)
    if t.src and t.loader:
        geo = _load_geo(t, chem, canon)
    elif t.source_key:
        # Derived: fetch source asset geometry first
        src_key = t.source_key or ""
        src_asset = runtime_assets.get(src_key)
        if src_asset is None:
            raise KeyError(
                f"Target '{t.key}' source_key '{src_key}' not found in loaded assets"
            )
        if src_asset.mesh:
            geo = src_asset.mesh.raw
        elif src_asset.points:
            geo = src_asset.points.raw
        else:
            raise ValueError(
                f"Target '{t.key}': source asset '{src_key}' has no geometry loaded"
            )
    elif t.from_resource and t.selector:
        geo = t.selector.select(resource_registry[t.from_resource])
    else:
        raise ValueError(f"Target '{t.key}' has no src, source_key, or from_resource")

    # Apply reducer if present
    if t.reducer:
        reducer_fn = reducer_registry.get(t.reducer)
        if reducer_fn is None:
            raise KeyError(f"Unknown target reducer '{t.reducer}' for target '{t.key}'")

        geo = reducer_fn(geo, **(t.reducer_kwargs or {}))  # should return (3,) or (1,3)
        geo = np.asarray(geo, dtype=np.float64).reshape(1, 3)

    if isinstance(geo, trimesh.Trimesh):
        raise TypeError(f"Target '{t.key}' loader returned mesh; expected points")
    geo = np.asarray(geo, dtype=np.float64)
    if geo.ndim == 1:
        geo = geo.reshape(1, 3)

    spec = TargetSpec(
        **base_kwargs,
        source_path=Path(t.src) if t.src else None,
        loader=t.loader,
        source_key=t.source_key,
        reducer=t.reducer,
        reducer_kwargs=dict(t.reducer_kwargs),
        points=PointsTransformable(geo),
        approach_vector=np.array(t.approach_vector, float)
        if t.approach_vector
        else None,
        uncertainty_mm=t.uncertainty_mm,
    )
    return spec, geo


def load_resource(
    r: ResourceModel,
    chem: ChemShiftContext,
    canon: Optional[CanonicalizationRuntime] = None,
) -> GeometryOut:
    # Load the resource using the provided context and canonicalization
    geo = _load_geo(r, chem, canon)
    return geo


def build_runtime_from_config(cfg: ConfigModel) -> RuntimeBundle:  # noqa: C901
    # 1) collision labels → bit mapping
    labels: list[str] = []
    for a in cfg.assets:
        if a.collision.group:
            labels.append(a.collision.group)
        labels.extend(a.collision.mask)
    for t in cfg.targets:
        if t.collision.group:
            labels.append(t.collision.group)
        labels.extend(t.collision.mask)
    label_to_bit = _compile_collision_labels(labels)
    bit_to_label = {v: k for k, v in label_to_bit.items()}
    label_index = CollisionLabelIndex(
        label_to_bit=label_to_bit, bit_to_label=bit_to_label
    )
    # 2) assets
    chem = ChemShiftContext.from_config(cfg)
    compiled_transforms = compile_all_transforms(cfg.transforms)
    runtime_assets: dict[str, AssetSpec] = {}
    for a in cfg.assets:
        maybe_cannon = _resolve_canon_model_to_runtime(a, cfg, compiled_transforms)
        base_kwargs = _base_spec_kwargs_from_model(a, label_to_bit, cfg.materials)
        runtime_assets[a.key] = build_asset_spec(a, base_kwargs, chem, maybe_cannon)

    # 3) resources
    runtime_resources: dict[str, GeometryOut] = {}
    for r in cfg.resources:
        maybe_cannon = _resolve_canon_model_to_runtime(r, cfg, compiled_transforms)
        runtime_resources[r.key] = load_resource(r, chem, maybe_cannon)

    # 4) targets (specs + points index)
    runtime_targets: dict[str, TargetSpec] = {}
    target_index: dict[str, Float3] = {}
    for t in cfg.targets:
        maybe_cannon = _resolve_canon_model_to_runtime(t, cfg, compiled_transforms)
        base_kwargs = _base_spec_kwargs_from_model(t, label_to_bit, cfg.materials)
        tspec, pts = build_target_spec(
            t,
            runtime_assets,
            base_kwargs,
            chem,
            maybe_cannon,
            _REDUCER_REGISTRY,
            resource_registry=runtime_resources,
        )
        runtime_targets[tspec.key] = tspec
        target_index[tspec.key] = pts

    catalog = AssetCatalog(assets=runtime_assets, targets=runtime_targets)

    # 5) scene
    scene = Scene()
    for n in cfg.scene.nodes:
        asset_key = n.asset
        if asset_key not in runtime_assets and asset_key not in runtime_targets:
            raise KeyError(
                f"Scene node '{n.key}' references unknown asset '{asset_key}'"
            )

        node_tf = _resolve_scene_node_transform(n.transform, compiled_transforms)

        extras: dict[str, Any] = {}
        locked_axes: set[str] = set()
        if n.pose_source_probe:
            extras["pose_source_probe"] = n.pose_source_probe
            decl = cfg.plan.probes.get(n.pose_source_probe)
            if decl and decl.calibrated:
                locked_axes.update({"ap_tilt", "ml_tilt"})

        scene.upsert(
            NodeInstance(
                key=n.key,
                asset_key=asset_key,
                transform=node_tf,
                tags=set(n.tags),
                material_override=None,
                enabled=True,
                locked_axes=locked_axes,
                extras=extras,
            )
        )

    # 5b) Resolve ALL target positions through scene transforms
    for key in list(runtime_targets):
        transformed = resolve_base_geometry(catalog, scene, key)
        if transformed is not None:
            target_index[key] = transformed.raw

    # 6) kinematics, calibrations, plans (build PlanningState)
    kinematics = Kinematics(
        arc_angles=dict(cfg.plan.arcs),
        subject_from_rig=_build_subject_from_rig(cfg.plan.subject_from_rig),
    )
    calibrations = _get_calibration_rt(cfg.plan.calibrations, cfg.plan.reticles)
    probes: dict[str, ProbePlan] = {}
    for probe_name, probe_decl in cfg.plan.probes.items():
        probe_calibrated = probe_name in calibrations
        if probe_calibrated:
            ap, ml = find_probe_angle(calibrations[probe_name].rotation)
        elif probe_decl.ap_local is not None:
            ap = probe_decl.ap_local
            ml = probe_decl.slider_ml
        elif probe_decl.arc is not None:
            ap = kinematics.get_arc(probe_decl.arc)
            ml = probe_decl.slider_ml
        else:
            ap = 0.0
            ml = probe_decl.slider_ml
        # Resolve target: inline RAS point, node, or catalog key
        if probe_decl.target.kind == "inline":
            target_key = None
            target_point_RAS = tuple(probe_decl.target.point_RAS)
        elif probe_decl.target.kind == "node":
            key = probe_decl.target.key
            transformed_points = resolve_base_geometry(catalog, scene, key)
            if not transformed_points:
                raise RuntimeError(
                    f"Probe '{probe_name}' references unknown target "
                    f"'{probe_decl.target.key}'"
                )
            transformed_points = transformed_points.raw
            target_index[key] = transformed_points
            target_key = key
            target_point_RAS = None
        else:  # catalog
            target_key = probe_decl.target.key
            target_point_RAS = None
        probes[probe_name] = ProbePlan(
            kind=probe_decl.kind,
            arc_id=probe_decl.arc,
            bind_ap_to_arc=probe_decl.bind_ap_to_arc,
            ap_local=ap,
            ml_local=ml,
            spin=probe_decl.spin,
            past_target_mm=probe_decl.past_target_mm,
            offsets_RA=tuple(probe_decl.offsets_RA),
            target_key=target_key,
            target_point_RAS=target_point_RAS,
            position_bearing_shank=probe_decl.position_bearing_shank,
            calibrated=probe_decl.calibrated,
        )
    plan_state = PlanningState(
        kinematics=kinematics,
        probes=probes,
        calibrations=calibrations,
        target_index=target_index,
    )

    return RuntimeBundle(
        asset_catalog=catalog,
        targets_pts=target_index,
        scene=scene,
        collision_labels=label_index,
        plan_state=plan_state,
    )
