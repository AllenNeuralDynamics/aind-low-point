"""Build runtime from config"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import SimpleITK as sitk
import trimesh
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from aind_anatomical_utils.slicer import read_slicer_fcsv
from aind_mri_utils.chemical_shift import (
    chemical_shift_transform,
    compute_chemical_shift,
)
from aind_mri_utils.meshes import (
    mask_to_trimesh,
)
from aind_mri_utils.reticle_calibrations import (
    find_probe_angle,
    fit_rotation_params_from_manual_calibration,
    fit_rotation_params_from_parallax,
)
from aind_mri_utils.rotations import apply_rotate_translate
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from aind_low_point.assets import (
    AssetCatalog,
    AssetSpec,
    TargetSpec,
)
from aind_low_point.common import Capability, Kind, Role
from aind_low_point.config import (
    AssetSpecModel,
    BaseSpecModel,
    CalibrationReticleModel,
    CalibrationsModel,
    CalibrationSourceModel,
    CanonicalizationDefModel,
    CollisionPolicyModel,
    ConfigModel,
    LoadSITKTxOpModel,
    MaterialModel,
    RotateEulerTxOpModel,
    SourceSpace,
    TargetSpecModel,
    TransformRecipeModel,
    TransformRefModel,
    TranslateTxOpModel,
    _TxOpBase,
    merge_material_cfg,
)
from aind_low_point.core import (
    AffineTransform,
    Float3,
    FloatNx3,
    Material,
    MeshTransformable,
    PointsTransformable,
    TransformChain,
)
from aind_low_point.orientation_codes import OrientationCode
from aind_low_point.planning import Kinematics, PlanningState, ProbePlan
from aind_low_point.scene import NodeInstance, Scene, resolve_base_geometry

FloatAABB = NDArray[np.float64]  # shape (2, 3)


def _op_to_affine(op: _TxOpBase) -> AffineTransform:
    if isinstance(op, TranslateTxOpModel):
        R = np.eye(3)
        t = np.asarray(op.delta, dtype=np.float64)
        return AffineTransform(R, t)
    elif isinstance(op, RotateEulerTxOpModel):
        rotation = Rotation.from_euler(op.order, op.angles_deg, degrees=True)
        # Get the rotation matrix
        R = rotation.as_matrix()
        return AffineTransform(R, np.zeros(3))
    elif isinstance(op, LoadSITKTxOpModel):
        from aind_mri_utils.file_io.simpleitk import load_sitk_transform

        R, t, _ = load_sitk_transform(op.path)
        return AffineTransform(R, t, op.inverted)
    else:
        raise TypeError(f"Unsupported op {type(op)}")


def compile_recipe_to_chain(recipe: TransformRecipeModel) -> TransformChain:
    """
    Compile a recipe to a TransformChain. We keep individual ops as separate
    AffineTransforms (nice for debugging) but you could also pre-compose.
    """
    affines = [_op_to_affine(op) for op in recipe.sequence if op]
    return TransformChain.new(affines)


# --- target builder (returns spec + resolved points array) -------------------

# --- target builder (returns spec + resolved points array) -------------------
CompiledTransforms = dict[str, AffineTransform]


def compile_all_transforms(
    transforms: dict[str, TransformRecipeModel],
) -> CompiledTransforms:
    compiled: CompiledTransforms = {}
    for key, recipe in transforms.items():
        chain = compile_recipe_to_chain(recipe)
        R, t = chain.composed_transform
        compiled[key] = AffineTransform(R, t)
    return compiled


def resolve_transform_key_cached(
    key: Optional[str], cache: CompiledTransforms
) -> Optional[AffineTransform]:
    if not key:
        return None
    if key not in cache:
        raise KeyError(
            f"Unknown transform_key '{key}' (not found in compiled transforms)"
        )
    return cache[key]


def resolve_transform_ref_cached(
    ref: Optional[TransformRefModel], cache: CompiledTransforms
) -> Optional[AffineTransform]:
    if ref is None:
        return None
    if ref.key:
        return resolve_transform_key_cached(ref.key, cache)
    # Inline recipe: compile on the fly (not in cache by design)
    return compile_recipe_to_chain(ref.inline).composed_transform  # type: ignore[arg-type]


GeometryOut = Union[
    trimesh.Trimesh,  # surface mesh
    FloatNx3,  # (N,3) points
    dict[str, Float3],  # (name, point) pairs
]

# ---- Registry core --------------------------------------------------------
_GEOMETRY_LOADER_REGISTRY: dict[str, Callable[..., GeometryOut]] = {}


def register_loader_fn(fn: Callable[..., GeometryOut], name: Optional[str] = None):
    key = name or fn.__name__
    if key in _GEOMETRY_LOADER_REGISTRY:
        raise KeyError(f"Loader '{key}' already registered")
    _GEOMETRY_LOADER_REGISTRY[key] = fn
    return fn


def register_loader(arg: str | Callable[..., GeometryOut] | None = None):
    """Decorator to register a loader function by name."""

    def _wrap(fn: Callable[..., GeometryOut]):
        name = arg.__name__ if callable(arg) else arg
        return register_loader_fn(fn, name)

    if callable(arg):
        return _wrap(arg)
    else:
        return _wrap


def load_geometry(src: Union[str, Path], loader: str, **kwargs) -> GeometryOut:
    """Dispatch to a named loader. kwargs are passed to the loader."""
    fn = _GEOMETRY_LOADER_REGISTRY.get(loader)
    if fn is None:
        raise KeyError(
            (
                f"Unknown loader '{loader}'. Known: "
                f"{', '.join(sorted(_GEOMETRY_LOADER_REGISTRY)) or '(none)'}"
            )
        )
    return fn(Path(src), **kwargs)


@register_loader
def trimesh_from_sitk_mask(mask: sitk.Image) -> trimesh.Trimesh:
    """Convert a SimpleITK mask image to a trimesh."""
    structure_mesh = mask_to_trimesh(mask)
    trimesh.repair.fix_normals(structure_mesh)
    trimesh.repair.fix_inversion(structure_mesh)
    return structure_mesh


@register_loader
def load_trimesh_lps(path: Path, src_coordinate_system: str = "ASR") -> trimesh.Trimesh:
    """Load a trimesh from a SimpleITK image file."""
    mesh = trimesh.load(str(path))
    vertices_lps = convert_coordinate_system(
        mesh.vertices, src_coordinate_system, "LPS"
    )
    mesh.vertices = vertices_lps
    return mesh


register_loader_fn(read_slicer_fcsv)
register_loader_fn(trimesh.load, "trimesh")

SourceGeo = Union[trimesh.Trimesh, NDArray[np.float64]]
ReduceOut = NDArray[np.float64]  # usually (3,) single point; could be (N,3)

_REDUCER_REGISTRY: dict[str, Callable[..., ReduceOut]] = {}


def register_reducer_fn(fn: Callable[..., ReduceOut], name: Optional[str] = None):
    key = name or fn.__name__
    if key in _REDUCER_REGISTRY:
        raise KeyError(f"Reducer '{key}' already registered")
    _REDUCER_REGISTRY[key] = fn
    return fn


def register_reducer(arg: str | Callable[..., ReduceOut] | None = None):
    def _wrap(fn: Callable[..., ReduceOut]):
        name = arg.__name__ if callable(arg) else arg
        return register_reducer_fn(fn, name)

    if callable(arg):
        return _wrap(arg)
    else:
        return _wrap


def reduce_target(source: SourceGeo, reducer: str, **kwargs) -> ReduceOut:
    fn = _REDUCER_REGISTRY.get(reducer)
    if fn is None:
        raise KeyError(
            f"Unknown reducer '{reducer}'. Known: {', '.join(sorted(_REDUCER_REGISTRY)) or '(none)'}"
        )
    return fn(source, **kwargs)


@register_reducer
def mesh_center_mass(source: SourceGeo, **_) -> ReduceOut:
    # Compute the center of mass for the given geometry.
    if isinstance(source, trimesh.Trimesh):
        return np.array(source.center_mass)
    raise TypeError(f"Unsupported source type: {type(source)}")


@register_reducer
def mesh_centroid(source: SourceGeo, **_) -> ReduceOut:
    # Compute the centroid for the given geometry.
    if isinstance(source, trimesh.Trimesh):
        return np.array(source.centroid)
    raise TypeError(f"Unsupported source type: {type(source)}")


@dataclass(frozen=True)
class CanonicalizationRuntime:
    source_space: SourceSpace = OrientationCode.LPS  # e.g. "LPS", "RAS", "ASR"
    scale_to_mm: float = 1.0  # e.g. 0.001 if µm → mm
    transform_file_to_canonical: Optional[AffineTransform] = None


@dataclass(frozen=True)
class ChemShiftContext:
    enabled: bool
    magnet_MHz: float
    default_ppm: float = 3.7
    apply_by_role: set[Role] = field(default_factory=set)
    # transforms to apply to geometry in image/LPS space
    image: Optional[sitk.Image] = None
    # lazy cache: ppm -> AffineTransform (observed → corrected)
    _cache: dict[float, AffineTransform] = field(
        default_factory=dict, repr=False, compare=False
    )

    def pt_transform_for_ppm(self, ppm: Optional[float] = None) -> "AffineTransform":
        """
        Return the transform that moves points from observed (chem-shifted)
        positions to corrected positions for the given ppm, in LPS mm.
        """
        if self.image:
            if ppm is None:
                ppm = self.default_ppm
            if ppm in self._cache:
                return self._cache[ppm]
            chem_shift_pt_R, chem_shift_pt_t = chemical_shift_transform(
                compute_chemical_shift(self.image, ppm=ppm)
            )
            tf = AffineTransform(chem_shift_pt_R, chem_shift_pt_t)
            self._cache[ppm] = tf
        else:
            tf = AffineTransform.identity()

        return tf

    @classmethod
    def from_config(cls, cfg: ConfigModel) -> ChemShiftContext:
        im = cfg.imaging
        if im is None:
            return ChemShiftContext(False, 0.0, 0.0)
        # Build correction using your existing aind_mri_utils helpers.
        # If your `compute_chemical_shift` accepts only ppm, scale ppm if you want
        # frequency-awareness; otherwise pass ppm through (common in practice).
        if im.image_path:
            brain_image = sitk.ReadImage(str(im.image_path))
        else:
            brain_image = None
        return ChemShiftContext(
            enabled=True,
            magnet_MHz=im.magnet_frequency_MHz,
            default_ppm=im.chem_shift_ppm_default,
            apply_by_role=set(im.chem_shift_apply_by_role),
            image=brain_image,
        )


def _should_apply_chem(asset_model: BaseSpecModel, chem: ChemShiftContext) -> bool:
    if not chem.enabled:
        return False
    mode = asset_model.chem_shift_policy  # "on"|"off"|"auto"
    if mode == "on":
        return True
    if mode == "off":
        return False
    # "auto": follow role defaults
    return asset_model.role in chem.apply_by_role


def _load_calibration_bank(
    cal_file: CalibrationSourceModel, reticles: dict[str, CalibrationReticleModel]
) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load a calibration file that contains multiple probe entries.
    Return a dict mapping probe_code (string) -> (R,t).
    """
    # --- EXAMPLE STUBS (replace with real parsing) -------------------------
    if cal_file.directory:
        if cal_file.reticle is None:
            raise ValueError("Reticle model is required for directory calibration")
        reticle = reticles.get(cal_file.reticle)
        offset = np.array(reticle.offset_RAS, dtype=float)
        rotation = reticle.rotation_z
        cal_by_probe = fit_rotation_params_from_parallax(
            cal_file.directory, offset, rotation
        )[0]
    else:
        cal_by_probe = fit_rotation_params_from_manual_calibration(cal_file.file)[0]
    return {str(k): v for k, v in cal_by_probe.items()}


def _get_calibration_rt(
    calibrations: CalibrationsModel,
    reticles: dict[str, CalibrationReticleModel] = {},
) -> dict[str, "AffineTransform"]:
    """
    For each domain probe name, resolve (cal_id → path) then (probe_code → R,t).
    Cache each file load so it’s read once.
    """
    cal_files = calibrations.files
    probe_to_ref = calibrations.probe_to_ref
    cache: dict[str, dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    out: dict[str, AffineTransform] = {}

    for probe_name, ref in probe_to_ref.items():
        # load or reuse the bank
        if ref.cal_id not in cache:
            cal_file = cal_files[ref.cal_id]
            bank = _load_calibration_bank(cal_file, reticles)
            cache[ref.cal_id] = bank
        else:
            bank = cache[ref.cal_id]

        code = str(ref.probe_code)
        if code not in bank:
            # Clear error message showing available keys
            avail = ", ".join(sorted(bank.keys())[:8])
            raise KeyError(
                f"Calibration probe_code '{code}' not found in cal_id '{ref.cal_id}'. "
                f"Examples available: {avail}{' …' if len(bank) > 8 else ''}"
            )

        R, t = bank[code]
        out[probe_name] = AffineTransform(
            rotation=np.asarray(R, float), translation=np.asarray(t, float)
        )

    return out


def _compile_collision_labels(labels_in_use: Iterable[str]) -> dict[str, int]:
    """
    Assign each collision label a bit. Bit 0 is reserved for 'NONE' (unused).
    """
    labels = [l for l in dict.fromkeys(labels_in_use) if l]  # unique, drop falsy
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
    )


def resolve_material_for_spec(
    spec_like,  # AssetSpecModel or TargetSpecModel (post-template-merge)
    registry: dict[str, MaterialModel],  # config.materials
) -> "Material":
    base = registry.get(spec_like.material_ref) if spec_like.material_ref else None
    merged = merge_material_cfg(base, spec_like.material)
    mm = merged or MaterialModel()  # fallback default
    return _material_from_model(mm)


def _apply_canonicalization_mesh(
    mesh: trimesh.Trimesh,
    source_space: str,
    scale_to_mm: float,
    transform: Optional[AffineTransform] = None,
) -> trimesh.Trimesh:
    """
    Make a shallow copy of mesh in LPS mm.
    - Apply unit scaling.
    - Convert coordinate system to LPS.
    """
    if source_space == "FILE_NATIVE":
        if transform:
            R, t = transform.rotate_translate
            if np.linalg.det(R) < 0:
                new_faces = mesh.faces[:, ::-1].copy()  # flip faces if R is inverted
            else:
                new_faces = mesh.faces.copy()
            new_vertices = apply_rotate_translate(mesh.vertices, R, t)
            m = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
        else:
            raise ValueError("Transform is required for FILE_NATIVE source space")
    else:
        vertices = mesh.vertices.copy()
        if scale_to_mm != 1.0:
            vertices *= float(scale_to_mm)
        if source_space != "LPS":
            # convert_coordinate_system expects (N,3) array
            vertices = convert_coordinate_system(vertices, source_space, "LPS")
        m = trimesh.Trimesh(vertices=vertices, faces=mesh.faces.copy(), process=False)
    return m


def _apply_canonicalization_points(
    pts: np.ndarray,
    source_space: str,
    scale_to_mm: float,
    transform: Optional[AffineTransform] = None,
) -> np.ndarray:
    if source_space == "FILE_NATIVE":
        if transform:
            R, t = transform.rotate_translate
            p = apply_rotate_translate(pts, R, t)
        else:
            raise ValueError("Transform is required for FILE_NATIVE source space")
    else:
        p = np.asarray(pts, dtype=np.float64)
        if scale_to_mm != 1.0:
            p *= float(scale_to_mm)
        if source_space != "LPS":
            p = convert_coordinate_system(p, source_space, "LPS")
    return p


def _transform_chain_from_ref(transforms_model, key: str | None) -> TransformChain:
    """
    Resolve a ConfigModel.transforms entry into a TransformChain.
    Falls back to identity when key is None.
    """
    if not key:
        return TransformChain.new([AffineTransform.identity()])

    ref = transforms_model.get(key)
    if ref is None:
        raise KeyError(f"Unknown transform_key '{key}'")

    if ref.kind == "identity":
        return TransformChain.new([AffineTransform.identity()])

    if ref.kind == "sitk_file":
        return TransformChain.new(
            [AffineTransform.from_sitk_path(Path(ref.path), inverted=ref.invert)]
        )

    raise ValueError(f"Unsupported transform kind: {ref.kind!r}")


# -------------------------------------------------------------------------
# Output bundle
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
# Main builder
# -------------------------------------------------------------------------


def _capabilities_from_list(lst) -> Capability:
    val = 0
    for c in lst or []:
        # if Pydantic parsed as Capability already, bit-or directly
        if isinstance(c, int):
            val |= c
        else:
            # string fallback
            val |= getattr(Capability, str(c))
    return Capability(val)


def _collision_bits(
    policy: CollisionPolicyModel, label_to_bit: dict[str, int]
) -> tuple[int, int]:
    group_bits = label_to_bit.get(policy.group or "", 0)
    mask_bits = 0
    for lab in policy.mask:
        mask_bits |= label_to_bit.get(lab, 0)
    return group_bits, mask_bits


def _resolve_canonicalization_model(
    spec: BaseSpecModel,
    cfg: ConfigModel,
) -> Optional[CanonicalizationDefModel]:
    # pick base: from ref, or inline, or safe default
    if spec.canonicalization_ref:
        try:
            base = cfg.canonicalizations[spec.canonicalization_ref]
        except KeyError:
            raise KeyError(
                f"Unknown canonicalization_ref '{spec.canonicalization_ref}' for '{spec.key}'"
            )
    elif spec.canonicalization:
        base = spec.canonicalization
    else:
        return None

    # overlay overrides (only provided fields)
    if spec.canonicalization_override:
        ov = spec.canonicalization_override
        base = base.model_copy(
            update={k: v for k, v in ov.model_dump().items() if v is not None}
        )

    return base


def _canon_runtime_from_model(
    c: CanonicalizationDefModel, compiled_transforms: dict[str, AffineTransform] = {}
) -> CanonicalizationRuntime:
    # Keep transform_file_to_canonical as identity at runtime unless you
    # actually bake/load it.
    maybe_transform = resolve_transform_ref_cached(c.transform, compiled_transforms)
    return CanonicalizationRuntime(
        source_space=c.source_space,
        scale_to_mm=c.scale_to_mm,
        transform_file_to_canonical=maybe_transform,
    )


def _resolve_canon_model_to_runtime(
    spec: BaseSpecModel,
    cfg: ConfigModel,
    compiled_transforms: dict[str, AffineTransform],
) -> Optional[CanonicalizationRuntime]:
    cdef = _resolve_canonicalization_model(spec, cfg)
    if cdef:
        return _canon_runtime_from_model(cdef, compiled_transforms)
    return None


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


# --- asset builder -----------------------------------------------------------


def build_asset_spec(
    a: AssetSpecModel,
    label_to_bit: dict[str, int],
    chem: ChemShiftContext,
    canon: Optional[CanonicalizationRuntime],
    material_models: dict[str, MaterialModel] = {},
) -> AssetSpec:
    base_kwargs = _base_spec_kwargs_from_model(a, label_to_bit, material_models)

    mesh_tf: MeshTransformable | None = None
    pts_tf: PointsTransformable | None = None

    if a.src and a.loader:
        loader_kwargs = a.loader_kwargs or {}
        geo = load_geometry(Path(a.src), loader=a.loader, **loader_kwargs)
        if a.kind == Kind.MESH:
            if not isinstance(geo, trimesh.Trimesh):
                raise TypeError(f"Asset '{a.key}' loader returned points but kind=MESH")
            if canon:
                canon_mesh = _apply_canonicalization_mesh(
                    geo,
                    canon.source_space,
                    canon.scale_to_mm,
                    canon.transform_file_to_canonical,
                )
            else:
                canon_mesh = geo
            if _should_apply_chem(a, chem):
                chem_point_transform = chem.pt_transform_for_ppm(a.chem_shift_ppm)
                shifted_vertices = chem_point_transform.apply_to(canon_mesh.vertices)
                canon_mesh = trimesh.Trimesh(
                    vertices=shifted_vertices, faces=canon_mesh.faces, process=False
                )
            mesh_tf = MeshTransformable(canon_mesh)
        elif a.kind == Kind.POINTS:
            if isinstance(geo, trimesh.Trimesh):
                raise TypeError(f"Asset '{a.key}' loader returned mesh but kind=POINTS")
            if canon:
                pts = _apply_canonicalization_points(
                    geo,
                    canon.source_space,
                    canon.scale_to_mm,
                    canon.transform_file_to_canonical,
                )
            else:
                pts = geo
            if _should_apply_chem(a, chem):
                chem_point_transform = chem.pt_transform_for_ppm(a.chem_shift_ppm)
                pts = chem_point_transform.apply_to(pts)
            pts_tf = PointsTransformable(pts)
        elif a.kind == Kind.LINES:
            raise NotImplementedError("kind='lines' not implemented in loader")

    return AssetSpec(
        **base_kwargs,
        source_path=Path(a.src) if a.src else None,
        loader=a.loader,
        mesh=mesh_tf,
        points=pts_tf,
    )


def build_target_spec(
    t: TargetSpecModel,
    runtime_assets: dict[str, AssetSpec],
    label_to_bit: dict[str, int],
    chem: ChemShiftContext,
    canon: Optional[CanonicalizationRuntime] = None,
    reducer_registry: dict[str, Callable[..., np.ndarray]] = _REDUCER_REGISTRY,
    material_models: dict[str, MaterialModel] = {},
) -> tuple[TargetSpec, np.ndarray]:
    base_kwargs = _base_spec_kwargs_from_model(t, label_to_bit, material_models)

    # Targets must be non-collidable by default; enforce here (even if config forgot).
    base_kwargs["caps"] = Capability.RENDERABLE
    base_kwargs["collidable_group"] = 0
    base_kwargs["collidable_mask"] = 0

    # Resolve points (explicit file or derived by reducer)
    if t.src and t.loader:
        loader_kwargs = t.loader_kwargs or {}
        pts = load_geometry(Path(t.src), loader=t.loader, **loader_kwargs)
        if isinstance(pts, trimesh.Trimesh):
            raise TypeError(f"Target '{t.key}' loader returned mesh; expected points")
        if canon:
            pts = _apply_canonicalization_points(
                pts,
                canon.source_space,
                canon.scale_to_mm,
                canon.transform_file_to_canonical,
            )
        if _should_apply_chem(t, chem):
            chem_point_transform = chem.pt_transform_for_ppm(t.chem_shift_ppm)
            pts = chem_point_transform.apply_to(pts)
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
    else:
        # Derived: fetch source asset geometry first
        src_key = t.source_key or ""
        src_asset = runtime_assets.get(src_key)
        if src_asset is None:
            raise KeyError(
                f"Target '{t.key}' source_key '{src_key}' not found in loaded assets"
            )
        src_geo = (
            src_asset.mesh.raw
            if src_asset.mesh is not None
            else (src_asset.points.raw if src_asset.points is not None else None)
        )
        if src_geo is None:
            raise ValueError(
                f"Target '{t.key}': source asset '{src_key}' has no geometry loaded"
            )

        reducer_name = t.reducer or ""
        reducer_fn = reducer_registry.get(reducer_name)
        if reducer_fn is None:
            raise KeyError(
                f"Unknown target reducer '{reducer_name}' for target '{t.key}'"
            )

        pt = reducer_fn(
            src_geo, **(t.reducer_kwargs or {})
        )  # should return (3,) or (1,3)
        pts = np.asarray(pt, dtype=np.float64).reshape(1, 3)

    spec = TargetSpec(
        **base_kwargs,
        kind="points",  # targets are points in runtime
        role=Role.TARGET,
        source_path=Path(t.src) if t.src else None,
        loader=t.loader,
        source_key=t.source_key,
        reducer=t.reducer,
        reducer_kwargs=dict(t.reducer_kwargs),
        points=PointsTransformable(pts),
        approach_vector=np.array(t.approach_vector, float)
        if t.approach_vector
        else None,
        uncertainty_mm=t.uncertainty_mm,
    )
    return spec, pts


def build_runtime_from_config(cfg: ConfigModel) -> RuntimeBundle:
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
        runtime_assets[a.key] = build_asset_spec(
            a, label_to_bit, chem, maybe_cannon, cfg.materials
        )

    # 3) targets (specs + points index)
    runtime_targets: dict[str, TargetSpec] = {}
    target_index: dict[str, Float3] = {}
    for t in cfg.targets:
        maybe_cannon = _resolve_canon_model_to_runtime(t, cfg, compiled_transforms)
        tspec, pts = build_target_spec(
            t, runtime_assets, label_to_bit, chem, _REDUCER_REGISTRY, maybe_cannon
        )
        runtime_targets[tspec.key] = tspec
        target_index[tspec.key] = pts

    catalog = AssetCatalog(assets=runtime_assets, targets=runtime_targets)

    # 4) scene
    scene = Scene()
    for n in cfg.scene.nodes:
        asset_key = n.asset
        if asset_key not in runtime_assets:
            raise KeyError(
                f"Scene node '{n.id}' references unknown asset '{asset_key}'"
            )

        node_tf = _transform_chain_from_ref(cfg.transforms, n.transform)

        extras: dict[str, Any] = {}
        locked_axes: set[str] = set()
        if n.pose_source_probe:
            extras["pose_source_probe"] = n.pose_source_probe
            decl = cfg.plan.probes.get(n.pose_source_probe)
            if decl and decl.calibrated:
                locked_axes.update({"ap_tilt", "ml_tilt"})

        scene.upsert(
            NodeInstance(
                id=n.id,
                asset_key=asset_key,
                transform=node_tf,
                tags=set(n.tags),
                material_override=None,
                enabled=True,
                locked_axes=locked_axes,
                extras=extras,
            )
        )

    # 5) kinematics, calibrations, plans (build PlanningState)
    kinematics = Kinematics(arc_angles=dict(cfg.plan.arcs))
    calibrations = _get_calibration_rt(cfg.plan.calibrations, cfg.plan.reticles)
    probes: dict[str, ProbePlan] = {}
    for probe_name, probe_decl in cfg.plan.probes.items():
        probe_calibrated = probe_name in calibrations
        if probe_calibrated:
            ap, ml = find_probe_angle(calibrations[probe_name])
        else:
            ap = kinematics.get_arc(probe_decl.arc)
            ml = probe_decl.slider_ml
        if probe_decl.target.kind == "node":
            key = probe_decl.target.key
            transformed_points = resolve_base_geometry(catalog, scene, key)
            if not transformed_points:
                raise RuntimeError(
                    f"Probe '{probe_name}' references unknown target '{probe_decl.target.key}'"
                )
            transformed_points = transformed_points.raw
            target_index[key] = transformed_points
        probes[probe_name] = ProbePlan(
            probe_type=probe_decl.kind,
            arc_id=probe_decl.arc,
            bind_ap_to_arc=probe_calibrated,
            ap_local=ap,
            ml_local=ml,
            spin=probe_decl.spin,
            past_target_mm=probe_decl.past_target_mm,
            offsets_RA=probe_decl.offsets_RA,
            target_key=probe_decl.target.key,
            target_point_RAS=None,
            calibrated=probe_calibrated,
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
