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
    CatalogTargetRefModel,
    CollisionPolicyModel,
    ConfigModel,
    InlineTargetRefModel,
    LoadSITKTxOpModel,
    MaterialModel,
    NodeTargetRefModel,
    PlanningModel,
    ProbeDeclModel,
    ResourceModel,
    RotateEulerTxOpModel,
    SourceSpace,
    TargetSpecModel,
    TransformRecipeModel,
    TransformRefModel,
    TranslateTxOpModel,
    _merge_dict_shallow,
    _TxOpBase,
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
    return fn(str(src), **kwargs)


def trimesh_from_sitk_mask(mask: sitk.Image) -> trimesh.Trimesh:
    """Convert a SimpleITK mask image to a trimesh."""
    structure_mesh = mask_to_trimesh(mask)
    trimesh.repair.fix_normals(structure_mesh)
    trimesh.repair.fix_inversion(structure_mesh)
    return structure_mesh


@register_loader
def sitk_volume(path: str) -> trimesh.Trimesh:
    """Read a SimpleITK volume file (.nrrd, .nii, .nii.gz) and mesh it."""
    mask = sitk.ReadImage(path)
    return trimesh_from_sitk_mask(mask)


@register_loader
def load_trimesh_lps(path: str, src_coordinate_system: str = "ASR") -> trimesh.Trimesh:
    """Load a trimesh from a file and convert to LPS."""
    mesh = trimesh.load(path)
    vertices_lps = convert_coordinate_system(
        mesh.vertices, src_coordinate_system, "LPS"
    )
    mesh.vertices = vertices_lps
    return mesh


register_loader_fn(read_slicer_fcsv)


@register_loader("trimesh")
def _load_trimesh(path: str) -> trimesh.Trimesh:
    return trimesh.load(path, force="mesh")


@register_loader
def csv_points(
    path: str, max_points: int | None = None
) -> NDArray[np.float64]:
    """Load an (N,3) point cloud from a CSV with x, y, z columns.

    Parameters
    ----------
    max_points
        If set, randomly subsample to this many points.
    """
    import pandas as pd

    df = pd.read_csv(path, index_col=0)
    pts = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if max_points is not None and len(pts) > max_points:
        idx = np.random.default_rng().choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    return pts

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
        known = ", ".join(sorted(_REDUCER_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown reducer '{reducer}'. Known: {known}")
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


@register_reducer
def hemisphere_center_mass(
    source: SourceGeo,
    *,
    hemisphere: str = "left",
    plane: float = 0.0,
    **_,
) -> ReduceOut:
    """Volumetric centre of mass of one LPS hemisphere of the source mesh.

    In LPS the L axis points to the patient's left, so vertices with
    ``x > plane`` are in the LEFT hemisphere; ``x < plane`` is the right.
    The mesh is sliced at ``x = plane`` (capped) and the resulting
    half-mesh's ``center_mass`` returned. Falls back to a vertex-mean of
    the requested side if slicing fails or yields an empty mesh.

    Parameters
    ----------
    hemisphere
        ``"left"`` / ``"l"`` or ``"right"`` / ``"r"``.
    plane
        Sagittal split plane in LPS mm (default 0 = midline).
    """
    hemi = hemisphere.lower()
    if hemi in {"left", "l"}:
        sign = 1.0
    elif hemi in {"right", "r"}:
        sign = -1.0
    else:
        raise ValueError(
            f"hemisphere must be 'left' or 'right', got {hemisphere!r}"
        )

    if isinstance(source, trimesh.Trimesh):
        half = None
        try:
            half = source.slice_plane(
                plane_origin=[float(plane), 0.0, 0.0],
                plane_normal=[sign, 0.0, 0.0],
                cap=True,
            )
        except Exception:
            half = None
        if half is not None and len(half.vertices) > 0 and not half.is_empty:
            return np.asarray(half.center_mass, dtype=np.float64)
        verts = np.asarray(source.vertices, dtype=np.float64)
    else:
        verts = np.asarray(source, dtype=np.float64)

    mask = (verts[:, 0] - plane) * sign > 0
    if not mask.any():
        return verts.mean(axis=0)
    return verts[mask].mean(axis=0)


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


def _resolve_scene_node_transform(
    ref: Optional[TransformRefModel],
    compiled_transforms: CompiledTransforms,
) -> TransformChain:
    """Resolve a scene node's TransformRefModel into a TransformChain."""
    if ref is None:
        return TransformChain.new([AffineTransform.identity()])
    if ref.key:
        affine = resolve_transform_key_cached(ref.key, compiled_transforms)
        if affine is None:
            return TransformChain.new([AffineTransform.identity()])
        return TransformChain.new([affine])
    if ref.inline:
        return compile_recipe_to_chain(ref.inline)
    return TransformChain.new([AffineTransform.identity()])


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


def _resolve_canonicalization_model(
    spec: BaseSpecModel | ResourceModel,
    cfg: ConfigModel,
) -> Optional[CanonicalizationDefModel]:
    # pick base: from ref, or inline, or safe default
    if spec.canonicalization_ref:
        try:
            base = cfg.canonicalizations[spec.canonicalization_ref]
        except KeyError:
            raise KeyError(
                f"Unknown canonicalization_ref "
                f"'{spec.canonicalization_ref}' for '{spec.key}'"
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
    spec: BaseSpecModel | ResourceModel,
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
            geo = trimesh.Trimesh(
                vertices=shifted, faces=geo.faces, process=False
            )
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


# --- asset builder -----------------------------------------------------------


def build_asset_spec(
    a: AssetSpecModel,
    base_kwargs: dict[str, Any],
    chem: ChemShiftContext,
    canon: Optional[CanonicalizationRuntime],
) -> AssetSpec:
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
    # This is a placeholder implementation
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
    kinematics = Kinematics(arc_angles=dict(cfg.plan.arcs))
    calibrations = _get_calibration_rt(cfg.plan.calibrations, cfg.plan.reticles)
    probes: dict[str, ProbePlan] = {}
    for probe_name, probe_decl in cfg.plan.probes.items():
        probe_calibrated = probe_name in calibrations
        if probe_calibrated:
            ap, ml = find_probe_angle(calibrations[probe_name])
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


# -------------------------------------------------------------------------
# Reverse path: PlanningState → config models
# -------------------------------------------------------------------------


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
            calibrated=plan.calibrated,
            auto_scene=orig_decl.auto_scene if orig_decl else True,
            scene_tags=orig_decl.scene_tags if orig_decl else ["probe", "dynamic"],
        )

    return PlanningModel(
        arcs=dict(state.kinematics.arc_angles),
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
