"""Configuration/dsl for low point"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Annotated,
    Any,
    List,
    Literal,
    Optional,
    TypeAlias,
    Union,
)

from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    field_validator,
    model_validator,
)

from aind_low_point.common import Capability, Kind, Role
from aind_low_point.orientation_codes import OrientationCode

# Generate all 48 3-axis orientation codes (e.g., RAS, LPS, ASR, ...)

# Add FILE_NATIVE as a sentinel without mixing semantics
SourceSpace: TypeAlias = OrientationCode | Literal["FILE_NATIVE"]


class ImagingModel(BaseModel):
    magnet_frequency_MHz: float
    chem_shift_ppm_default: float = 3.7
    chem_shift_apply_by_role: List[Role] = Field(default_factory=lambda: [Role.ANATOMY])
    # optionally, where to read the reference image from if needed by your library
    image_path: Optional[FilePath] = None


ChemMode = Literal["on", "off", "auto"]


class MaterialModel(BaseModel):
    name: str = "default"
    color: str = Field("#C8C8C8", description="Hex #RRGGBB")
    opacity: float = 1.0
    wireframe: bool = False
    visible: bool = True

    @field_validator("opacity")
    @classmethod
    def _opacity_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("opacity must be in [0,1]")
        return v


class CanonicalizationDefModel(BaseModel):
    source_space: SourceSpace
    scale_to_mm: float = 1.0
    transform: Optional[TransformRefModel] = None  # name in your transforms registry
    version: str = "canon-v1"


class CanonicalizationOverrideModel(BaseModel):
    # all optional: only supplied fields override the referenced def
    source_space: Optional[SourceSpace] = None
    scale_to_mm: Optional[float] = None
    transform: Optional[TransformRefModel] = None
    version: Optional[str] = None


class ResourceModel(BaseModel):
    """
    A load-once file. The loader may return a structured container:
    - dict[str, np.ndarray] of named points
    - dict[str|int, trimesh.Trimesh] for labelmaps
    - GLTF scene graph keyed by node paths, etc.
    """

    key: str
    kind: Kind  # POINTS, MESH, LABELS, GLTF, etc. (choose the set you support)
    src: str
    loader: str  # e.g., "named_points_npz", "labelmap_to_meshes", "gltf"
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)
    canonicalization_ref: Optional[str] = None
    canonicalization: Optional[CanonicalizationDefModel] = None
    canonicalization_override: Optional[CanonicalizationOverrideModel] = None


class SelectorBase(BaseModel):
    kind: Literal["name", "index", "path", "label"]

    def select(self, payload: Any) -> Any:
        raise NotImplementedError


class NameSelector(SelectorBase):
    kind: Literal["name"]
    name: str

    def select(self, payload: Any) -> Any:
        return payload[self.name]


class IndexSelector(SelectorBase):
    kind: Literal["index"]
    index: int

    def select(self, payload: Any) -> Any:
        return payload[self.index]


class PathSelector(SelectorBase):
    kind: Literal["path"]  # e.g., HDF5 dataset path or GLTF node path
    path: str

    def select(self, payload: Any) -> Any:
        return payload[self.path]


class LabelSelector(SelectorBase):
    kind: Literal["label"]  # e.g., integer label id or string label name
    label: Union[int, str]

    def select(self, payload: Any) -> Any:
        return payload[self.label]


Selector = Annotated[
    Union[NameSelector, IndexSelector, PathSelector, LabelSelector],
    Field(discriminator="kind"),
]


def select_from_resource(payload: Any, selector: Selector) -> Any:
    return selector.select(payload)


class CollisionPolicyModel(BaseModel):
    """Label-based policy; compile to bitmasks in loader."""

    group: Optional[str] = Field(
        default=None, description="e.g., STATIC, FIXTURE, PROBE"
    )
    mask: List[str] = Field(
        default_factory=list, description="Labels it can collide with"
    )


class _TxOpBase(BaseModel):
    invert: bool = False


class TranslateTxOpModel(_TxOpBase):
    kind: Literal["translate_mm"] = "translate_mm"
    delta: List[float] = Field(..., min_length=3, max_length=3)


class RotateEulerTxOpModel(_TxOpBase):
    kind: Literal["rotate_euler_deg"] = "rotate_euler_deg"
    order: Literal[
        "XYZ",
        "XZY",
        "YXZ",
        "YZX",
        "ZXY",
        "ZYX",
        "xyz",
        "xzy",
        "yxz",
        "yzx",
        "zxy",
        "zyx",
    ] = "ZYX"
    angles_deg: List[float] = Field(..., min_length=3, max_length=3)


class LoadSITKTxOpModel(_TxOpBase):
    kind: Literal["sitk_file"] = "sitk_file"
    path: FilePath
    inverted: bool = False


TransformOp = Annotated[
    Union[TranslateTxOpModel, RotateEulerTxOpModel, LoadSITKTxOpModel],
    Field(discriminator="kind"),
]


class TransformRecipeModel(BaseModel):
    """Sequence of ops; accepts a single op or a list and normalizes to list."""

    sequence: List[TransformOp] = Field(default_factory=list)

    # Allow top-level single-op form:
    #   transforms:
    #     fit: { kind: sitk_file, path: ... }
    @model_validator(mode="before")
    @classmethod
    def _coerce_root_single_op(cls, data: Any):
        if isinstance(data, dict) and "sequence" not in data and "kind" in data:
            return {"sequence": [data]}
        return data

    # Allow 'sequence' itself to be a single op (dict or parsed model)
    @field_validator("sequence", mode="before")
    @classmethod
    def _coerce_sequence(cls, v: Any):
        if v is None:
            return []
        # if already a list, keep it
        if isinstance(v, list):
            return v
        # if a single op dict (has 'kind'), wrap it
        if isinstance(v, dict) and "kind" in v:
            return [v]
        # if a single parsed op model, wrap it
        if isinstance(v, _TxOpBase):
            return [v]
        raise TypeError("sequence must be a list[TransformOp] or a single TransformOp")


# Optional: key-or-inline reference, with the same single-op convenience
class TransformRefModel(BaseModel):
    key: Optional[str] = None
    inline: Optional[TransformRecipeModel] = None

    # Coerce various shorthand syntaxes into {key: ...} or {inline: {sequence: [...]}}
    @model_validator(mode="before")
    @classmethod
    def _coerce_root(cls, v: Any):
        # string → key
        if isinstance(v, str):
            return {"key": v}
        # list[op] → inline.sequence
        if isinstance(v, list):
            return {"inline": {"sequence": v}}
        # dict with a single op (has 'kind', no 'inline'/'key') → inline.sequence
        if isinstance(v, dict) and "kind" in v and "inline" not in v and "key" not in v:
            return {"inline": {"sequence": [v]}}
        # dict with a full recipe (has 'sequence' but no 'inline'/'key') → inline
        if (
            isinstance(v, dict)
            and "sequence" in v
            and "inline" not in v
            and "key" not in v
        ):
            return {"inline": v}
        return v

    # Also allow inline: {kind: ...} → inline: {sequence: [ ... ]}
    @field_validator("inline", mode="before")
    @classmethod
    def _coerce_inline(cls, v: Any):
        if v is None:
            return None
        if isinstance(v, dict) and "sequence" not in v and "kind" in v:
            return {"sequence": [v]}
        return v

    @model_validator(mode="after")
    def _xor(self):
        if bool(self.key) == bool(self.inline):
            raise ValueError("TransformRefModel: provide exactly one of {key | inline}")
        return self


# -----------------------------------------------------------------------------
# Catalog specs (WHAT an asset/target is; not where placed)
# -----------------------------------------------------------------------------


class BaseSpecModel(BaseModel):
    key: str
    kind: Kind
    role: Role = Role.GEOMETRY
    default_material: MaterialModel = Field(default_factory=MaterialModel)  # type: ignore[arg-type]
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

    canonicalization_ref: Optional[str] = None
    canonicalization: Optional[CanonicalizationDefModel] = (
        None  # inline (legacy/one-off)
    )
    canonicalization_override: Optional[CanonicalizationOverrideModel] = None

    # capabilities are parsed from strings like ["RENDERABLE", "COLLIDABLE"]
    caps: List[Capability] = Field(default_factory=lambda: [Capability.RENDERABLE])
    collision: CollisionPolicyModel = Field(default_factory=CollisionPolicyModel)

    # UI/layout hints
    pivot_LPS: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    bbox_hint: Optional[List[List[float]]] = Field(default=None)

    chem_shift_policy: ChemMode = "auto"
    chem_shift_ppm: Optional[float] = None

    @model_validator(mode="after")
    def _check_canon_choice(self):
        # allow: (ref) or (inline); not both
        if self.canonicalization_ref and self.canonicalization:
            raise ValueError(
                "Provide either canonicalization_ref or canonicalization, not both."
            )
        return self

    @field_validator("bbox_hint")
    @classmethod
    def _bbox_shape(cls, v):
        if v is None:
            return v
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(row, list) and len(row) == 3 for row in v)
        ):
            raise ValueError("bbox_hint must be [[minx,miny,minz],[maxx,maxy,maxz]]")
        return v


class AssetSpecModel(BaseSpecModel):
    """Geometry/points/lines that can be loaded by a named loader."""

    src: Optional[Path] = None
    loader: Optional[str] = None
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)

    from_resource: Optional[str] = None
    selector: Optional[Selector] = None

    @model_validator(mode="after")
    def _check_src_loader(self):
        if (self.src is None) ^ (self.loader is None):
            raise ValueError(
                "Asset must provide both 'src' and 'loader', or neither (if injected elsewhere)."
            )
        if (self.src and self.loader) and (self.from_resource or self.selector):
            raise ValueError(
                "Choose either (src+loader) or (from_resource+selector), not both."
            )
        if (self.from_resource is None) ^ (self.selector is None):
            # only one given
            raise ValueError(
                "When using from_resource, you must also provide a selector."
            )
        return self


class TargetSpecModel(BaseSpecModel):
    """Targets are points; explicit (src+loader) or derived (source_key+reducer)."""

    kind: Kind = Kind.POINTS
    role: Role = Role.TARGET

    # Explicit points (file)
    src: Optional[Path] = None
    loader: Optional[str] = None  # e.g., "numpy_points"
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)

    # Or derived from an existing asset in catalog
    source_key: Optional[str] = None
    reducer: Optional[str] = None
    reducer_kwargs: dict[str, Any] = Field(default_factory=dict)

    # Or resource
    from_resource: Optional[str] = None
    selector: Optional[Selector] = None
    post_reducer: Optional[str] = (
        None  # optional final reduction (e.g., COM of a selected mesh)
    )
    post_reducer_kwargs: dict[str, Any] = Field(default_factory=dict)

    approach_vector: Optional[List[float]] = Field(
        default=None, min_length=3, max_length=3
    )
    uncertainty_mm: Optional[float] = None

    @model_validator(mode="after")
    def _exactly_one_source(self):
        explicit = self.src is not None and self.loader is not None
        derived = self.source_key is not None and self.reducer is not None
        from_res = (self.from_resource is not None) and (self.selector is not None)
        paths = sum([explicit, derived, from_res])
        if paths != 1:
            raise ValueError(
                f"{self.key}: provide exactly one of (src+loader) | (source_key+reducer) | (from_resource+selector)"
            )
        return self

    @model_validator(mode="after")
    def _noncollidable_default(self):
        if Capability.COLLIDABLE in self.caps:
            raise ValueError(
                f"{self.key}: targets should not be collidable by default."
            )
        return self


# -----------------------------------------------------------------------------
# Scene (WHERE: instances and bindings)
# -----------------------------------------------------------------------------


class SceneNodeModel(BaseModel):
    id: str
    asset: str = Field(description="Key of an AssetSpec in catalog")
    tags: List[str] = Field(default_factory=list)

    # Reference a named transform (from ConfigModel.transforms) or leave None for identity
    transform: Optional[TransformRefModel] = None

    # Optional domain binding for pose (use for probes): ties node to domain.probes[name]
    pose_source_probe: Optional[str] = Field(
        default=None,
        description="If set, renderer should take pose from domain.probes[pose_source_probe].",
    )


class SceneModel(BaseModel):
    nodes: List[SceneNodeModel] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Domain (mechanics: arcs, probes, calibrations, target declarations)
# -----------------------------------------------------------------------------


class ProbeDeclModel(BaseModel):
    kind: str
    arc: str
    slider_ml: float = 0.0
    spin: float = 0.0

    target: str = Field(description="Key of a target (TargetSpecModel.key)")
    past_target_mm: float = 0.0
    offsets_RA: List[float] = Field(
        default_factory=lambda: [0.0, 0.0], min_length=2, max_length=2
    )

    calibrated: bool = False  # initial lock state; actual calibration affine comes from 'calibrations' map


class CalibrationRefModel(BaseModel):
    """Reference a specific calibration entry inside a calibration file."""

    cal_id: str  # key into CalibrationsModel.files
    probe_code: str  # 5-digit code in the file (keep as str; accept ints)

    # allow shorthand "cal_id:probe_code"
    @classmethod
    def from_string(cls, s: str) -> "CalibrationRefModel":
        if ":" not in s:
            raise ValueError("Expected '<cal_id>:<probe_code>'")
        cal_id, probe_code = s.split(":", 1)
        return cls(cal_id=cal_id.strip(), probe_code=str(probe_code).strip())


class CalibrationReticleModel(BaseModel):
    """Model for calibration reticle used in calibrations"""

    offset_RAS: List[float] = Field(default_factory=list, min_length=3, max_length=3)
    rotation_z: float = 0.0


class CalibrationSourceModel(BaseModel):
    """
    One calibration 'bank' source:
      - EITHER a single file (e.g., .xlsx). In this case NO reticle is allowed.
      - OR a directory for parallax. In this case a reticle IS REQUIRED.
    """

    file: Optional[FilePath] = Field(
        default=None, description="Path to a single calibration file (e.g., .xlsx)"
    )
    directory: Optional[DirectoryPath] = Field(
        default=None, description="Path to a parallax calibration directory"
    )
    reticle: Optional[str] = Field(
        default=None, description="Name of reticle (required when 'directory' is set)"
    )

    @model_validator(mode="after")
    def _xor_and_require_reticle(self):
        has_file = self.file is not None
        has_dir = self.directory is not None
        if has_file == has_dir:
            # both set or both None → invalid
            raise ValueError(
                "Specify exactly one of 'file' or 'directory' in a calibration source"
            )

        if has_file and self.reticle is not None:
            # forbid reticle with file
            raise ValueError("'reticle' must not be provided when 'file' is used")

        if has_dir and not self.reticle:
            # require reticle with directory
            raise ValueError("'reticle' is required when 'directory' is used")

        return self


class CalibrationsModel(BaseModel):
    files: dict[str, CalibrationSourceModel] = Field(default_factory=dict)
    # domain_probe_name → either {"cal_id": "...", "probe_code": "..."} OR "cal_id:probe_code"
    probe_to_ref: dict[str, Union[CalibrationRefModel, str]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _normalize_refs(self):
        # convert any string refs to CalibrationRefModel
        normalized: dict[str, CalibrationRefModel] = {}
        for probe_name, ref in self.probe_to_ref.items():
            if isinstance(ref, str):
                normalized[probe_name] = CalibrationRefModel.from_string(ref)
            else:
                normalized[probe_name] = ref
        object.__setattr__(self, "probe_to_ref", normalized)
        return self


class PlanningModel(BaseModel):
    arcs: dict[str, float] = Field(
        default_factory=dict, description="arc_id → AP angle (deg)"
    )
    probes: dict[str, ProbeDeclModel] = Field(
        default_factory=dict, description="probe_name → probe declaration"
    )
    reticles: dict[str, CalibrationReticleModel] = Field(default_factory=dict)
    calibrations: CalibrationsModel = Field(default_factory=CalibrationsModel)
    # Targets can live in the catalog (TargetSpecModel), or you can also allow simple inline targets here if desired.


# -----------------------------------------------------------------------------
# Transforms, Paths, Options
# -----------------------------------------------------------------------------


class PathsModel(BaseModel):
    """Freeform helper; keep loose so Hydra/OmegaConf interpolation is easy."""

    model_config = {"extra": "allow"}

    def __init__(self, **data):
        super().__init__(**data)


class OptionsModel(BaseModel):
    color_map: str = "rainbow"
    remove_last_color: bool = True


# -----------------------------------------------------------------------------
# Root config (everything together) + cross-reference validation
# -----------------------------------------------------------------------------


class ConfigModel(BaseModel):
    version: int = 1

    paths: PathsModel = Field(default_factory=PathsModel)
    imaging: Optional[ImagingModel] = None
    # Catalog
    resources: List[ResourceModel] = Field(default_factory=list)
    assets: List[AssetSpecModel] = Field(default_factory=list)
    targets: List[TargetSpecModel] = Field(default_factory=list)

    # Scene
    scene: SceneModel = Field(default_factory=SceneModel)

    # Domain
    plan: PlanningModel = Field(default_factory=PlanningModel)

    # Named transforms & misc
    transforms: dict[str, TransformRecipeModel] = Field(default_factory=dict)
    canonicalizations: dict[str, CanonicalizationDefModel] = Field(default_factory=dict)
    options: OptionsModel = Field(default_factory=OptionsModel)

    # ---------- Cross-file integrity checks ----------
    @model_validator(mode="after")
    def _xref(self):
        errors: List[str] = []

        # ---------- sets for quick membership ----------
        asset_keys = {a.key for a in self.assets}
        target_keys = {t.key for t in self.targets}
        transform_keys = set(self.transforms.keys())
        arc_ids = set(self.plan.arcs.keys())
        probe_names = set(self.plan.probes.keys())
        reticle_names = set(self.plan.reticles.keys())
        cal_files = self.plan.calibrations.files  # dict[id -> CalFileDecl]

        # ---------- helpers ----------
        def err(msg: str) -> None:
            errors.append(msg)

        def _where_key(obj) -> str:
            return getattr(obj, "key", "?")

        def _check_transform_ref(
            ref: Optional["TransformRefModel"], where: str
        ) -> None:
            # Only key references need validation; inline is self-contained.
            if ref and ref.key and ref.key not in transform_keys:
                err(f"{where}: transform key '{ref.key}' not found in transforms")

        def _check_canon_def(
            cdef: Optional["CanonicalizationDefModel"], where: str
        ) -> None:
            if not cdef:
                return
            _check_transform_ref(cdef.transform, f"{where}.transform")

        def _check_canon_fields(spec, where_prefix: str) -> None:
            # canonicalization_ref must exist; if it does, validate its transform ref
            cref = getattr(spec, "canonicalization_ref", None)
            if cref:
                cdef = self.canonicalizations.get(cref)
                if cdef is None:
                    err(
                        f"{where_prefix} '{_where_key(spec)}': canonicalization_ref '{cref}' not found"
                    )
                else:
                    _check_canon_def(cdef, f"canonicalizations['{cref}']")

            # Inline and override canonicalizations may each carry a transform ref
            _check_canon_def(
                getattr(spec, "canonicalization", None),
                f"{where_prefix} '{_where_key(spec)}'.canonicalization",
            )
            _check_canon_def(
                getattr(spec, "canonicalization_override", None),
                f"{where_prefix} '{_where_key(spec)}'.canonicalization_override",
            )

        # ---------- scene checks ----------
        for n in self.scene.nodes:
            if n.asset not in asset_keys:
                err(f"scene.nodes['{n.id}']: asset '{n.asset}' not found in assets")
            _check_transform_ref(
                getattr(n, "transform", None), f"scene.nodes['{n.id}'].transform"
            )
            if (
                getattr(n, "pose_source_probe", None)
                and n.pose_source_probe not in probe_names
            ):
                err(
                    f"scene.nodes['{n.id}'].pose_source_probe '{n.pose_source_probe}' not in plan.probes"
                )

        # ---------- targets ----------
        for t in self.targets:
            # Derived targets must reference existing assets
            if getattr(t, "source_key", None) and t.source_key not in asset_keys:
                err(
                    f"target '{t.key}': source_key '{t.source_key}' not found in assets"
                )
            _check_canon_fields(t, "target")

        # ---------- assets / resources ----------
        for a in self.assets:
            _check_canon_fields(a, "asset")
        for r in self.resources:
            _check_canon_fields(r, "resource")

        # ---------- plan (arcs, probes, calibrations) ----------
        for pname, p in self.plan.probes.items():
            if p.arc not in arc_ids:
                err(f"plan.probes['{pname}']: arc '{p.arc}' not found in plan.arcs")
            if p.target not in target_keys:
                err(f"plan.probes['{pname}']: target '{p.target}' not found in targets")

        # each calibration file must reference an existing reticle
        for cal_id, cal in cal_files.items():
            if cal.reticle not in reticle_names:
                err(
                    f"plan.calibrations.files['{cal_id}']: reticle '{cal.reticle}' not defined in plan.reticles"
                )

        # probe_to_ref must point to a valid probe and cal file id
        for probe_name, ref in self.plan.calibrations.probe_to_ref.items():
            if probe_name not in probe_names:
                err(
                    f"plan.calibrations.probe_to_ref: probe '{probe_name}' not in plan.probes"
                )
            if ref.cal_id not in cal_files:
                err(
                    f"plan.calibrations.probe_to_ref['{probe_name}']: cal_id '{ref.cal_id}' not in plan.calibrations.files"
                )

        # ---------- canonicalization defs themselves ----------
        for cname, cdef in self.canonicalizations.items():
            _check_transform_ref(
                cdef.transform, f"canonicalizations['{cname}'].transform"
            )
            # Require a transform (key OR inline) when the source space is FILE_NATIVE
            if cdef.source_space == "FILE_NATIVE" and cdef.transform is None:
                errors.append(
                    f"canonicalizations['{cname}']: source_space=FILE_NATIVE requires a transform (key or inline)"
                )

        # ---------- final ----------
        if errors:
            raise ValueError(
                "Config cross-reference errors:\n  - " + "\n  - ".join(errors)
            )
        return self
