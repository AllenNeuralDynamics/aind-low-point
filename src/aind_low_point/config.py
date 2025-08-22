"""Configuration/dsl for low point"""

from __future__ import annotations

from copy import deepcopy
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
class BaseTemplateModel(BaseModel):
    """Common defaults for both assets and targets."""

    name: str
    kind: Optional["Kind"] = None
    role: Optional["Role"] = None

    material_ref: Optional[str] = None
    material: Optional[MaterialModel] = None

    tags: List[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    canonicalization_ref: Optional[str] = None
    canonicalization: Optional["CanonicalizationDefModel"] = None
    canonicalization_override: Optional["CanonicalizationDefModel"] = None

    caps: Optional[List["Capability"]] = None
    collision: Optional["CollisionPolicyModel"] = None

    pivot_LPS: Optional[List[float]] = None
    bbox_hint: Optional[List[List[float]]] = None

    # Chem-shift hints (optional, ignored if not applicable)
    chem_shift_ppm: Optional[float] = None
    chem_shift_policy: ChemMode = "auto"


class AssetTemplateModel(BaseTemplateModel):
    """Defaults oriented to geometry assets."""

    src: Optional[Path] = None
    loader: Optional[str] = None
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)

    from_resource: Optional[str] = None
    selector: Optional[Selector] = None


class TargetTemplateModel(BaseTemplateModel):
    """Defaults oriented to targets."""

    # explicit points
    src: Optional[Path] = None
    loader: Optional[str] = None
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)
    # or derived
    source_key: Optional[str] = None
    reducer: Optional[str] = None
    reducer_kwargs: dict[str, Any] = Field(default_factory=dict)

    # or from resource
    from_resource: Optional[str] = None
    selector: Optional[Selector] = None

    post_reducer: Optional[str] = (
        None  # optional final reduction (e.g., COM of a selected mesh)
    )

    post_reducer_kwargs: dict[str, Any] = Field(default_factory=dict)
    approach_vector: Optional[List[float]] = None
    uncertainty_mm: Optional[float] = None


class BaseSpecModel(BaseModel):
    key: str
    kind: Kind
    role: Role = Role.GEOMETRY

    material_ref: Optional[str] = None
    material: Optional[MaterialModel] = None

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

    # NEW: list of template names to apply, left→right priority
    templates: List[str] = Field(default_factory=list)

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

    templates: List[str] = Field(default_factory=list)

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
    resources: List[ResourceModel] = Field(default_factory=list)
    materials: dict[str, MaterialModel] = Field(default_factory=dict)

    # Catalog
    asset_templates: dict[str, AssetTemplateModel] = Field(default_factory=dict)
    target_templates: dict[str, TargetTemplateModel] = Field(default_factory=dict)
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
    def _xref_and_expand_templates(self):
        # 1) Expand templates into concrete specs

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

        def _check_template_ref(spec, templates, where_prefix: str):
            trefs = getattr(spec, "templates", [])
            for tref in trefs:
                if tref not in templates:
                    err(
                        f"{where_prefix} '{_where_key(spec)}': template '{tref}' not found in templates"
                    )

        for a in self.assets:
            _check_template_ref(a, self.asset_templates, "asset")
        for t in self.targets:
            _check_template_ref(t, self.target_templates, "target")

        # Expand templates into concrete specs
        if self.asset_templates and self.assets:
            self.assets = [
                apply_asset_templates(a, self.asset_templates) for a in self.assets
            ]
        if self.target_templates and self.targets:
            self.targets = [
                apply_target_templates(t, self.target_templates) for t in self.targets
            ]

        def _check_material_ref(spec, where_prefix: str):
            mref = getattr(spec, "material_ref", None)
            if mref and mref not in self.materials:
                err(
                    f"{where_prefix} '{_where_key(spec)}': material_ref '{mref}' not found in materials"
                )

        for a in self.assets:
            _check_material_ref(a, "asset")
        for t in self.targets:
            _check_material_ref(t, "target")
        for tmpl in self.asset_templates.values():
            if tmpl.material_ref and tmpl.material_ref not in self.materials:
                err(
                    f"asset_templates['{tmpl.name}']: material_ref '{tmpl.material_ref}' not found"
                )
        for tmpl in self.target_templates.values():
            if tmpl.material_ref and tmpl.material_ref not in self.materials:
                err(
                    f"target_templates['{tmpl.name}']: material_ref '{tmpl.material_ref}' not found"
                )

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


def _union_list(a: Optional[List[Any]], b: Optional[List[Any]]) -> Optional[List[Any]]:
    if a is None and b is None:
        return None
    seen: Set[Any] = set()
    out: List[Any] = []
    for src in (a or []), (b or []):
        for x in src:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


def _merge_dict_shallow(
    a: Optional[dict[str, Any]], b: Optional[dict[str, Any]]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if a:
        out.update(a)
    if b:
        out.update(b)
    return out


def merge_material_cfg(
    base: Optional["MaterialModel"], over: Optional["MaterialModel"]
) -> Optional["MaterialModel"]:
    if base is None:
        return over
    if over is None:
        return base
    d = base.model_dump(exclude_none=True)
    d.update(over.model_dump(exclude_none=True))
    return type(base)(**d)


def _merge_model_generic(base, over):
    """Shallow overlay for pydantic models with simple fields."""
    if base is None:
        return over
    if over is None:
        return base
    d = base.model_dump(exclude_none=True)
    d.update(over.model_dump(exclude_none=True))
    # Prefer 'over' class if different; both should be compatible
    cls = type(over) if type(base) is not type(over) else type(base)
    return cls(**d)


def _merge_collision(
    base: Optional["CollisionPolicyModel"], over: Optional["CollisionPolicyModel"]
) -> Optional["CollisionPolicyModel"]:
    if base is None:
        return over
    if over is None:
        return base
    db = base.model_dump(exclude_none=True)
    do = over.model_dump(exclude_none=True)
    # union mask; overlay group if provided
    merged_mask = _union_list(db.get("mask"), do.get("mask")) or []
    group = do.get("group", db.get("group"))
    db.update(do)
    db["mask"] = merged_mask
    db["group"] = group
    return type(base)(**db)


# -----------------------------
# Asset template merge
# -----------------------------


def _merge_asset_source_fields(
    base: "AssetTemplateModel | AssetSpecModel",
    over: "AssetTemplateModel | AssetSpecModel",
) -> dict[str, Any]:
    """
    Choose exactly one source mode:
      - (src + loader [+ loader_kwargs])
      - (from_resource + selector)
    If overlay specifies any field of a mode, that mode wins; the other mode is cleared.
    """
    out: dict[str, Any] = {}

    base_file = (getattr(base, "src", None) is not None) or (
        getattr(base, "loader", None) is not None
    )
    base_res = (getattr(base, "from_resource", None) is not None) or (
        getattr(base, "selector", None) is not None
    )

    over_file = (getattr(over, "src", None) is not None) or (
        getattr(over, "loader", None) is not None
    )
    over_res = (getattr(over, "from_resource", None) is not None) or (
        getattr(over, "selector", None) is not None
    )

    if over_file and over_res:
        raise ValueError("Asset: overlay specifies both file and resource source modes")

    if over_file:
        # file mode wins; take overlay values if present, else fall back to base for the same mode
        out["src"] = getattr(over, "src", None) or getattr(base, "src", None)
        out["loader"] = getattr(over, "loader", None) or getattr(base, "loader", None)
        out["loader_kwargs"] = _merge_dict_shallow(
            getattr(base, "loader_kwargs", None), getattr(over, "loader_kwargs", None)
        )
        out["from_resource"] = None
        out["selector"] = None

    elif over_res:
        out["from_resource"] = getattr(over, "from_resource", None) or getattr(
            base, "from_resource", None
        )
        out["selector"] = getattr(over, "selector", None) or getattr(
            base, "selector", None
        )
        # clear file mode
        out["src"] = None
        out["loader"] = None
        out["loader_kwargs"] = {}
    else:
        # overlay did not change mode → keep base (as-is)
        out["src"] = getattr(base, "src", None)
        out["loader"] = getattr(base, "loader", None)
        out["loader_kwargs"] = getattr(base, "loader_kwargs", {}) or {}
        out["from_resource"] = getattr(base, "from_resource", None)
        out["selector"] = getattr(base, "selector", None)

    return out


def merge_template_into_asset(
    base: "AssetTemplateModel",
    over: "AssetTemplateModel | AssetSpecModel",
) -> "AssetTemplateModel":
    A = base.model_dump(exclude_none=True)
    B = over.model_dump(exclude_none=True)

    out = deepcopy(A)
    out.update(B)

    # unions
    out["tags"] = _union_list(A.get("tags"), B.get("tags")) or []
    out["caps"] = _union_list(A.get("caps"), B.get("caps")) or None

    # metadata shallow merge
    out["metadata"] = _merge_dict_shallow(A.get("metadata"), B.get("metadata"))

    # nested merges
    out["material"] = merge_material_cfg(base.material, getattr(over, "material", None))
    out["canonicalization"] = _merge_model_generic(
        base.canonicalization, getattr(over, "canonicalization", None)
    )
    out["canonicalization_override"] = _merge_model_generic(
        base.canonicalization_override, getattr(over, "canonicalization_override", None)
    )
    out["collision"] = _merge_collision(
        base.collision, getattr(over, "collision", None)
    )

    # refs (replace-on-write)
    out["material_ref"] = getattr(over, "material_ref", None) or getattr(
        base, "material_ref", None
    )

    # hints (replace-on-write)
    out["pivot_LPS"] = getattr(over, "pivot_LPS", None) or getattr(
        base, "pivot_LPS", None
    )
    out["bbox_hint"] = getattr(over, "bbox_hint", None) or getattr(
        base, "bbox_hint", None
    )

    # chem-shift hints (replace-on-write)
    out["chem_shift_ppm"] = (
        getattr(over, "chem_shift_ppm", None)
        if hasattr(over, "chem_shift_ppm")
        else getattr(base, "chem_shift_ppm", None)
    )
    out["chem_shift_policy"] = getattr(over, "chem_shift_policy", None) or getattr(
        base, "chem_shift_policy", None
    )

    # source modes
    out.update(_merge_asset_source_fields(base, over))

    return AssetTemplateModel(**out)


def apply_asset_templates(
    spec: "AssetSpecModel",
    registry: dict[str, "AssetTemplateModel"],
) -> "AssetSpecModel":
    if not spec.templates:
        return spec

    # fold templates left→right
    acc: Optional["AssetTemplateModel"] = None
    for name in spec.templates:
        t = registry.get(name)
        if t is None:
            raise ValueError(f"asset '{spec.key}' references unknown template '{name}'")
        acc = t if acc is None else merge_template_into_asset(acc, t)

    merged_tmpl = merge_template_into_asset(acc, spec)  # overlay spec onto templates
    # materialize back to AssetSpecModel; ensure we keep spec's identity fields
    payload = merged_tmpl.model_dump(exclude_none=True)
    payload.update(
        {
            "key": spec.key,
            "kind": spec.kind,
            "role": spec.role,
            "templates": [],  # clear to avoid re-applying in any subsequent validation
        }
    )
    return AssetSpecModel(**payload)


# -----------------------------
# Target template merge
# -----------------------------


def _detect_target_mode(obj) -> Optional[str]:
    explicit = (getattr(obj, "src", None) is not None) or (
        getattr(obj, "loader", None) is not None
    )
    derived = (getattr(obj, "source_key", None) is not None) or (
        getattr(obj, "reducer", None) is not None
    )
    res = (getattr(obj, "from_resource", None) is not None) or (
        getattr(obj, "selector", None) is not None
    )
    cnt = int(explicit) + int(derived) + int(res)
    if cnt == 0:
        return None
    if cnt == 1:
        return "explicit" if explicit else ("derived" if derived else "resource")
    return "conflict"


def _merge_target_source_fields(
    base: "TargetTemplateModel | TargetSpecModel",
    over: "TargetTemplateModel | TargetSpecModel",
) -> dict[str, Any]:
    """
    Exactly one of:
      - explicit: src + loader [+ loader_kwargs]
      - derived : source_key + reducer [+ reducer_kwargs]
      - resource: from_resource + selector [+ post_reducer]
    Overlay choosing any field of a mode selects that mode and clears others.
    """
    out: dict[str, Any] = {}

    mode_base = _detect_target_mode(base)
    mode_over = _detect_target_mode(over)
    if mode_over == "conflict":
        raise ValueError("Target: overlay specifies conflicting source modes")

    # if overlay picks a mode, use it; else keep base
    mode = mode_over or mode_base

    if mode == "explicit":
        out["src"] = getattr(over, "src", None) or getattr(base, "src", None)
        out["loader"] = getattr(over, "loader", None) or getattr(base, "loader", None)
        out["loader_kwargs"] = _merge_dict_shallow(
            getattr(base, "loader_kwargs", None), getattr(over, "loader_kwargs", None)
        )
        # clear others
        out.update(
            {
                "source_key": None,
                "reducer": None,
                "reducer_kwargs": {},
                "from_resource": None,
                "selector": None,
                "post_reducer": None,
                "post_reducer_kwargs": {},
            }
        )

    elif mode == "derived":
        out["source_key"] = getattr(over, "source_key", None) or getattr(
            base, "source_key", None
        )
        out["reducer"] = getattr(over, "reducer", None) or getattr(
            base, "reducer", None
        )
        out["reducer_kwargs"] = _merge_dict_shallow(
            getattr(base, "reducer_kwargs", None), getattr(over, "reducer_kwargs", None)
        )
        # clear others
        out.update(
            {
                "src": None,
                "loader": None,
                "loader_kwargs": {},
                "from_resource": None,
                "selector": None,
                "post_reducer": None,
                "post_reducer_kwargs": {},
            }
        )

    elif mode == "resource":
        out["from_resource"] = getattr(over, "from_resource", None) or getattr(
            base, "from_resource", None
        )
        out["selector"] = getattr(over, "selector", None) or getattr(
            base, "selector", None
        )
        out["post_reducer"] = getattr(over, "post_reducer", None) or getattr(
            base, "post_reducer", None
        )
        out["post_reducer_kwargs"] = _merge_dict_shallow(
            getattr(base, "post_reducer_kwargs", None),
            getattr(over, "post_reducer_kwargs", None),
        )
        # clear others
        out.update(
            {
                "src": None,
                "loader": None,
                "loader_kwargs": {},
                "source_key": None,
                "reducer": None,
                "reducer_kwargs": {},
            }
        )

    else:
        # neither base nor overlay specified a mode → keep as is (all None/empty)
        out.update(
            {
                "src": getattr(base, "src", None),
                "loader": getattr(base, "loader", None),
                "loader_kwargs": getattr(base, "loader_kwargs", {}) or {},
                "source_key": getattr(base, "source_key", None),
                "reducer": getattr(base, "reducer", None),
                "reducer_kwargs": getattr(base, "reducer_kwargs", {}) or {},
                "from_resource": getattr(base, "from_resource", None),
                "selector": getattr(base, "selector", None),
                "post_reducer": getattr(base, "post_reducer", None),
                "post_reducer_kwargs": getattr(base, "post_reducer_kwargs", {}) or {},
            }
        )

    return out


def merge_template_into_target(
    base: "TargetTemplateModel",
    over: "TargetTemplateModel | TargetSpecModel",
) -> "TargetTemplateModel":
    A = base.model_dump(exclude_none=True)
    B = over.model_dump(exclude_none=True)

    out = deepcopy(A)
    out.update(B)

    # unions
    out["tags"] = _union_list(A.get("tags"), B.get("tags")) or []
    out["caps"] = _union_list(A.get("caps"), B.get("caps")) or None

    # metadata shallow merge
    out["metadata"] = _merge_dict_shallow(A.get("metadata"), B.get("metadata"))

    # nested merges
    out["material"] = merge_material_cfg(base.material, getattr(over, "material", None))
    out["canonicalization"] = _merge_model_generic(
        base.canonicalization, getattr(over, "canonicalization", None)
    )
    out["canonicalization_override"] = _merge_model_generic(
        base.canonicalization_override, getattr(over, "canonicalization_override", None)
    )
    out["collision"] = _merge_collision(
        base.collision, getattr(over, "collision", None)
    )

    # refs (replace-on-write)
    out["material_ref"] = getattr(over, "material_ref", None) or getattr(
        base, "material_ref", None
    )

    # hints (replace-on-write)
    out["pivot_LPS"] = getattr(over, "pivot_LPS", None) or getattr(
        base, "pivot_LPS", None
    )
    out["bbox_hint"] = getattr(over, "bbox_hint", None) or getattr(
        base, "bbox_hint", None
    )
    out["approach_vector"] = getattr(over, "approach_vector", None) or getattr(
        base, "approach_vector", None
    )
    out["uncertainty_mm"] = getattr(over, "uncertainty_mm", None) or getattr(
        base, "uncertainty_mm", None
    )

    # chem-shift hints (targets rarely need it; still honor if present)
    out["chem_shift_ppm"] = (
        getattr(over, "chem_shift_ppm", None)
        if hasattr(over, "chem_shift_ppm")
        else getattr(base, "chem_shift_ppm", None)
    )
    out["chem_shift_policy"] = getattr(over, "chem_shift_policy", None) or getattr(
        base, "chem_shift_policy", None
    )

    # source modes
    out.update(_merge_target_source_fields(base, over))

    return TargetTemplateModel(**out)


def apply_target_templates(
    spec: "TargetSpecModel",
    registry: dict[str, "TargetTemplateModel"],
) -> "TargetSpecModel":
    if not spec.templates:
        return spec

    acc: Optional["TargetTemplateModel"] = None
    for name in spec.templates:
        t = registry.get(name)
        if t is None:
            raise ValueError(
                f"target '{spec.key}' references unknown template '{name}'"
            )
        acc = t if acc is None else merge_template_into_target(acc, t)

    merged_tmpl = merge_template_into_target(acc, spec)  # overlay spec last
    payload = merged_tmpl.model_dump(exclude_none=True)
    payload.update(
        {
            "key": spec.key,
            "kind": spec.kind,
            "role": spec.role,
            "templates": [],
        }
    )
    return TargetSpecModel(**payload)
