"""Configuration/dsl for low point"""

from __future__ import annotations

import fnmatch
from copy import deepcopy
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    TypeAlias,
    TypeVar,
    Union,
)

from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    PrivateAttr,
    field_validator,
    model_validator,
)

from aind_low_point.common import Capability, Kind, Role
from aind_low_point.orientation_codes import OrientationCode

# -----------------------------------------------------------------------------
# Extension-based inference for kind and loader
# -----------------------------------------------------------------------------

EXTENSION_DEFAULTS: dict[str, dict[str, str]] = {
    ".obj": {"kind": "mesh", "loader": "trimesh"},
    ".stl": {"kind": "mesh", "loader": "trimesh"},
    ".ply": {"kind": "mesh", "loader": "trimesh"},
    ".nrrd": {"kind": "mesh", "loader": "sitk_volume"},
    ".nii": {"kind": "mesh", "loader": "sitk_volume"},
    ".nii.gz": {"kind": "mesh", "loader": "sitk_volume"},
    ".npy": {"kind": "points", "loader": "numpy_points"},
}

# -----------------------------------------------------------------------------
# Role inference from key prefix
# -----------------------------------------------------------------------------

ROLE_PREFIX_DEFAULTS: list[tuple[str, Role]] = [
    ("structure:", Role.ANATOMY),
    ("brain", Role.ANATOMY),
    ("target:", Role.TARGET),
    ("landmark:", Role.LANDMARK),
]


def _infer_from_extension(spec: "AssetSpecModel | TargetSpecModel") -> None:
    """Mutate spec to fill in kind/loader from src extension if not set."""
    if spec.src is None:
        return
    src_str = str(spec.src)
    # Check longer extensions first (e.g., .nii.gz before .nii)
    for ext in sorted(EXTENSION_DEFAULTS.keys(), key=len, reverse=True):
        if src_str.endswith(ext):
            defaults = EXTENSION_DEFAULTS[ext]
            if spec.kind is None:
                object.__setattr__(spec, "kind", Kind(defaults["kind"]))
            if spec.loader is None:
                object.__setattr__(spec, "loader", defaults["loader"])
            break


def _infer_role_from_key(spec: "AssetSpecModel | TargetSpecModel") -> None:
    """Mutate spec to fill in role from key prefix if not set."""
    if spec.role is not None or spec.key is None:
        return
    for prefix, role in ROLE_PREFIX_DEFAULTS:
        if spec.key.startswith(prefix):
            object.__setattr__(spec, "role", role)
            return
    object.__setattr__(spec, "role", Role.GEOMETRY)  # default


def _find_matching_templates(key: str, templates: dict[str, Any]) -> list[str]:
    """Return template names that match the key (glob patterns).

    Exact matches have priority over glob patterns - if an exact match exists,
    only that template is returned. Otherwise, multiple glob matches apply
    in template dict order.
    """
    if key is None:
        return []

    matches: list[str] = []

    for tname in templates:
        if tname == key:
            # Exact match has highest priority - return only this template
            return [tname]
        elif fnmatch.fnmatch(key, tname):
            matches.append(tname)

    return matches


# Add FILE_NATIVE as a sentinel without mixing semantics
SourceSpace: TypeAlias = OrientationCode | Literal["FILE_NATIVE"]


class ImagingModel(BaseModel):
    model_config = {"extra": "forbid"}

    magnet_frequency_MHz: float
    chem_shift_ppm_default: float = 3.7
    chem_shift_apply_by_role: list[Role] = Field(default_factory=lambda: [Role.ANATOMY])
    # optionally, where to read the reference image from if needed by your library
    image_path: Optional[FilePath] = None


ChemMode = Literal["on", "off", "auto"]


class MaterialModel(BaseModel):
    model_config = {"extra": "forbid"}

    name: str = "default"
    color: str = Field("#C8C8C8", description="Hex #RRGGBB")
    opacity: float = 1.0
    wireframe: bool = False
    visible: bool = True
    point_size: float = 5.0

    @field_validator("opacity")
    @classmethod
    def _opacity_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("opacity must be in [0,1]")
        return v


class CanonicalizationDefModel(BaseModel):
    model_config = {"extra": "forbid"}

    source_space: SourceSpace
    scale_to_mm: float = 1.0
    transform: Optional[TransformRefModel] = None
    version: str = "canon-v1"


class CanonicalizationOverrideModel(BaseModel):
    model_config = {"extra": "forbid"}

    # all optional: only supplied fields override the referenced def
    source_space: Optional[SourceSpace] = None
    scale_to_mm: Optional[float] = None
    transform: Optional[TransformRefModel] = None
    version: Optional[str] = None


class GeometrySourceModel(BaseModel):
    model_config = {"extra": "forbid"}

    key: Optional[str] = None
    kind: Optional[Kind] = None

    # Explicit points (file)
    src: Optional[Path] = None
    loader: Optional[str] = None  # e.g., "numpy_points"
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)

    canonicalization_ref: Optional[str] = None
    canonicalization: Optional[CanonicalizationDefModel] = (
        None  # inline (legacy/one-off)
    )
    canonicalization_override: Optional[CanonicalizationOverrideModel] = None

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


class ResourceModel(GeometrySourceModel):
    """
    A load-once file. The loader may return a structured container:
    - dict[str, np.ndarray] of named points
    - dict[str|int, trimesh.Trimesh] for labelmaps
    - GLTF scene graph keyed by node paths, etc.
    """

    @model_validator(mode="after")
    def _require_fields(self):
        if self.key is None:
            raise ValueError("ResourceModel.key is required")
        if self.kind is None:
            raise ValueError("ResourceModel.kind is required")
        if self.src is None:
            raise ValueError("ResourceModel.src is required")
        if self.loader is None:
            raise ValueError("ResourceModel.loader is required")
        return self


class SelectorBase(BaseModel):
    model_config = {"extra": "forbid"}

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

    model_config = {"extra": "forbid"}

    group: Optional[str] = Field(
        default=None, description="e.g., STATIC, FIXTURE, PROBE"
    )
    mask: list[str] = Field(
        default_factory=list, description="Labels it can collide with"
    )


class _TxOpBase(BaseModel):
    model_config = {"extra": "forbid"}

    invert: bool = False


class TranslateTxOpModel(_TxOpBase):
    kind: Literal["translate_mm"] = "translate_mm"
    delta: list[float] = Field(..., min_length=3, max_length=3)


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
    angles_deg: list[float] = Field(..., min_length=3, max_length=3)


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

    model_config = {"extra": "forbid"}

    sequence: list[TransformOp] = Field(default_factory=list)

    # Allow top-level single-op form:
    #   transforms:
    #     fit:  kind: sitk_file, path: ...
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
    model_config = {"extra": "forbid"}

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
class BaseTemplateModel(GeometrySourceModel):
    """Common defaults for both assets and targets."""

    kind: Optional[Kind] = None
    role: Optional[Role] = None

    material_ref: Optional[str] = None
    material: Optional[MaterialModel] = None

    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    canonicalization_ref: Optional[str] = None
    canonicalization: Optional["CanonicalizationDefModel"] = None
    canonicalization_override: Optional["CanonicalizationDefModel"] = None

    caps: Optional[list["Capability"]] = None
    collision: Optional["CollisionPolicyModel"] = None

    pivot_LPS: Optional[list[float]] = None
    bbox_hint: Optional[list[list[float]]] = None

    # Chem-shift hints (optional, ignored if not applicable)
    chem_shift_ppm: Optional[float] = None
    chem_shift_policy: ChemMode = "auto"

    @field_validator("caps", mode="before")
    @classmethod
    def _coerce_caps(cls, v):
        if v is None:
            return v
        if not isinstance(v, list):
            v = [v]
        out = []
        for item in v:
            if isinstance(item, Capability):
                out.append(item)
            elif isinstance(item, str):
                out.append(Capability[item.upper()])
            elif isinstance(item, int):
                out.append(Capability(item))
            else:
                out.append(item)
        return out


class AssetTemplateModel(BaseTemplateModel):
    """Defaults oriented to geometry assets."""

    src: Optional[Path] = None
    loader: Optional[str] = None
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)

    from_resource: Optional[str] = None
    selector: Optional[Selector] = None


class TargetTemplateModel(BaseTemplateModel):
    """Defaults oriented to targets."""

    kind: Optional[Kind] = Kind.POINTS
    role: Optional[Role] = Role.TARGET

    # explicit points
    src: Optional[Path] = None
    loader: Optional[str] = None
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)

    # or derived
    source_key: Optional[str] = None

    # or from resource
    from_resource: Optional[str] = None
    selector: Optional[Selector] = None

    # Reduces geometry from any of the above sources (e.g. COM of a mesh)
    reducer: Optional[str] = None
    reducer_kwargs: dict[str, Any] = Field(default_factory=dict)

    approach_vector: Optional[list[float]] = None
    uncertainty_mm: Optional[float] = None


class BaseSpecModel(BaseModel):
    model_config = {"extra": "forbid"}

    key: Optional[str] = None
    kind: Optional[Kind] = None
    role: Optional[Role] = None

    material_ref: Optional[str] = None
    material: Optional[MaterialModel] = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    # Scene placement (auto-creates scene node if transform or scene_tags set)
    transform: Optional["TransformRefModel"] = None
    scene_tags: list[str] = Field(default_factory=list)
    auto_scene: bool = True  # set False to suppress auto scene node generation

    # Explicit points (file)
    src: Optional[Path] = None
    loader: Optional[str] = None  # e.g., "numpy_points"
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)

    canonicalization_ref: Optional[str] = None
    canonicalization: Optional[CanonicalizationDefModel] = (
        None  # inline (legacy/one-off)
    )
    canonicalization_override: Optional[CanonicalizationOverrideModel] = None

    # capabilities are parsed from strings like ["RENDERABLE", "COLLIDABLE"]
    caps: list[Capability] = Field(default_factory=lambda: [Capability.RENDERABLE])
    collision: CollisionPolicyModel = Field(default_factory=CollisionPolicyModel)

    # UI/layout hints
    pivot_LPS: Optional[list[float]] = Field(default=None, min_length=3, max_length=3)
    bbox_hint: Optional[list[list[float]]] = Field(default=None)

    chem_shift_policy: ChemMode = "auto"
    chem_shift_ppm: Optional[float] = None

    @field_validator("caps", mode="before")
    @classmethod
    def _coerce_caps(cls, v):
        if v is None:
            return v
        if not isinstance(v, list):
            v = [v]
        out = []
        for item in v:
            if isinstance(item, Capability):
                out.append(item)
            elif isinstance(item, str):
                out.append(Capability[item.upper()])
            elif isinstance(item, int):
                out.append(Capability(item))
            else:
                out.append(item)
        return out

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

    # NEW: list of template names to apply, left→right priority
    templates: list[str] = Field(default_factory=list)

    from_resource: Optional[str] = None
    selector: Optional[Selector] = None

    @model_validator(mode="before")
    @classmethod
    def _infer_kind_loader_from_extension(cls, data: Any) -> Any:
        """Infer kind and loader from src extension if not explicitly set."""
        if not isinstance(data, dict):
            return data
        src = data.get("src")
        if src is None:
            return data
        src_str = str(src)
        # Check longer extensions first (e.g., .nii.gz before .nii)
        for ext in sorted(EXTENSION_DEFAULTS.keys(), key=len, reverse=True):
            if src_str.endswith(ext):
                defaults = EXTENSION_DEFAULTS[ext]
                if data.get("kind") is None:
                    data["kind"] = defaults["kind"]
                if data.get("loader") is None:
                    data["loader"] = defaults["loader"]
                break
        return data

    @model_validator(mode="after")
    def _check_source_modes(self):
        has_src = self.src is not None
        has_loader = self.loader is not None
        has_resource = self.from_resource is not None
        has_selector = self.selector is not None

        if has_src ^ has_loader:
            raise ValueError(
                f"Asset '{self.key}' must provide both 'src' and 'loader', or neither."
            )
        if (has_src and has_loader) and (has_resource or has_selector):
            raise ValueError(
                f"Asset '{self.key}': Choose either "
                "(src+loader) or (from_resource+selector), "
                "not both."
            )
        if has_resource ^ has_selector:
            raise ValueError(
                f"Asset '{self.key}': When using "
                "from_resource, you must also provide "
                "a selector."
            )
        return self


def _passthrough_kwargs(
    bulk_model: BaseModel,
    exclude: set[str],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Build kwargs for an expanded child model from a bulk/range model.

    Only forwards fields that were explicitly set in the YAML (via
    ``model_fields_set``), plus any *overrides*.  Fields that used their
    default value are **not** forwarded, so that template merging can
    later fill them in without being clobbered.
    """
    from copy import deepcopy

    kwargs: dict[str, Any] = {}
    for name in bulk_model.model_fields_set - exclude:
        kwargs[name] = deepcopy(getattr(bulk_model, name))
    kwargs.update(overrides)
    return kwargs


class BulkAssetSpecModel(BaseModel):
    """Bulk asset declaration with multiple keys sharing the same configuration.

    Supports placeholders in src:
      - {name}: suffix after last ':' (e.g., 'structure:PL' → 'PL')
      - {key}: full key (e.g., 'structure:PL')

    Example::

        - keys: [structure:PL, structure:MD, structure:CLA]
          src: ${paths.structure_path}/${paths.mouse}-{name}-Mask.nrrd
          templates: [structure]
          transform: headframe_to_lps
    """

    model_config = {"extra": "forbid"}

    keys: list[str] = Field(..., min_length=1)

    # Same fields as AssetSpecModel (except key)
    kind: Optional[Kind] = None
    role: Optional[Role] = None
    material_ref: Optional[str] = None
    material: Optional[MaterialModel] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    # Scene placement
    transform: Optional["TransformRefModel"] = None
    scene_tags: list[str] = Field(default_factory=list)
    auto_scene: bool = True

    # Source (with placeholders)
    src: Optional[str] = None  # string for placeholder support
    loader: Optional[str] = None
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)

    canonicalization_ref: Optional[str] = None
    canonicalization: Optional[CanonicalizationDefModel] = None
    canonicalization_override: Optional[CanonicalizationOverrideModel] = None

    caps: list[Capability] = Field(default_factory=lambda: [Capability.RENDERABLE])
    collision: CollisionPolicyModel = Field(default_factory=CollisionPolicyModel)

    pivot_LPS: Optional[list[float]] = Field(default=None, min_length=3, max_length=3)
    bbox_hint: Optional[list[list[float]]] = Field(default=None)

    chem_shift_policy: ChemMode = "auto"
    chem_shift_ppm: Optional[float] = None

    templates: list[str] = Field(default_factory=list)
    from_resource: Optional[str] = None
    selector: Optional[Selector] = None

    @field_validator("caps", mode="before")
    @classmethod
    def _coerce_caps(cls, v):
        if v is None:
            return v
        if not isinstance(v, list):
            v = [v]
        out = []
        for item in v:
            if isinstance(item, Capability):
                out.append(item)
            elif isinstance(item, str):
                out.append(Capability[item.upper()])
            elif isinstance(item, int):
                out.append(Capability(item))
            else:
                out.append(item)
        return out

    def expand(self) -> list[AssetSpecModel]:
        """Expand into individual AssetSpecModel instances."""
        results = []
        for key in self.keys:
            name = key.split(":")[-1] if ":" in key else key
            src = None
            if self.src is not None:
                src = Path(self.src.replace("{name}", name).replace("{key}", key))
            overrides: dict[str, Any] = {"key": key}
            if src is not None:
                overrides["src"] = src
            kwargs = _passthrough_kwargs(self, {"keys"}, overrides)
            results.append(AssetSpecModel(**kwargs))
        return results


# Type alias for assets that can be either single or bulk
AssetSpecUnion = Union[AssetSpecModel, BulkAssetSpecModel]


class TargetSpecModel(BaseSpecModel):
    """Targets are points; explicit (src+loader) or derived (source_key+reducer)."""

    kind: Kind = Kind.POINTS
    role: Role = Role.TARGET

    # Or derived from an existing asset in catalog
    source_key: Optional[str] = None

    # Or resource
    from_resource: Optional[str] = None
    selector: Optional[Selector] = None

    # Optional reduction (e.g., COM of a selected mesh)
    reducer: Optional[str] = None
    reducer_kwargs: dict[str, Any] = Field(default_factory=dict)

    templates: list[str] = Field(default_factory=list)

    approach_vector: Optional[list[float]] = Field(
        default=None, min_length=3, max_length=3
    )
    uncertainty_mm: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _infer_loader_from_extension(cls, data: Any) -> Any:
        """Infer loader from src extension if not explicitly set."""
        if not isinstance(data, dict):
            return data
        src = data.get("src")
        if src is None:
            return data
        src_str = str(src)
        # Check longer extensions first (e.g., .nii.gz before .nii)
        for ext in sorted(EXTENSION_DEFAULTS.keys(), key=len, reverse=True):
            if src_str.endswith(ext):
                defaults = EXTENSION_DEFAULTS[ext]
                # For targets, kind defaults to POINTS, so only infer loader
                if data.get("loader") is None:
                    data["loader"] = defaults["loader"]
                break
        return data

    @model_validator(mode="after")
    def _check_target_source_and_caps(self):
        explicit = self.src is not None and self.loader is not None
        derived = self.source_key is not None
        from_res = (self.from_resource is not None) and (self.selector is not None)
        paths = sum([explicit, derived, from_res])
        if paths != 1:
            raise ValueError(
                f"Target '{self.key}': provide exactly one of "
                "(src+loader) | (source_key+reducer) | (from_resource+selector)"
            )
        if Capability.COLLIDABLE in self.caps:
            raise ValueError(
                f"Target '{self.key}': targets should not be collidable by default."
            )
        return self


class RangeTargetSpecModel(BaseModel):
    """Bulk target declaration using numeric ranges.

    Supports placeholders in key_pattern and src:
      - {n}: the current number in the range

    Example::

        - key_pattern: "target:hole:{n}"
          range: [1, 13]
          src: ${paths.hole_model_path}/Hole{n}.obj
          templates: [hole]
          transform: implant_to_lps
    """

    model_config = {"extra": "forbid"}

    key_pattern: str  # e.g., "target:hole:{n}"
    range: list[int] = Field(..., min_length=2, max_length=2)  # [start, end] inclusive

    # Same fields as TargetSpecModel (except key)
    kind: Kind = Kind.POINTS
    role: Role = Role.TARGET
    material_ref: Optional[str] = None
    material: Optional[MaterialModel] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    # Scene placement
    transform: Optional["TransformRefModel"] = None
    scene_tags: list[str] = Field(default_factory=list)
    auto_scene: bool = True

    # Source (with placeholders)
    src: Optional[str] = None  # string for placeholder support
    loader: Optional[str] = None
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)

    source_key: Optional[str] = None  # also supports {n} placeholder

    from_resource: Optional[str] = None
    selector: Optional[Selector] = None

    reducer: Optional[str] = None
    reducer_kwargs: dict[str, Any] = Field(default_factory=dict)

    canonicalization_ref: Optional[str] = None
    canonicalization: Optional[CanonicalizationDefModel] = None
    canonicalization_override: Optional[CanonicalizationOverrideModel] = None

    caps: list[Capability] = Field(default_factory=lambda: [Capability.RENDERABLE])
    collision: CollisionPolicyModel = Field(default_factory=CollisionPolicyModel)

    pivot_LPS: Optional[list[float]] = Field(default=None, min_length=3, max_length=3)
    bbox_hint: Optional[list[list[float]]] = Field(default=None)

    chem_shift_policy: ChemMode = "auto"
    chem_shift_ppm: Optional[float] = None

    templates: list[str] = Field(default_factory=list)

    approach_vector: Optional[list[float]] = Field(
        default=None, min_length=3, max_length=3
    )
    uncertainty_mm: Optional[float] = None

    @field_validator("caps", mode="before")
    @classmethod
    def _coerce_caps(cls, v):
        if v is None:
            return v
        if not isinstance(v, list):
            v = [v]
        out = []
        for item in v:
            if isinstance(item, Capability):
                out.append(item)
            elif isinstance(item, str):
                out.append(Capability[item.upper()])
            elif isinstance(item, int):
                out.append(Capability(item))
            else:
                out.append(item)
        return out

    def expand(self) -> list[TargetSpecModel]:
        """Expand into individual TargetSpecModel instances."""
        start, end = self.range
        results = []
        for n in range(start, end + 1):
            n_str = str(n)
            overrides: dict[str, Any] = {
                "key": self.key_pattern.replace("{n}", n_str),
            }
            if self.src is not None:
                overrides["src"] = Path(self.src.replace("{n}", n_str))
            if self.source_key is not None:
                overrides["source_key"] = self.source_key.replace("{n}", n_str)
            kwargs = _passthrough_kwargs(
                self, {"key_pattern", "range"}, overrides
            )
            results.append(TargetSpecModel(**kwargs)
            )
        return results


class DerivedTargetSpecModel(BaseModel):
    """Bulk target declaration deriving targets from existing assets.

    Creates targets that reference existing assets via source_key.

    Example::

        - derive_from: [structure:PL, structure:MD, structure:CLA]
          key_prefix: "target:"
          templates: [structure]
          transform: headframe_to_lps
    """

    model_config = {"extra": "forbid"}

    derive_from: list[str] = Field(..., min_length=1)  # asset keys to derive from
    key_prefix: str = "target:"  # prepended to derive target key

    # Same fields as TargetSpecModel (except key, source_key)
    kind: Kind = Kind.POINTS
    role: Role = Role.TARGET
    material_ref: Optional[str] = None
    material: Optional[MaterialModel] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    # Scene placement
    transform: Optional["TransformRefModel"] = None
    scene_tags: list[str] = Field(default_factory=list)
    auto_scene: bool = True

    from_resource: Optional[str] = None
    selector: Optional[Selector] = None

    reducer: Optional[str] = None
    reducer_kwargs: dict[str, Any] = Field(default_factory=dict)

    canonicalization_ref: Optional[str] = None
    canonicalization: Optional[CanonicalizationDefModel] = None
    canonicalization_override: Optional[CanonicalizationOverrideModel] = None

    caps: list[Capability] = Field(default_factory=lambda: [Capability.RENDERABLE])
    collision: CollisionPolicyModel = Field(default_factory=CollisionPolicyModel)

    pivot_LPS: Optional[list[float]] = Field(default=None, min_length=3, max_length=3)
    bbox_hint: Optional[list[list[float]]] = Field(default=None)

    chem_shift_policy: ChemMode = "auto"
    chem_shift_ppm: Optional[float] = None

    templates: list[str] = Field(default_factory=list)

    approach_vector: Optional[list[float]] = Field(
        default=None, min_length=3, max_length=3
    )
    uncertainty_mm: Optional[float] = None

    @field_validator("caps", mode="before")
    @classmethod
    def _coerce_caps(cls, v):
        if v is None:
            return v
        if not isinstance(v, list):
            v = [v]
        out = []
        for item in v:
            if isinstance(item, Capability):
                out.append(item)
            elif isinstance(item, str):
                out.append(Capability[item.upper()])
            elif isinstance(item, int):
                out.append(Capability(item))
            else:
                out.append(item)
        return out

    def expand(self) -> list[TargetSpecModel]:
        """Expand into individual TargetSpecModel instances."""
        results = []
        for asset_key in self.derive_from:
            suffix = asset_key.split(":")[-1] if ":" in asset_key else asset_key
            overrides: dict[str, Any] = {
                "key": f"{self.key_prefix}{suffix}",
                "source_key": asset_key,
            }
            kwargs = _passthrough_kwargs(
                self, {"derive_from", "key_prefix"}, overrides
            )
            results.append(TargetSpecModel(**kwargs))
        return results


# Type alias for targets that can be single, range, or derived
TargetSpecUnion = Union[TargetSpecModel, RangeTargetSpecModel, DerivedTargetSpecModel]


# -----------------------------------------------------------------------------
# Scene (WHERE: instances and bindings)
# -----------------------------------------------------------------------------


class SceneNodeModel(BaseModel):
    model_config = {"extra": "forbid"}

    key: str
    asset: str = Field(description="Key of an AssetSpec in catalog")
    tags: list[str] = Field(default_factory=list)

    # Reference a named transform (from ConfigModel.transforms) or None for identity
    transform: Optional[TransformRefModel] = None

    # Optional domain binding for pose (probes): ties node to domain.probes[name]
    pose_source_probe: Optional[str] = Field(
        default=None,
        description="If set, renderer takes pose from plan.probes[pose_source_probe].",
    )


class SceneModel(BaseModel):
    model_config = {"extra": "forbid"}

    nodes: list[SceneNodeModel] = Field(default_factory=list)
    _explicit_node_keys: set[str] = PrivateAttr(default_factory=set)


# -----------------------------------------------------------------------------
# Domain (mechanics: arcs, probes, calibrations, target declarations)
# -----------------------------------------------------------------------------
class CatalogTargetRefModel(BaseModel):
    model_config = {"extra": "forbid"}

    kind: Literal["catalog"] = "catalog"
    key: str  # TargetSpecModel.key


class NodeTargetRefModel(BaseModel):
    model_config = {"extra": "forbid"}

    kind: Literal["node"] = "node"
    key: str  # SceneNodeModel.id / NodeInstance.id


class InlineTargetRefModel(BaseModel):
    """Ad-hoc target specified as a single RAS coordinate."""

    model_config = {"extra": "forbid"}

    kind: Literal["inline"] = "inline"
    point_RAS: list[float] = Field(..., min_length=3, max_length=3)


TargetRef = Annotated[
    Union[CatalogTargetRefModel, NodeTargetRefModel, InlineTargetRefModel],
    Field(discriminator="kind"),
]


class ProbeDeclModel(BaseModel):
    model_config = {"extra": "forbid"}

    kind: str
    arc: Optional[str] = None
    slider_ml: float = 0.0
    spin: float = 0.0

    # Per-probe AP override. If None, AP comes from arc angle at load time.
    ap_local: Optional[float] = None
    bind_ap_to_arc: bool = True

    target: TargetRef
    past_target_mm: float = 0.0
    offsets_RA: list[float] = Field(
        default_factory=lambda: [0.0, 0.0], min_length=2, max_length=2
    )

    # initial lock state; actual calibration affine comes from 'calibrations' map
    calibrated: bool = False

    # Auto scene node generation (creates "probe:{name}" node)
    auto_scene: bool = True
    scene_tags: list[str] = Field(default_factory=lambda: ["probe", "dynamic"])

    @model_validator(mode="before")
    @classmethod
    def _coerce_target(cls, v):
        if isinstance(v, dict) and "target" in v:
            t = v["target"]
            if isinstance(t, str):
                v["target"] = {"kind": "catalog", "key": t}
            elif isinstance(t, (list, tuple)) and len(t) == 3:
                v["target"] = {"kind": "inline", "point_RAS": list(t)}
        return v


class CalibrationRefModel(BaseModel):
    """Reference a specific calibration entry inside a calibration file."""

    model_config = {"extra": "forbid"}

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

    model_config = {"extra": "forbid"}

    offset_RAS: list[float] = Field(default_factory=list, min_length=3, max_length=3)
    rotation_z: float = 0.0


class CalibrationSourceModel(BaseModel):
    """
    One calibration 'bank' source:
      - EITHER a single file (e.g., .xlsx). In this case NO reticle is allowed.
      - OR a directory for parallax. In this case a reticle IS REQUIRED.
    """

    model_config = {"extra": "forbid"}

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
    model_config = {"extra": "forbid"}

    files: dict[str, CalibrationSourceModel] = Field(default_factory=dict)
    # probe_name → {"cal_id": "...", "probe_code": "..."} OR "cal_id:probe_code"
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
    model_config = {"extra": "forbid"}

    arcs: dict[str, float] = Field(
        default_factory=dict, description="arc_id → AP angle (deg)"
    )
    probes: dict[str, ProbeDeclModel] = Field(
        default_factory=dict, description="probe_name → probe declaration"
    )
    reticles: dict[str, CalibrationReticleModel] = Field(default_factory=dict)
    calibrations: CalibrationsModel = Field(default_factory=CalibrationsModel)
    # Targets can also live in the catalog (TargetSpecModel) or as inline targets.


# -----------------------------------------------------------------------------
# Transforms, Paths, Options
# -----------------------------------------------------------------------------


class PathsModel(BaseModel):
    """Freeform helper; keep loose so Hydra/OmegaConf interpolation is easy."""

    model_config = {"extra": "allow"}

    def __init__(self, **data):
        super().__init__(**data)


class OptionsModel(BaseModel):
    model_config = {"extra": "forbid"}

    color_map: str = "rainbow"
    remove_last_color: bool = True


# -----------------------------------------------------------------------------
# Root config (everything together) + cross-reference validation
# -----------------------------------------------------------------------------


class ConfigModel(BaseModel):
    model_config = {"extra": "forbid"}

    version: int = 1

    paths: PathsModel = Field(default_factory=PathsModel)
    imaging: Optional[ImagingModel] = None
    resources: list[ResourceModel] = Field(default_factory=list)
    materials: dict[str, MaterialModel] = Field(default_factory=dict)

    # Catalog
    asset_templates: dict[str, AssetTemplateModel] = Field(default_factory=dict)
    target_templates: dict[str, TargetTemplateModel] = Field(default_factory=dict)
    assets: list[AssetSpecUnion] = Field(default_factory=list)
    targets: list[TargetSpecUnion] = Field(default_factory=list)

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
    def _xref_and_expand_templates(self):  # noqa: C901
        errors: list[str] = []

        # ---------- Expand bulk specs first ----------
        expanded_assets: list[AssetSpecModel] = []
        for item in self.assets:
            if isinstance(item, BulkAssetSpecModel):
                expanded_assets.extend(item.expand())
            else:
                expanded_assets.append(item)
        self.assets = expanded_assets

        expanded_targets: list[TargetSpecModel] = []
        for item in self.targets:
            if isinstance(item, (RangeTargetSpecModel, DerivedTargetSpecModel)):
                expanded_targets.extend(item.expand())
            else:
                expanded_targets.append(item)
        self.targets = expanded_targets

        # ---------- Auto-match templates by key glob ----------
        for asset in self.assets:
            if not asset.templates:  # no explicit templates
                auto = _find_matching_templates(asset.key, self.asset_templates)
                if auto:
                    object.__setattr__(asset, "templates", auto)

        for target in self.targets:
            if not target.templates:
                auto = _find_matching_templates(target.key, self.target_templates)
                if auto:
                    object.__setattr__(target, "templates", auto)

        # ---------- sets for quick membership ----------
        asset_keys = {a.key for a in self.assets}
        target_keys = {t.key for t in self.targets}
        node_keys = {n.key for n in self.scene.nodes}
        transform_keys = set(self.transforms.keys())
        arc_keys = set(self.plan.arcs.keys())
        probe_names = set(self.plan.probes.keys())
        reticle_names = set(self.plan.reticles.keys())
        cal_files = self.plan.calibrations.files  # dict[id -> CalFileDecl]

        node_idx_by_key = {k: i for i, k in enumerate(node_keys)}

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
                        f"{where_prefix} '{_where_key(spec)}': "
                        f"template '{tref}' not found"
                    )

        for a in self.assets:
            _check_template_ref(a, self.asset_templates, "asset")
        for t in self.targets:
            _check_template_ref(t, self.target_templates, "target")
        # Expand templates into concrete specs
        if self.asset_templates and self.assets:
            assets = []
            for a in self.assets:
                assets.append(
                    apply_templates_generic(
                        merge_asset_template_model_dumps,
                        a,
                        a.templates,
                        self.asset_templates,
                    )
                )
            self.assets = assets
        if self.target_templates and self.targets:
            targets = []
            for t in self.targets:
                targets.append(
                    apply_templates_generic(
                        merge_target_template_model_dumps,
                        t,
                        t.templates,
                        self.target_templates,
                    )
                )
            self.targets = targets

        # ---------- Infer kind/loader from extension and role from key ----------
        for asset in self.assets:
            _infer_from_extension(asset)
            _infer_role_from_key(asset)
        for target in self.targets:
            _infer_from_extension(target)
            _infer_role_from_key(target)

        # ---------- Auto-generate scene nodes ----------
        generated_nodes: list[SceneNodeModel] = []
        explicit_keys = {n.key for n in self.scene.nodes}
        self.scene._explicit_node_keys = set(explicit_keys)

        # From assets with transform or scene_tags
        for asset in self.assets:
            if not asset.auto_scene:
                continue
            if asset.key in explicit_keys:
                continue  # explicit node takes precedence
            if asset.transform or asset.scene_tags:
                generated_nodes.append(
                    SceneNodeModel(
                        key=asset.key,
                        asset=asset.key,
                        transform=asset.transform,
                        tags=asset.scene_tags,
                    )
                )

        # From targets with transform or scene_tags
        for target in self.targets:
            if not target.auto_scene:
                continue
            if target.key in explicit_keys:
                continue
            if target.transform or target.scene_tags:
                generated_nodes.append(
                    SceneNodeModel(
                        key=target.key,
                        asset=target.key,
                        transform=target.transform,
                        tags=target.scene_tags,
                    )
                )

        # From plan.probes (auto-generate probe:{name} nodes)
        for probe_name, probe_decl in self.plan.probes.items():
            node_key = f"probe:{probe_name}"
            if not probe_decl.auto_scene:
                continue
            if node_key in explicit_keys:
                continue
            asset_key = f"probe:{probe_decl.kind}"
            generated_nodes.append(
                SceneNodeModel(
                    key=node_key,
                    asset=asset_key,
                    tags=probe_decl.scene_tags,
                    pose_source_probe=probe_name,
                )
            )

        # Prepend generated nodes (explicit nodes are already in self.scene.nodes)
        if generated_nodes:
            self.scene.nodes = generated_nodes + list(self.scene.nodes)

        # Update node_keys after generation
        node_keys = {n.key for n in self.scene.nodes}
        node_idx_by_key = {n.key: i for i, n in enumerate(self.scene.nodes)}

        def _check_spec_kind(spec, where_prefix: str, allowable=None):
            kind = getattr(spec, "kind", None)
            if not kind:
                err(f"{where_prefix} '{_where_key(spec)}': kind not set")
            if allowable and kind not in allowable:
                err(f"{where_prefix} '{_where_key(spec)}': kind '{kind}' not allowed")

        def _check_spec_role(spec, where_prefix: str, allowable=None):
            role = getattr(spec, "role", None)
            if not role:
                err(f"{where_prefix} '{_where_key(spec)}': role not set")
            if allowable and role not in allowable:
                err(f"{where_prefix} '{_where_key(spec)}': role '{role}' not allowed")

        def _check_asset_spec_src_loader(spec: AssetSpecModel):
            if (spec.src is None) ^ (spec.loader is None):
                err(
                    f"Asset '{_where_key(spec)}' must provide "
                    "both 'src' and 'loader', or neither (if injected elsewhere)."
                )
            if (spec.src and spec.loader) and (spec.from_resource or spec.selector):
                err(
                    f"Asset '{_where_key(spec)}': Choose either "
                    "(src+loader) or (from_resource+selector), not both."
                )
            if (spec.from_resource is None) ^ (spec.selector is None):
                # only one given
                err(
                    f"Asset '{_where_key(spec)}': When using "
                    "from_resource, you must also provide a selector."
                )

        def _check_target_spec_single_source_and_caps(spec: TargetSpecModel):
            explicit = spec.src is not None and spec.loader is not None
            derived = spec.source_key is not None
            from_res = (spec.from_resource is not None) and (spec.selector is not None)
            paths = sum([explicit, derived, from_res])
            if paths != 1:
                err(
                    f"Target '{_where_key(spec)}': provide exactly one of "
                    "(src+loader) | (source_key+reducer) | (from_resource+selector)"
                )
            if Capability.COLLIDABLE in spec.caps:
                err(f"Target '{_where_key(spec)}': targets should not be collidable.")

        def _check_material_ref(spec, where_prefix: str):
            mref = getattr(spec, "material_ref", None)
            if mref and mref not in self.materials:
                err(
                    f"{where_prefix} '{_where_key(spec)}': "
                    f"material_ref '{mref}' not found"
                )

        for a in self.assets:
            _check_material_ref(a, "asset")
            _check_spec_kind(a, "asset")
            _check_spec_role(a, "asset")
            _check_asset_spec_src_loader(a)
        for t in self.targets:
            _check_material_ref(t, "target")
            _check_spec_kind(t, "target", allowable={Kind.POINTS})
            _check_spec_role(t, "target", allowable={Role.TARGET})
            _check_target_spec_single_source_and_caps(t)

        for name, tmpl in self.asset_templates.items():
            if tmpl.material_ref and tmpl.material_ref not in self.materials:
                err(
                    f"asset_templates['{name}']: "
                    f"material_ref '{tmpl.material_ref}' not found"
                )
        for name, tmpl in self.target_templates.items():
            if tmpl.material_ref and tmpl.material_ref not in self.materials:
                err(
                    f"target_templates['{name}']: "
                    f"material_ref '{tmpl.material_ref}' not found"
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
                        f"{where_prefix} '{_where_key(spec)}': "
                        f"canonicalization_ref '{cref}' not found"
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
        catalog_keys = asset_keys | target_keys
        dups = asset_keys & target_keys
        if dups:
            err(f"Catalog has duplicate keys: {dups}")
        for n in self.scene.nodes:
            if n.asset not in catalog_keys:
                err(f"scene.nodes['{n.key}']: asset '{n.asset}' not found in catalog")
            _check_transform_ref(
                getattr(n, "transform", None), f"scene.nodes['{n.key}'].transform"
            )
            if (
                getattr(n, "pose_source_probe", None)
                and n.pose_source_probe not in probe_names
            ):
                err(
                    f"scene.nodes['{n.key}'].pose_source_probe "
                    f"'{n.pose_source_probe}' not in plan.probes"
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
        seen_catalog_target_names = set()
        seen_node_target_names = set()
        for pname, p in self.plan.probes.items():
            if p.arc is not None and p.arc not in arc_keys:
                err(f"plan.probes['{pname}']: arc '{p.arc}' not found in plan.arcs")
            if p.bind_ap_to_arc and p.arc is None:
                err(f"plan.probes['{pname}']: bind_ap_to_arc=True but arc is not set")
            if p.target.kind == "inline":
                pass  # self-contained RAS point, no xref needed
            elif p.target.kind == "catalog":
                target_key = p.target.key
                if target_key not in target_keys:
                    err(
                        f"plan.probes['{pname}']: catalog target "
                        f"'{target_key}' not found"
                    )
                seen_catalog_target_names.add(target_key)
            else:  # node
                target_key = p.target.key
                if target_key not in node_keys:
                    err(
                        f"plan.probes['{pname}']: node target "
                        f"'{target_key}' not in scene.nodes"
                    )
                node_idx = node_idx_by_key.get(target_key)
                target_ref = self.scene.nodes[node_idx].asset
                if target_ref in asset_keys:
                    err(
                        f"plan.probes['{pname}']: node target '{target_key}' "
                        f"references asset '{target_ref}' instead of target"
                    )
                seen_node_target_names.add(target_key)
        conflicting_names = seen_catalog_target_names.intersection(
            seen_node_target_names
        )
        if conflicting_names:
            err(
                f"plan.probes: targets '{conflicting_names}' ambiguous "
                "(in both catalog and node targets)"
            )
        # each calibration file must reference an existing reticle
        for cal_id, cal in cal_files.items():
            if cal.reticle not in reticle_names:
                err(
                    f"plan.calibrations.files['{cal_id}']: "
                    f"reticle '{cal.reticle}' not in plan.reticles"
                )

        # probe_to_ref must point to a valid probe and cal file id
        for probe_name, ref in self.plan.calibrations.probe_to_ref.items():
            if probe_name not in probe_names:
                err(
                    f"plan.calibrations.probe_to_ref: "
                    f"probe '{probe_name}' not in plan.probes"
                )
            if ref.cal_id not in cal_files:
                err(
                    f"plan.calibrations.probe_to_ref['{probe_name}']: "
                    f"cal_id '{ref.cal_id}' not in files"
                )

        # ---------- canonicalization defs themselves ----------
        for cname, cdef in self.canonicalizations.items():
            _check_transform_ref(
                cdef.transform, f"canonicalizations['{cname}'].transform"
            )
            # Require a transform (key OR inline) when source_space is FILE_NATIVE
            if cdef.source_space == "FILE_NATIVE" and cdef.transform is None:
                errors.append(
                    f"canonicalizations['{cname}']: FILE_NATIVE requires "
                    "a transform (key or inline)"
                )

        for seq, kind in (
            (self.assets, "asset"),
            (self.targets, "target"),
            (self.resources, "resource"),
        ):
            for item in seq:
                eff = _effective_canon_for_spec(item, self.canonicalizations)
                if eff is None:
                    continue

                # ----- XOR rule -----
                named_space = eff.source_space != "FILE_NATIVE"
                has_tx = _has_transform(eff.transform)

                # Exactly one must be true:
                if named_space == has_tx:  # both True or both False
                    # Build a concise, actionable message:
                    key = getattr(item, "key", "?")
                    if named_space and has_tx:
                        err(
                            f"{kind} '{key}': use source_space "
                            f"({eff.source_space}) OR transform, not both"
                        )
                    else:
                        err(
                            f"{kind} '{key}': provide source_space "
                            "(RAS/LPS/…) OR transform (for FILE_NATIVE)"
                        )

        # ---------- final ----------
        if errors:
            raise ValueError(
                "Config cross-reference errors:\n  - " + "\n  - ".join(errors)
            )
        return self

    def to_explicit_dict(self) -> dict[str, Any]:
        """Export expanded config as a dict suitable for YAML serialization.

        The output contains no bulk specs (they've been expanded), templates
        have been applied, and auto-generated scene nodes are included.
        The result can be re-loaded as a valid ConfigModel.

        Returns
        -------
        dict
            JSON-serializable dict (Path→str, Enum→value).
        """
        return self.model_dump(mode="json", exclude_defaults=False)


def expand_config(config_data: dict[str, Any]) -> dict[str, Any]:
    """Parse config, expand all bulk specs and auto-generation, return explicit dict.

    This is a convenience function for the common pattern of loading a concise
    config and exporting it to an explicit form for inspection or archival.

    Parameters
    ----------
    config_data : dict
        Raw config data (e.g., from YAML).

    Returns
    -------
    dict
        Fully expanded, explicit config that can be re-loaded.
    """
    config = ConfigModel.model_validate(config_data)
    return config.to_explicit_dict()


def _as_overlay(model: BaseModel | None) -> dict[str, Any]:
    return {} if model is None else model.model_dump(exclude_unset=True)


def _union_list(a: Optional[list[Any]], b: Optional[list[Any]]) -> Optional[list[Any]]:
    if a is None and b is None:
        return None
    seen: set[Any] = set()
    out: list[Any] = []
    for src in (a or []), (b or []):
        for x in src:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


def _merge_dict_shallow(
    a: Optional[dict[str, Any]], b: Optional[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return {**a, **b}


def _merge_collision(
    base: Optional[dict[str, Any]], over: Optional[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    if base is None:
        return over
    if over is None:
        return base
    # union mask; overlay group if provided
    merged_mask = _union_list(base.get("mask"), over.get("mask")) or []
    group = over.get("group", base.get("group"))
    base.update(over)
    base["mask"] = merged_mask
    base["group"] = group
    return type(base)(**base)


# -----------------------------
# Asset template merge
# -----------------------------


def _merge_asset_source_fields(
    base: dict[str, Any],
    over: dict[str, Any],
) -> dict[str, Any]:
    """
    Choose exactly one source mode:
      - (src + loader [+ loader_kwargs])
      - (from_resource + selector)
    If overlay specifies any field of a mode, that mode wins; the other mode is
    cleared.
    """
    out: dict[str, Any] = {}

    over_file = (over.get("src", None) is not None) or (
        over.get("loader", None) is not None
    )
    over_res = (over.get("from_resource", None) is not None) or (
        over.get("selector", None) is not None
    )

    if over_file and over_res:
        raise ValueError("Asset: overlay specifies both file and resource source modes")

    if over_file:
        # file mode wins; overlay values take priority, else fall back to base
        out["src"] = over.get("src", None) or base.get("src", None)
        out["loader"] = over.get("loader", None) or base.get("loader", None)
        out["loader_kwargs"] = _merge_dict_shallow(
            base.get("loader_kwargs", None), over.get("loader_kwargs", None)
        )
        out["from_resource"] = None
        out["selector"] = None

    elif over_res:
        out["from_resource"] = over.get("from_resource", None) or base.get(
            "from_resource", None
        )
        out["selector"] = over.get("selector", None) or base.get("selector", None)

        # clear file mode
        out["src"] = None
        out["loader"] = None
        out["loader_kwargs"] = {}
    else:
        # overlay did not change mode → keep base (as-is)
        out["src"] = over.get("src", None) or base.get("src", None)
        out["loader"] = over.get("loader", None) or base.get("loader", None)
        out["loader_kwargs"] = (
            over.get("loader_kwargs", {}) or base.get("loader_kwargs", {}) or {}
        )
        out["from_resource"] = over.get("from_resource", None) or base.get(
            "from_resource", None
        )
        out["selector"] = over.get("selector", None) or base.get("selector", None)

    return out


def merge_asset_template_model_dumps(
    base: dict[str, Any],
    over: dict[str, Any],
) -> dict[str, Any]:
    out = deepcopy(base)
    out.update(over)

    # unions
    out["tags"] = _union_list(base.get("tags"), over.get("tags")) or []
    out["caps"] = _union_list(base.get("caps"), over.get("caps")) or None

    # metadata shallow merge
    out["metadata"] = _merge_dict_shallow(base.get("metadata"), over.get("metadata"))

    # nested merges
    out["material"] = _merge_dict_shallow(
        base.get("material"), over.get("material", None)
    )
    out["canonicalization"] = _merge_dict_shallow(
        base.get("canonicalization"), over.get("canonicalization", None)
    )
    out["canonicalization_override"] = _merge_dict_shallow(
        base.get("canonicalization_override"),
        over.get("canonicalization_override", None),
    )
    out["collision"] = _merge_collision(
        base.get("collision"), over.get("collision", None)
    )

    # refs (replace-on-write)
    out["material_ref"] = over.get("material_ref", None) or base.get(
        "material_ref", None
    )

    # hints (replace-on-write)
    out["pivot_LPS"] = over.get("pivot_LPS", None) or base.get("pivot_LPS", None)
    out["bbox_hint"] = over.get("bbox_hint", None) or base.get("bbox_hint", None)

    # chem-shift hints (replace-on-write)
    out["chem_shift_ppm"] = (
        over.get("chem_shift_ppm", None)
        if "chem_shift_ppm" in over
        else base.get("chem_shift_ppm", None)
    )
    out["chem_shift_policy"] = over.get("chem_shift_policy", None) or base.get(
        "chem_shift_policy", None
    )

    # source modes
    out.update(_merge_asset_source_fields(base, over))

    return out


T = TypeVar("T", bound=BaseSpecModel)


def apply_templates_generic(
    mergefun: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    spec: T,
    template_names: list[str],
    registry: dict[str, Any],
) -> T:
    if not template_names:
        return spec

    # Start with spec
    base = spec.model_dump()
    # fold templates left→right
    merged = base
    for name in template_names:
        t = registry.get(name)
        if t is None:
            continue  # already reported by _check_template_ref
        merged = mergefun(merged, _as_overlay(t))

    merged_tmpl = mergefun(merged, _as_overlay(spec))  # overlay spec onto templates
    # materialize back to AssetSpecModel; ensure we keep spec's identity fields
    merged_tmpl["key"] = spec.key
    merged_tmpl["templates"] = []
    return spec.__class__(**merged_tmpl)


# -----------------------------
# Target template merge
# -----------------------------


def _detect_target_mode(target_dump: dict[str, Any]) -> Optional[str]:
    explicit = (target_dump.get("src", None) is not None) or (
        target_dump.get("loader", None) is not None
    )
    derived = target_dump.get("source_key", None) is not None
    res = (target_dump.get("from_resource", None) is not None) or (
        target_dump.get("selector", None) is not None
    )
    cnt = int(explicit) + int(derived) + int(res)
    if cnt == 0:
        return None
    if cnt == 1:
        return "explicit" if explicit else ("derived" if derived else "resource")
    return "conflict"


def _merge_target_source_fields(
    base: dict[str, Any],
    over: dict[str, Any],
) -> dict[str, Any]:
    """
    Exactly one of:
      - explicit: src + loader [+ loader_kwargs]
      - derived : source_key + reducer
      - resource: from_resource + selector
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
        out["src"] = over.get("src", None) or base.get("src", None)
        out["loader"] = over.get("loader", None) or base.get("loader", None)
        out["loader_kwargs"] = _merge_dict_shallow(
            base.get("loader_kwargs", None), over.get("loader_kwargs", None)
        )
        out["reducer"] = over.get("reducer", None) or base.get("reducer", None)
        out["reducer_kwargs"] = _merge_dict_shallow(
            base.get("reducer_kwargs", None), over.get("reducer_kwargs", None)
        )
        # clear others
        out.update(
            {
                "source_key": None,
                "from_resource": None,
                "selector": None,
            }
        )

    elif mode == "derived":
        out["source_key"] = over.get("source_key", None) or getattr(
            base, "source_key", None
        )
        out["reducer"] = over.get("reducer", None) or base.get("reducer", None)
        out["reducer_kwargs"] = _merge_dict_shallow(
            base.get("reducer_kwargs", None), over.get("reducer_kwargs", None)
        )
        # clear others
        out.update(
            {
                "src": None,
                "loader": None,
                "loader_kwargs": {},
                "from_resource": None,
                "selector": None,
            }
        )

    elif mode == "resource":
        out["from_resource"] = over.get("from_resource", None) or getattr(
            base, "from_resource", None
        )
        out["selector"] = over.get("selector", None) or getattr(base, "selector", None)
        # clear others
        out.update(
            {
                "src": None,
                "loader": None,
                "loader_kwargs": {},
                "source_key": None,
            }
        )

    else:
        # neither base nor overlay specified a mode → keep as is (all None/empty)
        out.update(
            {
                "src": base.get("src", None),
                "loader": base.get("loader", None),
                "loader_kwargs": base.get("loader_kwargs", {}) or {},
                "source_key": base.get("source_key", None),
                "from_resource": base.get("from_resource", None),
                "selector": base.get("selector", None),
            }
        )

    return out


def merge_target_template_model_dumps(
    base: dict[str, Any],
    over: dict[str, Any],
) -> dict[str, Any]:
    out = deepcopy(base)
    out.update(over)

    # unions
    out["tags"] = _union_list(base.get("tags"), over.get("tags")) or []
    out["caps"] = _union_list(base.get("caps"), over.get("caps")) or None

    # metadata shallow merge
    out["metadata"] = _merge_dict_shallow(base.get("metadata"), over.get("metadata"))

    # nested merges
    out["material"] = _merge_dict_shallow(
        base.get("material"), over.get("material", None)
    )
    out["canonicalization"] = _merge_dict_shallow(
        base.get("canonicalization"), over.get("canonicalization", None)
    )
    out["canonicalization_override"] = _merge_dict_shallow(
        base.get("canonicalization_override"),
        over.get("canonicalization_override", None),
    )
    out["collision"] = _merge_collision(
        base.get("collision"), over.get("collision", None)
    )

    # refs (replace-on-write)
    out["material_ref"] = over.get("material_ref", None) or base.get(
        "material_ref", None
    )

    # hints (replace-on-write)
    out["pivot_LPS"] = over.get("pivot_LPS", None) or base.get("pivot_LPS", None)
    out["bbox_hint"] = over.get("bbox_hint", None) or base.get("bbox_hint", None)
    out["approach_vector"] = over.get("approach_vector", None) or base.get(
        "approach_vector", None
    )
    out["uncertainty_mm"] = over.get("uncertainty_mm", None) or base.get(
        "uncertainty_mm", None
    )

    # chem-shift hints (targets rarely need it; still honor if present)
    out["chem_shift_ppm"] = (
        over.get("chem_shift_ppm", None)
        if "chem_shift_ppm" in over
        else base.get("chem_shift_ppm", None)
    )
    out["chem_shift_policy"] = over.get("chem_shift_policy", None) or base.get(
        "chem_shift_policy", None
    )

    # source modes
    out.update(_merge_target_source_fields(base, over))

    return out


# ---------------------------------------
def _effective_canon_for_spec(
    spec: Any, canonicalizations: dict[str, "CanonicalizationDefModel"]
) -> Optional["CanonicalizationDefModel"]:
    # merge: ref -> inline -> override (last wins)
    c = None
    if getattr(spec, "canonicalization_ref", None):
        base = canonicalizations.get(spec.canonicalization_ref)
        if base is not None:
            c = deepcopy(base)
    if getattr(spec, "canonicalization", None):
        over = spec.canonicalization
        c = CanonicalizationDefModel(
            **{**(c.model_dump() if c else {}), **over.model_dump(exclude_none=True)}
        )
    if getattr(spec, "canonicalization_override", None):
        over = spec.canonicalization_override
        c = CanonicalizationDefModel(
            **{**(c.model_dump() if c else {}), **over.model_dump(exclude_none=True)}
        )
    return c


def _has_transform(ref: Optional["TransformRefModel"]) -> bool:
    # Your TransformRefModel already enforces exactly one of {key | inline} if present.
    return bool(ref and (ref.key is not None or ref.inline is not None))
