"""Leaf config models: sources, materials, transforms and imaging."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    TypeAlias,
    Union,
)

from pydantic import (
    BaseModel,
    Field,
    FilePath,
    field_validator,
    model_validator,
)

from aind_rutter.domain.enums import Kind, MRSignal, OrientationCode, Role

# -----------------------------------------------------------------------------
# Extension-based inference for kind and loader
# -----------------------------------------------------------------------------


def find_matching_templates(key: str, templates: dict[str, Any]) -> list[str]:
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


# Fields the models no longer have, and what says the same thing now.
# `extra="forbid"` already refuses them; naming the replacement is the
# difference between "unexpected key" and knowing what to write instead.
RETIRED_FIELDS: dict[str, str] = {
    "caps": "`collidable: true`",
    "collision": "`collidable: true` plus `role: probe` or `role: fixture`",
    "chem_shift_policy": "`mr_signal`",
    "chem_shift_apply_by_role": "`mr_signal`, stated per asset and target",
    "options": "nothing — no code has read it since the notebook frontend went",
}

# The last commit whose `scripts/upgrade_config.py` can read the fields above
# and derive their replacements. Check it out to migrate a config written
# before them; this version cannot, because the models no longer hold them.
UPGRADE_FROM_COMMIT = "e7e345c"


def retired_fields_in(doc: Any) -> set[str]:
    """Which retired fields a raw config still states, and where they can be.

    Scoped to the blocks that carried them rather than walking the whole
    document, so a subject is free to use one of these words for its own key.
    """
    found: set[str] = set()
    if not isinstance(doc, dict):
        return found
    found |= RETIRED_FIELDS.keys() & doc.keys()
    found |= RETIRED_FIELDS.keys() & (doc.get("imaging") or {}).keys()
    for section in ("assets", "targets"):
        for item in doc.get(section) or ():
            if isinstance(item, dict):
                found |= RETIRED_FIELDS.keys() & item.keys()
    for section in ("asset_templates", "target_templates"):
        for item in (doc.get(section) or {}).values():
            if isinstance(item, dict):
                found |= RETIRED_FIELDS.keys() & item.keys()
    return found


# Add FILE_NATIVE as a sentinel without mixing semantics
SourceSpace: TypeAlias = OrientationCode | Literal["FILE_NATIVE"]


class ImagingModel(BaseModel):
    model_config = {"extra": "forbid"}

    magnet_frequency_MHz: float
    chem_shift_ppm_default: float = 3.7
    # optionally, where to read the reference image from if needed by your library
    image_path: Optional[FilePath] = None


# A probe with no recording array — a pipette — targets with its tip. Said out
# loud so a mistyped kind cannot quietly mean the same thing.
NO_RECORDING_ARRAY = "none"


class RecordingModel(BaseModel):
    """Where a probe's electrodes are, along each shank.

    Distances are mm from the shank tip along the shank axis, one
    ``(start, end)`` per shank the optimizer sums coverage across; shank order
    matches ``domain.pose.detect_shank_tips_local``. Declaring this is what
    lets a subject use a probe the built-in table has never heard of.
    """

    model_config = {"extra": "forbid"}

    active_ranges_mm: list[tuple[float, float]] = Field(..., min_length=1)
    shank_pitch_mm: float = 0.25  # informational; the pivot comes from the mesh

    @field_validator("active_ranges_mm")
    @classmethod
    def _ranges_ascend(cls, v):
        for start, end in v:
            if not end > start:
                raise ValueError(f"active range ({start}, {end}) must have end > start")
        return v


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
    role: Optional[Role] = None

    # Explicit points (file)
    src: Optional[Path] = None
    loader: Optional[str] = None  # e.g., "numpy_points"
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)

    canonicalization_ref: Optional[str] = None
    canonicalization: Optional[CanonicalizationDefModel] = (
        None  # inline (legacy/one-off)
    )
    canonicalization_override: Optional[CanonicalizationOverrideModel] = None

    mr_signal: Optional[MRSignal] = None
    # Unset resolves from the built-in table by kind; "none" declares a
    # probe with no recording array.
    recording: Union[RecordingModel, Literal["none"], None] = None
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

    tags: list[str] = Field(default_factory=list)

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


class TxOpBase(BaseModel):
    model_config = {"extra": "forbid"}

    invert: bool = False


class TranslateTxOpModel(TxOpBase):
    kind: Literal["translate_mm"] = "translate_mm"
    delta: list[float] = Field(..., min_length=3, max_length=3)


class RotateEulerTxOpModel(TxOpBase):
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


class LoadSITKTxOpModel(TxOpBase):
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
        if isinstance(v, TxOpBase):
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
# Transforms, Paths, Options
# -----------------------------------------------------------------------------


class PathsModel(BaseModel):
    """Freeform helper; keep loose so Hydra/OmegaConf interpolation is easy."""

    model_config = {"extra": "allow"}

    def __init__(self, **data):
        super().__init__(**data)
