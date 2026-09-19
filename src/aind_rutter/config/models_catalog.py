"""Catalog specs: what an asset or target is, in single and bulk form."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from aind_rutter.config.models_common import (
    CanonicalizationDefModel,
    CanonicalizationOverrideModel,
    GeometrySourceModel,
    MaterialModel,
    RecordingModel,
    Selector,
    TransformRefModel,
)
from aind_rutter.domain.enums import Kind, MRSignal, Role

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

# Only for configs that predate stating `role`. Declaring it is what lets a
# subject name a mesh anything it likes.
ROLE_PREFIX_DEFAULTS: list[tuple[str, Role]] = [
    ("structure:", Role.ANATOMY),
    ("brain", Role.ANATOMY),
    ("target:", Role.TARGET),
    ("landmark:", Role.LANDMARK),
    ("probe:", Role.PROBE),
]


def infer_from_extension(spec: "AssetSpecModel | TargetSpecModel") -> None:
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


def infer_role_from_key(spec: "AssetSpecModel | TargetSpecModel") -> None:
    """Mutate spec to fill in role from key prefix if not set."""
    if spec.role is not None or spec.key is None:
        return
    for prefix, role in ROLE_PREFIX_DEFAULTS:
        if spec.key.startswith(prefix):
            object.__setattr__(spec, "role", role)
            return
    object.__setattr__(spec, "role", Role.GEOMETRY)  # default


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

    collidable: Optional[bool] = None

    pivot_LPS: Optional[list[float]] = None
    bbox_hint: Optional[list[list[float]]] = None

    # Chem-shift hints (optional, ignored if not applicable)
    chem_shift_ppm: Optional[float] = None
    mr_signal: Optional[MRSignal] = None
    # Unset resolves from the built-in table by kind; "none" declares a
    # probe with no recording array.
    recording: Union[RecordingModel, Literal["none"], None] = None


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

    # Whether this asset gets an FCL body.
    collidable: Optional[bool] = None

    # UI/layout hints
    pivot_LPS: Optional[list[float]] = Field(default=None, min_length=3, max_length=3)
    bbox_hint: Optional[list[list[float]]] = Field(default=None)

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


def passthrough_kwargs(
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

    # Whether this asset gets an FCL body.
    collidable: Optional[bool] = None

    pivot_LPS: Optional[list[float]] = Field(default=None, min_length=3, max_length=3)
    bbox_hint: Optional[list[list[float]]] = Field(default=None)

    mr_signal: Optional[MRSignal] = None
    # Unset resolves from the built-in table by kind; "none" declares a
    # probe with no recording array.
    recording: Union[RecordingModel, Literal["none"], None] = None
    chem_shift_ppm: Optional[float] = None

    templates: list[str] = Field(default_factory=list)
    from_resource: Optional[str] = None
    selector: Optional[Selector] = None

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
            kwargs = passthrough_kwargs(self, {"keys"}, overrides)
            results.append(AssetSpecModel(**kwargs))
        return results


class AtlasMeshPackSpecModel(BaseModel):
    """Bulk asset declaration for an atlas mesh pack keyed by region acronym.

    Resolves CCF region acronyms via the bundled ontology and emits one
    AssetSpecModel per acronym, with src pointing at
    ``<atlas_dir>/<id><file_extension>``. The atlas itself is expected to
    contain one mesh per CCF integer label (e.g. brainglobe's
    ``allen_mouse_25um`` directory of ``<id>.obj`` files).

    Example::

        - atlas_dir: ${paths.atlas_dir}
          acronyms: [VISp, MOs, CA1, BLA]
          key_prefix: atlas
          canonicalization_ref: atlas-template-lps
          material_ref: structure
    """

    model_config = {"extra": "forbid"}

    atlas_dir: Path
    acronyms: list[str] = Field(..., min_length=1)
    key_prefix: str = "atlas"
    file_extension: str = ".obj"
    # When True, colour each expanded asset by the CCF region's bundled
    # color_hex. Overrides ``material.color`` for the expanded specs only;
    # other material fields (opacity, point_size, …) flow through from
    # ``material_ref`` / ``material`` as usual. If the user explicitly
    # sets a color in ``material:`` on the pack, that wins (explicit beats
    # implicit).
    use_ccf_color: bool = False

    # Same downstream fields as BulkAssetSpecModel (kind/loader forced in expand)
    role: Optional[Role] = Role.ANATOMY
    material_ref: Optional[str] = None
    material: Optional[MaterialModel] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    transform: Optional["TransformRefModel"] = None
    scene_tags: list[str] = Field(default_factory=list)
    auto_scene: bool = True

    loader_kwargs: dict[str, Any] = Field(default_factory=dict)

    canonicalization_ref: Optional[str] = None
    canonicalization: Optional[CanonicalizationDefModel] = None
    canonicalization_override: Optional[CanonicalizationOverrideModel] = None

    # Whether this asset gets an FCL body.
    collidable: Optional[bool] = None

    pivot_LPS: Optional[list[float]] = Field(default=None, min_length=3, max_length=3)
    bbox_hint: Optional[list[list[float]]] = Field(default=None)

    mr_signal: Optional[MRSignal] = None
    # Unset resolves from the built-in table by kind; "none" declares a
    # probe with no recording array.
    recording: Union[RecordingModel, Literal["none"], None] = None
    chem_shift_ppm: Optional[float] = None

    templates: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_acronyms(self) -> "AtlasMeshPackSpecModel":
        from aind_rutter.ccf_ontology import CCFOntology

        ontology = CCFOntology.from_bundled()
        unknown = [a for a in self.acronyms if ontology.find_by_acronym(a) is None]
        if unknown:
            raise ValueError(
                f"Unknown CCF acronym(s): {unknown}. "
                "Acronym match is case-sensitive; verify against the Allen CCF."
            )
        return self

    def expand(self) -> list[AssetSpecModel]:
        """Expand into individual AssetSpecModel instances, one per acronym."""
        from aind_rutter.ccf_ontology import CCFOntology

        ontology = CCFOntology.from_bundled()
        results: list[AssetSpecModel] = []
        for acronym in self.acronyms:
            structure = ontology.find_by_acronym(acronym)
            assert structure is not None  # validated in _validate_acronyms

            key = f"{self.key_prefix}:{acronym}"
            src = self.atlas_dir / f"{structure.id}{self.file_extension}"

            # Carry CCF identity in the spec metadata so downstream code
            # (e.g. probe-recolouring by target) can recover the region
            # without having to parse the asset key.
            base_metadata = dict(self.metadata)
            base_metadata.setdefault("ccf_id", structure.id)
            base_metadata.setdefault("ccf_acronym", structure.acronym)
            overrides: dict[str, Any] = {
                "key": key,
                "src": src,
                "kind": Kind.MESH,
                "loader": "trimesh",
                # Always forward role so the field default (Role.ANATOMY)
                # survives into the expanded spec; user-supplied role wins
                # because self.role reflects either the YAML or the default.
                "role": self.role,
                "metadata": base_metadata,
            }
            if self.use_ccf_color:
                # Build a per-region material that injects the CCF color.
                # Start from the pack's material (if any) so opacity etc.
                # carry through; only fill in color when the user didn't
                # set it explicitly.
                base_material = (
                    self.material.model_dump(exclude_unset=True)
                    if self.material is not None
                    else {}
                )
                if "color" not in base_material:
                    base_material["color"] = structure.color_hex
                overrides["material"] = MaterialModel(**base_material)
            kwargs = passthrough_kwargs(
                self,
                {
                    "acronyms",
                    "atlas_dir",
                    "key_prefix",
                    "file_extension",
                    "use_ccf_color",
                },
                overrides,
            )
            results.append(AssetSpecModel(**kwargs))
        return results


# Type alias for assets: single, bulk-by-keys, or atlas-by-acronym
AssetSpecUnion = Union[AssetSpecModel, BulkAssetSpecModel, AtlasMeshPackSpecModel]


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
    def _check_target_source(self):
        explicit = self.src is not None and self.loader is not None
        derived = self.source_key is not None
        from_res = (self.from_resource is not None) and (self.selector is not None)
        paths = sum([explicit, derived, from_res])
        if paths != 1:
            raise ValueError(
                f"Target '{self.key}': provide exactly one of "
                "(src+loader) | (source_key+reducer) | (from_resource+selector)"
            )
        if self.collidable:
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

    # Whether this asset gets an FCL body.
    collidable: Optional[bool] = None

    pivot_LPS: Optional[list[float]] = Field(default=None, min_length=3, max_length=3)
    bbox_hint: Optional[list[list[float]]] = Field(default=None)

    mr_signal: Optional[MRSignal] = None
    # Unset resolves from the built-in table by kind; "none" declares a
    # probe with no recording array.
    recording: Union[RecordingModel, Literal["none"], None] = None
    chem_shift_ppm: Optional[float] = None

    templates: list[str] = Field(default_factory=list)

    approach_vector: Optional[list[float]] = Field(
        default=None, min_length=3, max_length=3
    )
    uncertainty_mm: Optional[float] = None

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
            kwargs = passthrough_kwargs(self, {"key_pattern", "range"}, overrides)
            results.append(TargetSpecModel(**kwargs))
        return results


class DerivedTargetSpecModel(BaseModel):
    """Bulk target declaration deriving targets from existing assets.

    ``derive_from`` accepts either an explicit list of asset keys or a
    single fnmatch-style glob string (resolved against the catalog
    AFTER bulk-asset expansion). The glob form lets a single source of
    truth — e.g. an AtlasMeshPack's ``acronyms:`` list — drive both
    assets and targets without manual duplication.

    Example::

        # Explicit:
        - derive_from: [structure:PL, structure:MD, structure:CLA]
          key_prefix: "target:"
          templates: [structure]

        # Glob — matches whatever assets exist after expansion:
        - derive_from: "structure:*"
          key_prefix: "target:"
          templates: [structure]
    """

    model_config = {"extra": "forbid"}

    # Either an explicit list of asset keys or a single fnmatch pattern
    # string. Pattern strings are resolved against the (already-expanded)
    # asset list during ConfigModel validation.
    derive_from: Union[list[str], str] = Field(..., min_length=1)
    key_prefix: str = "target:"  # prepended to derive target key

    # When set, each derived target re-loads its source asset's region
    # restricted to one hemisphere ("left"/"right") at LOAD time, instead
    # of reusing the source asset's (bilateral) mesh. Requires the source
    # asset to load a lateralized annotation (left = negated label id);
    # see ``ccf_annotation_region``. The hemisphere is injected into the
    # source asset's ``loader_kwargs`` and the region is reduced normally
    # (e.g. ``mesh_center_mass``).
    hemisphere: Optional[str] = None

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

    # Whether this asset gets an FCL body.
    collidable: Optional[bool] = None

    pivot_LPS: Optional[list[float]] = Field(default=None, min_length=3, max_length=3)
    bbox_hint: Optional[list[list[float]]] = Field(default=None)

    mr_signal: Optional[MRSignal] = None
    # Unset resolves from the built-in table by kind; "none" declares a
    # probe with no recording array.
    recording: Union[RecordingModel, Literal["none"], None] = None
    chem_shift_ppm: Optional[float] = None

    templates: list[str] = Field(default_factory=list)

    approach_vector: Optional[list[float]] = Field(
        default=None, min_length=3, max_length=3
    )
    uncertainty_mm: Optional[float] = None

    def expand(
        self, asset_by_key: Optional[dict[str, "AssetSpecModel"]] = None
    ) -> list[TargetSpecModel]:
        """Expand into individual TargetSpecModel instances.

        ``asset_by_key`` (the expanded asset roster) is only required when
        ``hemisphere`` is set: each derived target then re-loads its source
        asset's region for one hemisphere rather than reusing the source's
        bilateral mesh.
        """
        results = []
        for asset_key in self.derive_from:
            suffix = asset_key.split(":")[-1] if ":" in asset_key else asset_key
            overrides: dict[str, Any] = {"key": f"{self.key_prefix}{suffix}"}
            if self.hemisphere is not None:
                src_asset = (asset_by_key or {}).get(asset_key)
                if src_asset is None:
                    raise ValueError(
                        f"derive_from target '{self.key_prefix}{suffix}' with "
                        f"hemisphere set requires source asset '{asset_key}' to "
                        f"be defined with a loader"
                    )
                self._wire_hemisphere_overrides(
                    asset_key, src_asset, asset_by_key or {}, overrides
                )
            else:
                overrides["source_key"] = asset_key
            kwargs = passthrough_kwargs(
                self, {"derive_from", "key_prefix", "hemisphere"}, overrides
            )
            results.append(TargetSpecModel(**kwargs))
        return results

    def _wire_hemisphere_overrides(
        self,
        asset_key: str,
        src_asset: "AssetSpecModel",
        asset_by_key: dict[str, "AssetSpecModel"],
        overrides: dict[str, Any],
    ) -> None:
        """Fill ``overrides`` so a hemisphere-set derived target selects its
        region by **voxel label** (the canonical mechanism).

        Branches on ``self.reducer`` — the differentiator already present on the
        spec, so no extra fields are needed:

        - ``points_in_region_center_mass`` (retro): keep the reducer; the target
          renders the source structure mesh (``source_key``) but selection is by
          voxel label, so inject the source asset's annotation path + region spec
          (and hemisphere) into ``reducer_kwargs``. The ``points_key`` retro cloud
          must share the annotation's frame, so guard it for identity
          canonicalization.
        - ``points_mean`` (anatomical): load the region's voxel cloud directly via
          ``ccf_region_voxel_points`` and take its mean.
        - otherwise (e.g. ``mesh_center_mass``): legacy behaviour — re-load the
          source asset's region restricted to the hemisphere as a mesh.
        """
        # The region spec lives in the source asset's loader_kwargs.
        src_lk = dict(src_asset.loader_kwargs or {})
        region = {
            k: src_lk[k]
            for k in ("acronym", "label_id", "include_descendants")
            if k in src_lk
        }
        src_str = str(src_asset.src) if src_asset.src is not None else None

        if self.reducer == "points_in_region_center_mass":
            # Structure mesh renders, but the reducer ignores it (label-based).
            overrides["source_key"] = asset_key
            rk = dict(self.reducer_kwargs or {})
            self._guard_retro_points_frame(rk.get("points_key"), asset_by_key)
            rk["annotation_path"] = src_str
            rk.update(region)
            rk["hemisphere"] = self.hemisphere
            overrides["reducer_kwargs"] = rk
        elif self.reducer == "points_mean":
            overrides["src"] = src_str
            overrides["loader"] = "ccf_region_voxel_points"
            lk = dict(region)
            lk["hemisphere"] = self.hemisphere
            overrides["loader_kwargs"] = lk
        else:
            overrides["src"] = src_str
            overrides["loader"] = src_asset.loader
            lk = dict(src_lk)
            lk["hemisphere"] = self.hemisphere
            overrides["loader_kwargs"] = lk

    @staticmethod
    def _guard_retro_points_frame(
        points_key: Optional[str],
        asset_by_key: dict[str, "AssetSpecModel"],
    ) -> None:
        """Require the retro ``points_key`` asset to be in the annotation frame.

        ``points_in_region_center_mass`` looks the retro cloud up against the
        natively-read annotation volume, which is only valid when the retro asset
        is in that same physical frame — i.e. it carries no canonicalization. A
        non-identity canonicalization would silently shift the points off the
        volume and mis-select; catch it here with a clear error rather than
        relying on the membership happening to come out empty.
        """
        if points_key is None:
            return
        pts_asset = asset_by_key.get(points_key)
        if pts_asset is None:
            return  # cross-reference validation elsewhere reports a missing key
        if (
            pts_asset.canonicalization_ref is not None
            or pts_asset.canonicalization is not None
        ):
            raise ValueError(
                f"retro points asset '{points_key}' must be in the annotation "
                "frame (identity canonicalization) for voxel-label selection, "
                f"but has canonicalization "
                f"{pts_asset.canonicalization_ref or pts_asset.canonicalization!r}"
            )


# Type alias for targets that can be single, range, or derived
TargetSpecUnion = Union[TargetSpecModel, RangeTargetSpecModel, DerivedTargetSpecModel]
