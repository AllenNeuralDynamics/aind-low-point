"""Catalog of what exists"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Literal,
    Optional,
    Union,
)

from aind_low_point.common import Capability, Role
from aind_low_point.core import (
    Float3,
    FloatAABB,
    Material,
    MeshTransformable,
    PointsTransformable,
)


@dataclass(frozen=True)
class BaseSpec:
    # WHAT it is
    key: str  # unique id, e.g. "probe:2.1", "structure:PL", "target:hole:1"
    kind: Literal["mesh", "points", "lines"]
    role: Role = Role.GEOMETRY
    default_material: Material = field(default_factory=lambda: Material("default"))
    metadata: dict[str, Any] = field(default_factory=dict)
    # free-form (scene/UI grouping)
    tags: set[str] = field(default_factory=set)

    # HOW it behaves (capabilities & collision policy)
    caps: Capability = Capability.RENDERABLE
    collidable_group: int = 0  # label-compiled group bit (0 = none)
    collidable_mask: int = 0  # set of groups it can collide with (bitmask)

    # Optional quick UI/layout hints (applies to meshes/points; ignored
    # otherwise)
    # rotation center in canonical local asset space
    pivot_LPS: Optional[Float3] = None
    bbox_hint: Optional[FloatAABB] = (
        None  # AABB (2×3) or sphere radius (use metadata if preferred)
    )

    # NOTE: BaseSpec does NOT carry concrete geometry; subclasses do.


# ---------------------------------------------------------------------------
# AssetSpec: concrete geometry (catalog items)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AssetSpec(BaseSpec):
    # SOURCE (how to load the asset)
    source_path: Optional[Path] = None
    loader: Optional[str] = (
        None  # name of a registered loader (e.g. "trimesh", "sitk_volume")
    )

    # CANONICAL GEOMETRY (post-load, guaranteed in canonical LPS mm)
    mesh: Optional[MeshTransformable] = None
    points: Optional[PointsTransformable] = None
    # (lines, volume, etc. could be added later)

    def __post_init__(self):
        # A few light invariants to catch common mistakes
        if self.kind == "mesh" and self.mesh is None and self.points is not None:
            raise ValueError(f"{self.key}: kind='mesh' but only points were provided")
        if self.kind == "points" and self.points is None and self.mesh is not None:
            raise ValueError(f"{self.key}: kind='points' but only mesh was provided")
        if self.role != Role.GEOMETRY and self.caps & Capability.COLLIDABLE:
            # Non-geometry roles default to non-collidable unless explicitly chosen
            object.__setattr__(self, "collidable_mask", 0)
            object.__setattr__(self, "collidable_group", 0)


# ---------------------------------------------------------------------------
# TargetSpec: logical targets (derived or explicit points)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetSpec(BaseSpec):
    # For targets we default to points, role TARGET, and non-collidable caps
    kind: Literal["points", "derived_point"] = "points"
    role: Role = Role.TARGET
    caps: Capability = Capability.RENDERABLE

    # SOURCE: either load explicit points, or derive from another asset
    # - If 'source_path' + 'loader' given → explicit points (like AssetSpec points)
    # - If 'source_key' given → derive from another AssetSpec
    # already in catalog
    source_path: Optional[Path] = None
    loader: Optional[str] = None  # e.g. "numpy_points"
    source_key: Optional[str] = None  # e.g. "structure:PL"

    # reducer to reduce loaded or derived geometry
    reducer: Optional[str] = None  # registered reducer name
    reducer_kwargs: dict[str, Any] = field(default_factory=dict)

    # DERIVED/LOADED canonical points
    points: Optional["PointsTransformable"] = None

    # Hints useful for planning/visualization (targets are often landmarks)
    approach_vector: Optional[Float3] = None  # preferred insertion direction (LPS)
    uncertainty_mm: Optional[float] = (
        None  # radius for UI (confidence, snap tolerance, etc.)
    )

    def __post_init__(self):
        # Enforce typical non-collidable defaults for targets
        if self.caps & Capability.COLLIDABLE:
            raise ValueError(f"{self.key}: targets should not be collidable by default")
        # Require either explicit points (source_path+loader) or derived
        # (source_key+reducer)
        explicit = self.source_path is not None and self.loader is not None
        derived = self.source_key is not None
        if not explicit and not derived and self.points is None:
            raise ValueError(
                f"{self.key}: must provide explicit points or a source_key"
            )


@dataclass(frozen=True, slots=True)
class AssetCatalog:
    assets: dict[str, AssetSpec]  # asset catalog
    targets: dict[str, TargetSpec] = field(default_factory=dict)

    def has_key(self, key: str) -> bool:
        return key in self.assets or key in self.targets

    def get_spec(self, key: str) -> Union[AssetSpec, TargetSpec]:
        if key in self.assets:
            return self.assets[key]
        if key in self.targets:
            return self.targets[key]
        raise KeyError(f"CatalogRuntime: unknown key '{key}'")

    def get_geometry(self, key: str) -> Union[MeshTransformable, PointsTransformable]:
        """
        Returns (kind, geometry) where kind is 'mesh'|'points'|'lines'.
        Targets always return 'points'.
        """
        if key in self.assets:
            spec = self.assets[key]
            if spec.mesh is not None:
                return spec.mesh
            if spec.points is not None:
                return spec.points
            raise ValueError(f"{key}: asset has no concrete geometry loaded")
        if key in self.targets:
            spec = self.targets[key]
            if spec.points is not None:
                return spec.points
            raise ValueError(f"{key}: target has no points (was it resolved?)")
        raise KeyError(f"CatalogRuntime: unknown key '{key}'")
