"""Building blocks of different parts of the run time"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Generic,
    Iterable,
    Protocol,
    Tuple,
    TypeAlias,
    TypeVar,
    overload,
    runtime_checkable,
)

import numpy as np
import trimesh
from aind_mri_utils.file_io.simpleitk import load_sitk_transform
from aind_mri_utils.rotations import (
    apply_rotate_translate,
    compose_transforms,
    invert_rotate_translate,
)
from numpy.typing import NDArray

Float3x3 = NDArray[np.float64]  # shape (3, 3)
Float3 = NDArray[np.float64]  # shape (3,)
FloatNx3 = NDArray[np.float64]  # shape (3, N)
FloatAABB = NDArray[np.float64]  # shape (2, 3)

RawT_co = TypeVar("RawT_co", covariant=True)
Pair = Tuple[str, str]


@dataclass(frozen=True)
class AffineTransform:
    rotation: Float3x3 = field(default_factory=lambda: np.eye(3), repr=False)
    translation: Float3 = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0]), repr=False
    )
    inverted: bool = False

    @classmethod
    def identity(cls) -> AffineTransform:
        return cls()

    @classmethod
    def from_sitk_path(cls, path: Path, inverted=False) -> AffineTransform:
        R, t, _ = load_sitk_transform(str(path))
        return cls(rotation=R, translation=t, inverted=inverted)

    @cached_property
    def rotate_translate(self) -> Tuple[Float3x3, Float3]:
        if self.inverted:
            R, t = invert_rotate_translate(self.rotation, self.translation)
        else:
            R, t = self.rotation, self.translation
        return R, t

    def apply_to(self, pts: FloatNx3) -> FloatNx3:
        """Apply the transform to a set of points."""
        R, t = self.rotate_translate
        return apply_rotate_translate(pts, R, t)

    def invert(self) -> "AffineTransform":
        """Invert the transform."""
        return AffineTransform(
            rotation=np.copy(self.rotation),
            translation=np.copy(self.translation),
            inverted=not self.inverted,
        )


@dataclass(frozen=True)
class TransformChain:
    elements: Tuple[AffineTransform, ...]

    def __post_init__(self):
        # Allow users to pass a list; store as an immutable tuple.
        if not isinstance(self.elements, tuple):
            object.__setattr__(self, "elements", tuple(self.elements))

    @cached_property
    def composed_transform(self) -> Tuple[NDArray, NDArray]:
        """Get the combined rotation and translation from all transforms in the
        chain."""
        pairs = []
        for e in self.elements:
            R, t = e.rotate_translate
            pairs.append(R)
            pairs.append(t)
        return compose_transforms(*pairs)

    def apply_to(self, pts: NDArray) -> NDArray:
        """Apply the transform chain to a set of points."""
        R, t = self.composed_transform
        return apply_rotate_translate(pts, R, t)

    def invert(self) -> "TransformChain":
        """Invert the transform chain."""
        return TransformChain(tuple(e.invert() for e in reversed(self.elements)))

    @classmethod
    def new(cls, items: Iterable[AffineTransform]) -> "TransformChain":
        # Convenience constructor to make intent explicit.
        return cls(tuple(items))


@runtime_checkable
class SupportsRigidTransform(Protocol[RawT_co]):
    @property
    def raw(self) -> RawT_co: ...
    def transformed(
        self, R: NDArray[np.float64], t: NDArray[np.float64]
    ) -> RawT_co: ...


W = TypeVar("W", bound=SupportsRigidTransform[Any])


@dataclass(frozen=True, slots=True)
class MeshTransformable(SupportsRigidTransform[trimesh.Trimesh]):
    _raw: trimesh.Trimesh

    @property
    def raw(self) -> trimesh.Trimesh:
        return self._raw

    def transformed(self, R: Float3x3, t: Float3) -> trimesh.Trimesh:
        v = apply_rotate_translate(self._raw.vertices, R, t)
        return trimesh.Trimesh(vertices=v, faces=self._raw.faces, process=False)


@dataclass(frozen=True, slots=True)
class PointsTransformable(SupportsRigidTransform[FloatNx3]):
    _raw: FloatNx3  # (N,3) float

    def __post_init__(self) -> None:
        a = self._raw
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError("PointsWrap expects shape (N,3)")
        if a.dtype.kind != "f":
            raise TypeError("PointsWrap expects float dtype")

    @property
    def raw(self) -> FloatNx3:
        return self._raw

    def transformed(self, R: Float3x3, t: Float3) -> FloatNx3:
        return apply_rotate_translate(self._raw, R, t)


@overload
def as_transformable(x: trimesh.Trimesh) -> MeshTransformable: ...
@overload
def as_transformable(x: FloatNx3) -> PointsTransformable: ...
def as_transformable(x):
    if isinstance(x, MeshTransformable) or isinstance(x, PointsTransformable):
        return x
    if isinstance(x, trimesh.Trimesh):
        return MeshTransformable(x)
    if isinstance(x, np.ndarray):
        return PointsTransformable(x)
    raise TypeError(f"Unsupported type {type(x)}")


@dataclass(frozen=True)
class Transformed(Generic[W, RawT_co]):
    original: W
    chain: TransformChain

    def __post_init__(self) -> None:
        # Runtime structural check (optional but nice)
        if not isinstance(self.original, SupportsRigidTransform):
            raise TypeError(
                "`original` must implement SupportsRigidTransform; got "
                f"{type(self.original).__name__}"
            )

    @cached_property
    def raw(self) -> RawT_co:
        R, t = self.chain.composed_transform
        return self.original.transformed(R, t)

    # If you often need the untransformed payload too:
    @property
    def original_raw(self) -> RawT_co:
        return self.original.raw


TransformedMesh: TypeAlias = Transformed[MeshTransformable, trimesh.Trimesh]
TransformedPoints: TypeAlias = Transformed[PointsTransformable, FloatNx3]


@dataclass(frozen=True, slots=True)
class Material:
    name: str
    color_hex_str: str = "#C8C8C8"
    opacity: float = 1.0
    wireframe: bool = False
    visible: bool = True
    point_size: float = 5.0

    def replace(self, **kw) -> "Material":
        """Return a new Material with the given fields overridden."""
        return dc_replace(self, **kw)
