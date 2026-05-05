"""CanonicalizationRuntime and helpers to apply orientation/scale/transform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from aind_mri_utils.rotations import apply_rotate_translate

from aind_low_point.config import (
    BaseSpecModel,
    CanonicalizationDefModel,
    ConfigModel,
    ResourceModel,
    SourceSpace,
    TransformRefModel,
)
from aind_low_point.core import AffineTransform, TransformChain
from aind_low_point.orientation_codes import OrientationCode
from aind_low_point.runtime.transforms import (
    CompiledTransforms,
    compile_recipe_to_chain,
    resolve_transform_key_cached,
    resolve_transform_ref_cached,
)


@dataclass(frozen=True)
class CanonicalizationRuntime:
    source_space: SourceSpace = OrientationCode.LPS  # e.g. "LPS", "RAS", "ASR"
    scale_to_mm: float = 1.0  # e.g. 0.001 if µm → mm
    transform_file_to_canonical: Optional[AffineTransform] = None


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
