"""Compile config transforms into runtime AffineTransforms / TransformChains."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from aind_low_point.config import (
    LoadSITKTxOpModel,
    RotateEulerTxOpModel,
    TransformRecipeModel,
    TransformRefModel,
    TranslateTxOpModel,
    _TxOpBase,
)
from aind_low_point.core import AffineTransform, TransformChain

CompiledTransforms = dict[str, AffineTransform]


def _op_to_affine(op: _TxOpBase) -> AffineTransform:
    if isinstance(op, TranslateTxOpModel):
        R = np.eye(3)
        t = np.asarray(op.delta, dtype=np.float64)
        return AffineTransform(R, t)
    elif isinstance(op, RotateEulerTxOpModel):
        rotation = Rotation.from_euler(op.order, op.angles_deg, degrees=True)
        R = rotation.as_matrix()
        return AffineTransform(R, np.zeros(3))
    elif isinstance(op, LoadSITKTxOpModel):
        from aind_mri_utils.file_io.simpleitk import load_sitk_transform

        R, t, _ = load_sitk_transform(op.path)
        return AffineTransform(R, t, op.inverted)
    else:
        raise TypeError(f"Unsupported op {type(op)}")


def compile_recipe_to_chain(recipe: TransformRecipeModel) -> TransformChain:
    """Compile a recipe into a TransformChain.

    Individual ops stay as separate AffineTransforms (nice for debugging);
    callers can collapse to (R, t) via ``chain.composed_transform`` when
    needed.
    """
    affines = [_op_to_affine(op) for op in recipe.sequence if op]
    return TransformChain.new(affines)


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
