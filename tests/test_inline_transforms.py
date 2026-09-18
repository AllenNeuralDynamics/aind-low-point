"""A transform written inline must resolve to the same type as a named one.

`resolve_transform_ref_cached` returned an `AffineTransform` for a transform
referenced by key and a bare `(R, t)` tuple for one written inline, so every
caller that read `.rotate_translate` off the result raised `AttributeError` the
moment a config used the inline form. A `type: ignore` on the return sat exactly
where the mismatch was.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from aind_rutter.config import ConfigModel, TransformRefModel
from aind_rutter.core import AffineTransform
from aind_rutter.runtime.build import build_runtime_from_config
from aind_rutter.runtime.transforms import (
    compile_all_transforms,
    resolve_transform_ref_cached,
)
from tests.synthetic_subject import write_subject

SHIFT_MM = (1.0, -2.0, 0.5)
INLINE_SEQUENCE = [{"kind": "translate_mm", "delta": list(SHIFT_MM)}]


def _inline_ref() -> TransformRefModel:
    return TransformRefModel.model_validate(INLINE_SEQUENCE)


def test_an_inline_reference_resolves_to_an_affine_transform() -> None:
    resolved = resolve_transform_ref_cached(_inline_ref(), {})
    assert isinstance(resolved, AffineTransform)
    rotation, translation = resolved.rotate_translate
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(translation, SHIFT_MM)


def test_both_reference_forms_resolve_to_the_same_transform() -> None:
    """The key and the inline recipe describe one transform; so must the result."""
    cache = compile_all_transforms(
        ConfigModel.model_validate(
            {"transforms": {"shift": {"sequence": INLINE_SEQUENCE}}}
        ).transforms
    )
    by_key = resolve_transform_ref_cached(TransformRefModel(key="shift"), cache)
    inline = resolve_transform_ref_cached(_inline_ref(), cache)
    assert type(by_key) is type(inline)
    for left, right in zip(by_key.rotate_translate, inline.rotate_translate):
        np.testing.assert_allclose(left, right)


def test_a_missing_reference_stays_none() -> None:
    assert resolve_transform_ref_cached(None, {}) is None


def test_an_unknown_key_says_which_one() -> None:
    with pytest.raises(KeyError, match="absent"):
        resolve_transform_ref_cached(TransformRefModel(key="absent"), {})


def test_a_config_with_an_inline_canonicalization_builds(tmp_path) -> None:
    """The reported crash: this raised AttributeError while loading the mesh."""
    subject = write_subject(tmp_path)
    document = yaml.safe_load(subject.config.read_text())
    document["canonicalizations"] = {
        "file-native": {
            "source_space": "FILE_NATIVE",
            "transform": INLINE_SEQUENCE,
        }
    }
    for asset in document["assets"]:
        if asset["key"] == "well":
            asset["canonicalization_ref"] = "file-native"
    path = tmp_path / "inline-canonicalization.yml"
    path.write_text(yaml.safe_dump(document))

    plain = build_runtime_from_config(ConfigModel.from_yaml(subject.config))
    canonicalized = build_runtime_from_config(ConfigModel.from_yaml(path))

    before = np.asarray(plain.asset_catalog.assets["well"].mesh.raw.vertices)
    after = np.asarray(canonicalized.asset_catalog.assets["well"].mesh.raw.vertices)
    np.testing.assert_allclose(
        after - before, np.broadcast_to(SHIFT_MM, before.shape), atol=1e-9
    )
