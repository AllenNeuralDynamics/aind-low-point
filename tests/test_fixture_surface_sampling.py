"""Surface samples for clearance queries are reproducible across processes."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from aind_rutter.optimization.pipeline import phase1_geometry
from aind_rutter.optimization.sdf import build as sdf_build
from aind_rutter.optimization.sdf import envelope


def test_probe_sdf_surface_points_are_seeded() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 3.0, 4.0))
    kwargs = {
        "spacing_mm": 0.5,
        "n_surface_points": 500,
        "use_cache": False,
        "sign_type": "pseudonormal",
    }
    a = sdf_build.build_probe_sdf(mesh, **kwargs)
    b = sdf_build.build_probe_sdf(mesh, **kwargs)
    np.testing.assert_array_equal(a.surface_points, b.surface_points)


def test_cropped_envelope_surface_is_seeded_and_cached(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIND_LOW_POINT_CACHE_DIR", str(tmp_path))
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    monkeypatch.setattr(
        envelope, "build_alpha_wrap_envelope", lambda mesh, **_: box, raising=True
    )
    kwargs = {
        "offset_mm": 0.15,
        "box_min": np.array([-5.0, -5.0, 0.0]),
        "box_max": np.array([5.0, 5.0, 5.0]),
        "n": 400,
    }
    first = phase1_geometry._resample_envelope_surface_in_box(box, **kwargs)
    assert first.shape == (400, 3)
    assert (first[:, 2] >= 0.0).all()
    assert list(tmp_path.rglob("*.npz"))

    def no_sampling(*_args, **_kwargs):
        raise AssertionError("a cached crop must not resample")

    monkeypatch.setattr(trimesh.sample, "sample_surface", no_sampling)
    np.testing.assert_array_equal(
        phase1_geometry._resample_envelope_surface_in_box(box, **kwargs), first
    )

    # A fresh cache reproduces the same draw from the seed alone.
    monkeypatch.undo()
    monkeypatch.setenv("AIND_LOW_POINT_CACHE_DIR", str(tmp_path / "fresh"))
    monkeypatch.setattr(envelope, "build_alpha_wrap_envelope", lambda mesh, **_: box)
    np.testing.assert_array_equal(
        phase1_geometry._resample_envelope_surface_in_box(box, **kwargs), first
    )


@pytest.mark.parametrize("n", [10, 50])
def test_cropped_envelope_surface_cache_key_includes_count(
    tmp_path, monkeypatch, n
) -> None:
    monkeypatch.setenv("AIND_LOW_POINT_CACHE_DIR", str(tmp_path))
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    monkeypatch.setattr(envelope, "build_alpha_wrap_envelope", lambda mesh, **_: box)
    out = phase1_geometry._resample_envelope_surface_in_box(
        box, offset_mm=0.15, box_min=np.full(3, -5.0), box_max=np.full(3, 5.0), n=n
    )
    assert out.shape == (n, 3)
