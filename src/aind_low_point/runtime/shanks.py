"""Detect shank-tip positions in a probe mesh's local frame.

Used by the over-insertion overlay: if any shank's tip in world coords
has 2+ ``brain`` mesh intersections along the +probe-z ray, the probe
has gone through to the underside of the brain. Multi-shank probes
(NP 2.0 four-shank, quadbase) need this run per shank — checking only
shank 0 misses the case where one outer shank pokes out the back.

The detector finds vertices at the probe's local ``min-z`` plane (where
the tips live for any "centeredOn" / canonical-LPS probe mesh) and
clusters them by ``xy`` using single-link agglomerative clustering. Each
cluster's centroid is one shank's tip in local coords.

Caller is expected to log the detected count once per probe asset so a
mis-detection on an exotic mesh is visible immediately rather than
producing silent false-positive warnings later.
"""

from __future__ import annotations

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fclusterdata


def detect_shank_tips_local(
    mesh: trimesh.Trimesh,
    *,
    z_tolerance_mm: float = 0.05,
    cluster_radius_mm: float = 0.15,
) -> NDArray[np.float64]:
    """Return shank-tip positions in the probe's local frame as ``(N, 3)``.

    The probe mesh is assumed to be canonicalized so the shaft runs in
    the local ``-z`` direction (tip at ``min-z``, base at ``max-z``) —
    that's the convention enforced by the ``probe-mesh`` canonicalization
    used throughout the codebase. Vertices within ``z_tolerance_mm`` of
    ``min-z`` are taken as candidate tip points; single-link clustering
    over their ``xy`` positions with ``cluster_radius_mm`` as the
    threshold groups within-shank tip neighbours together while keeping
    adjacent shanks (≥ 250 µm apart for NP 2.0 four-shank) separated.

    Parameters
    ----------
    mesh
        Probe mesh in local LPS mm.
    z_tolerance_mm
        Vertices with ``z ≤ min(z) + z_tolerance_mm`` are candidates.
        Default 50 µm — a probe shank tip is sharper than this in
        practice.
    cluster_radius_mm
        Single-link clustering threshold on ``xy`` distance. Default
        150 µm: smaller than the inter-shank pitch (250 µm on NP 2.0
        four-shank) so adjacent shanks stay separate, larger than
        within-shank vertex spread (~70 µm) so each shank groups into
        one cluster.

    Returns
    -------
    Float array shape ``(N, 3)`` of shank-tip centroids in probe local
    mm. Returns a single ``[0, 0, min(z)]`` if the mesh is empty.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) == 0:
        return np.zeros((1, 3), dtype=np.float64)

    z = verts[:, 2]
    z_min = float(z.min())
    bottom = verts[z <= z_min + z_tolerance_mm]
    if len(bottom) == 0:
        return np.array([[0.0, 0.0, z_min]], dtype=np.float64)
    if len(bottom) == 1:
        return bottom.copy()

    labels = fclusterdata(
        bottom[:, :2],
        t=cluster_radius_mm,
        criterion="distance",
        method="single",
        metric="euclidean",
    )
    n_clusters = int(labels.max())
    centroids = np.zeros((n_clusters, 3), dtype=np.float64)
    for cid in range(1, n_clusters + 1):
        centroids[cid - 1] = bottom[labels == cid].mean(axis=0)
    # Sort for stable output (helps tests + visual diff)
    order = np.lexsort(centroids[:, :3].T[::-1])
    return centroids[order]
