"""STEP (ISO 10303) solids tessellated into closed surface meshes.

OpenCascade comes from the ``step`` extra and is imported on first use, so the
rest of the package loads without it. Coordinates are the file's part frame in
millimetres (OpenCascade converts the file's length unit); orient them with the
asset's canonicalization like any other mesh.
"""

from __future__ import annotations

import warnings

import numpy as np
import trimesh
from numpy.typing import NDArray

DEFAULT_LINEAR_DEFLECTION_MM = 0.01
DEFAULT_ANGULAR_DEFLECTION_RAD = 0.1


def read_step_shape(path: str):
    """The STEP file's OpenCascade shape, in its part frame and mm."""
    try:
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
    except ImportError as exc:
        raise ImportError(
            f"{path}: reading STEP needs OpenCascade; install the 'step' extra "
            "(aind-rutter[step])"
        ) from exc
    reader = STEPControl_Reader()
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise ValueError(f"{path}: not a readable STEP file")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError(f"{path}: STEP file holds no shape")
    return shape


def _tessellate(
    shape, linear_deflection_mm: float, angular_deflection_rad: float
) -> trimesh.Trimesh:
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    # Meshing stores the triangulation on the shape and a later, coarser request
    # reuses it, so each call must start from a freshly read shape.
    BRepMesh_IncrementalMesh(
        shape, linear_deflection_mm, False, angular_deflection_rad, True
    )
    vertices: list[NDArray[np.float64]] = []
    faces: list[NDArray[np.int64]] = []
    n_vertices = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        explorer.Next()
        location = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, location)
        if tri is None:
            continue
        placement = location.Transformation()
        nodes = np.array(
            [
                tri.Node(k).Transformed(placement).Coord()
                for k in range(1, tri.NbNodes() + 1)
            ]
        )
        triangles = (
            np.array(
                [
                    [tri.Triangle(j).Value(k) for k in (1, 2, 3)]
                    for j in range(1, tri.NbTriangles() + 1)
                ],
                dtype=np.int64,
            )
            - 1
        )
        if face.Orientation() == TopAbs_REVERSED:
            triangles = triangles[:, ::-1]
        vertices.append(nodes)
        faces.append(triangles + n_vertices)
        n_vertices += len(nodes)
    if not faces:
        raise ValueError("STEP shape has no faces to tessellate")
    # Faces sharing a CAD edge share its nodes exactly, so merging closes the seams.
    mesh = trimesh.Trimesh(np.vstack(vertices), np.vstack(faces), process=False)
    mesh.merge_vertices()
    return mesh


def _drop_degenerate_and_doubled(mesh: trimesh.Trimesh) -> None:
    """Remove zero-area slivers and triangles present twice.

    A doubled triangle is a zero-volume fin between two faces whose
    triangulations overlap; keeping either copy leaves a non-manifold edge.
    """
    mesh.update_faces(mesh.nondegenerate_faces())
    _, inverse, counts = np.unique(
        np.sort(mesh.faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    mesh.update_faces(counts[inverse.ravel()] == 1)
    mesh.remove_unreferenced_vertices()


def _boundary_loops(mesh: trimesh.Trimesh) -> list[list[int]]:
    """Closed vertex loops along edges used by one face, in that face's direction."""
    _, inverse, counts = np.unique(
        mesh.edges_sorted, axis=0, return_inverse=True, return_counts=True
    )
    if (counts > 2).any():
        return []
    successor = {int(a): int(b) for a, b in mesh.edges[counts[inverse.ravel()] == 1]}
    loops: list[list[int]] = []
    visited: set[int] = set()
    for start in successor:
        loop: list[int] = []
        vertex = start
        while vertex not in visited and vertex in successor:
            visited.add(vertex)
            loop.append(vertex)
            vertex = successor[vertex]
        if vertex == start and len(loop) >= 3:
            loops.append(loop)
    return loops


def _patch_loop(mesh: trimesh.Trimesh, loop: list[int]) -> NDArray[np.int64] | None:
    """Triangles closing ``loop`` in its best-fit plane, or None if it folds there."""
    from shapely.geometry import Polygon

    points = mesh.vertices[loop]
    centred = points - points.mean(axis=0)
    _, _, basis = np.linalg.svd(centred)
    polygon = Polygon(centred @ basis[:2].T)
    if not polygon.is_valid:
        return None
    _, local = trimesh.creation.triangulate_polygon(polygon, engine="earcut")
    # earcut indexes the ring with its closing point repeated at the end.
    patch = np.asarray(loop)[np.asarray(local) % len(loop)]
    # The mesh runs each boundary edge a->b; the patch must run it b->a.
    a, b = loop[0], loop[1]
    if any((t[k], t[(k + 1) % 3]) == (a, b) for t in patch for k in range(3)):
        patch = patch[:, ::-1]
    return patch


def _close_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Patch the gaps a face's malformed triangulation leaves in a closed B-rep."""
    patches = [
        p
        for loop in _boundary_loops(mesh)
        if (p := _patch_loop(mesh, loop)) is not None
    ]
    if not patches:
        return mesh
    return trimesh.Trimesh(
        mesh.vertices, np.vstack([mesh.faces, *patches]), process=False
    )


def load_step(
    path: str,
    linear_deflection_mm: float = DEFAULT_LINEAR_DEFLECTION_MM,
    angular_deflection_rad: float = DEFAULT_ANGULAR_DEFLECTION_RAD,
) -> trimesh.Trimesh:
    """Read a STEP file and tessellate every face into one surface mesh.

    Parameters
    ----------
    linear_deflection_mm
        Largest distance a triangle may sit from the surface it approximates.
    angular_deflection_rad
        Largest angle between adjacent triangles along a curved surface.

    A closed solid yields a watertight mesh with outward normals; where the
    tessellation cannot be closed the mesh is returned open, with a warning.
    """
    mesh = _tessellate(
        read_step_shape(path), linear_deflection_mm, angular_deflection_rad
    )
    _drop_degenerate_and_doubled(mesh)
    if not mesh.is_watertight:
        mesh = _close_holes(mesh)
    if mesh.is_watertight:
        if mesh.volume < 0:
            mesh.invert()
    else:
        _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
        warnings.warn(
            f"{path}: tessellation is not watertight "
            f"({int((counts == 1).sum())} open edges, "
            f"{int((counts > 2).sum())} non-manifold)",
            stacklevel=2,
        )
    return mesh
