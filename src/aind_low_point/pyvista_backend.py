"""PyVista rendering backend for use with trame."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pyvista as pv

from aind_low_point.rendering import ViewMaterial


def _int_to_hex(color_int: int) -> str:
    """Convert 0xRRGGBB int to '#RRGGBB' string for PyVista."""
    return f"#{color_int:06x}"


def _faces_to_pyvista(indices: np.ndarray) -> np.ndarray:
    """Convert (M, 3) triangle indices to PyVista faces format [3, i, j, k, ...]."""
    m = indices.shape[0]
    faces = np.empty((m, 4), dtype=indices.dtype)
    faces[:, 0] = 3
    faces[:, 1:] = indices
    return faces.ravel()


@dataclass
class PyVistaBackend:
    plotter: pv.Plotter
    _flush_callback: Callable[[], None] | None = None
    _actors: dict[str, pv.Actor] = field(default_factory=dict)
    _kinds: dict[str, str] = field(default_factory=dict)

    def create_mesh(
        self,
        node_id: str,
        *,
        name: str,
        vertices: np.ndarray,
        indices: np.ndarray,
        material: ViewMaterial,
    ) -> None:
        mesh = pv.PolyData(vertices, _faces_to_pyvista(indices))
        actor = self.plotter.add_mesh(
            mesh,
            name=name,
            color=_int_to_hex(material.color),
            opacity=material.opacity,
            show_edges=material.wireframe,
        )
        self._actors[node_id] = actor
        self._kinds[node_id] = "mesh"

    def update_mesh(
        self,
        node_id: str,
        *,
        vertices: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        material: ViewMaterial | None = None,
    ) -> None:
        actor = self._actors.get(node_id)
        if actor is None or self._kinds.get(node_id) != "mesh":
            return
        if vertices is not None:
            pd = actor.mapper.dataset
            pd.points = vertices
            pd.Modified()
        if material is not None:
            prop = actor.prop
            prop.color = _int_to_hex(material.color)
            prop.opacity = material.opacity

    def create_points(
        self,
        node_id: str,
        *,
        name: str,
        positions: np.ndarray,
        material: ViewMaterial,
        point_size: float = 1.0,
    ) -> None:
        cloud = pv.PolyData(positions)
        actor = self.plotter.add_points(
            cloud,
            name=name,
            color=_int_to_hex(material.color),
            point_size=point_size,
        )
        self._actors[node_id] = actor
        self._kinds[node_id] = "points"

    def update_points(
        self,
        node_id: str,
        *,
        positions: np.ndarray | None = None,
        material: ViewMaterial | None = None,
    ) -> None:
        actor = self._actors.get(node_id)
        if actor is None or self._kinds.get(node_id) != "points":
            return
        if positions is not None:
            pd = actor.mapper.dataset
            pd.points = positions
            pd.Modified()
        if material is not None:
            actor.prop.color = _int_to_hex(material.color)

    def remove(self, node_ids: Iterable[str]) -> None:
        for nid in node_ids:
            actor = self._actors.pop(nid, None)
            self._kinds.pop(nid, None)
            if actor is not None:
                self.plotter.remove_actor(actor)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._actors

    def flush(self) -> None:
        if self._flush_callback:
            self._flush_callback()
