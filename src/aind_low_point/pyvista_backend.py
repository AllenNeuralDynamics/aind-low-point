"""PyVista rendering backend for use with trame."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pyvista as pv

from aind_low_point.rendering import ViewMaterial


class DebouncedFlush:
    """Debounce a flush callback on the running asyncio event loop.

    Each call cancels any pending timer and schedules a new one.
    Only the last call within *delay_s* actually fires.
    Falls back to immediate invocation outside an asyncio context.
    """

    def __init__(self, callback: Callable[[], None], delay_s: float = 0.1) -> None:
        self._callback = callback
        self._delay = delay_s
        self._handle: asyncio.TimerHandle | None = None

    def __call__(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._callback()
            return
        if self._handle is not None:
            self._handle.cancel()
        self._handle = loop.call_later(self._delay, self._fire)

    def _fire(self) -> None:
        self._handle = None
        self._callback()


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
        model_matrix: np.ndarray | None = None,
    ) -> None:
        mesh = pv.PolyData(vertices, _faces_to_pyvista(indices))
        actor = self.plotter.add_mesh(
            mesh,
            name=name,
            color=_int_to_hex(material.color),
            opacity=material.opacity,
            show_edges=material.wireframe,
        )
        if model_matrix is not None:
            actor.user_matrix = model_matrix
        self._actors[node_id] = actor
        self._kinds[node_id] = "mesh"

    def update_mesh(
        self,
        node_id: str,
        *,
        vertices: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        material: ViewMaterial | None = None,
        model_matrix: np.ndarray | None = None,
    ) -> None:
        actor = self._actors.get(node_id)
        if actor is None or self._kinds.get(node_id) != "mesh":
            return
        if model_matrix is not None:
            actor.user_matrix = model_matrix
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
        model_matrix: np.ndarray | None = None,
    ) -> None:
        cloud = pv.PolyData(positions)
        actor = self.plotter.add_points(
            cloud,
            name=name,
            color=_int_to_hex(material.color),
            opacity=material.opacity,
            point_size=point_size,
        )
        if model_matrix is not None:
            actor.user_matrix = model_matrix
        self._actors[node_id] = actor
        self._kinds[node_id] = "points"

    def update_points(
        self,
        node_id: str,
        *,
        positions: np.ndarray | None = None,
        material: ViewMaterial | None = None,
        model_matrix: np.ndarray | None = None,
    ) -> None:
        actor = self._actors.get(node_id)
        if actor is None or self._kinds.get(node_id) != "points":
            return
        if model_matrix is not None:
            actor.user_matrix = model_matrix
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
