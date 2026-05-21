"""K3D rendering backend"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Iterable,
)

import k3d
import numpy as np

from aind_low_point.rendering import RenderBackend


@dataclass
class K3DBackend(RenderBackend):
    plot: k3d.Plot
    _handles: dict[str, Any] = field(default_factory=dict)
    _kinds: dict[str, str] = field(default_factory=dict)  # 'mesh'|'points'

    def create_mesh(
        self, node_id, *, name, vertices, indices, material, model_matrix=None
    ):
        h = k3d.mesh(
            vertices.astype(float),
            indices.astype(np.uint32),
            name=name,
            color=int(material.color),
            opacity=float(material.opacity),
            wireframe=bool(material.wireframe),
        )
        if model_matrix is not None:
            h.model_matrix = model_matrix.astype(np.float32)
        if hasattr(h, "visible"):
            h.visible = bool(material.visible)
        self.plot += h
        self._handles[node_id] = h
        self._kinds[node_id] = "mesh"

    def update_mesh(
        self, node_id, *, vertices=None, indices=None, material=None, model_matrix=None
    ):
        h = self._handles.get(node_id)
        if h is None or self._kinds.get(node_id) != "mesh":
            return
        if model_matrix is not None:
            h.model_matrix = model_matrix.astype(np.float32)
        if vertices is not None:
            h.vertices = vertices.astype(float)
        if indices is not None:
            h.indices = indices.astype(np.uint32)
        if material is not None:
            h.color = int(material.color)
            h.opacity = float(material.opacity)
            if hasattr(h, "wireframe"):
                h.wireframe = bool(material.wireframe)
            if hasattr(h, "visible"):
                h.visible = bool(material.visible)

    def create_points(
        self, node_id, *, name, positions, material, point_size=1.0, model_matrix=None
    ):
        h = k3d.points(
            positions=positions.astype(float),
            name=name,
            color=int(material.color),
            point_size=float(point_size),
        )
        if model_matrix is not None:
            h.model_matrix = model_matrix.astype(np.float32)
        if hasattr(h, "visible"):
            h.visible = bool(material.visible)
        self.plot += h
        self._handles[node_id] = h
        self._kinds[node_id] = "points"

    def update_points(
        self, node_id, *, positions=None, material=None, model_matrix=None
    ):
        h = self._handles.get(node_id)
        if h is None or self._kinds.get(node_id) != "points":
            return
        if model_matrix is not None:
            h.model_matrix = model_matrix.astype(np.float32)
        if positions is not None:
            h.positions = positions.astype(float)
        if material is not None:
            h.color = int(material.color)
            if hasattr(h, "visible"):
                h.visible = bool(material.visible)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._handles

    def flush(self) -> None:
        pass  # K3D auto-syncs via ipywidgets traits

    def highlight(
        self,
        node_id: str | None,
        *,
        color: str = "#ffffff",
        width: float = 3.0,
    ) -> None:
        # K3D frontend doesn't expose per-actor edge visibility the same
        # way PyVista does; selection highlight is trame-only for now.
        pass

    def remove(self, node_ids: Iterable[str]) -> None:
        for nid in node_ids:
            h = self._handles.pop(nid, None)
            self._kinds.pop(nid, None)
            if h is not None:
                try:
                    self.plot -= h
                except Exception:
                    if hasattr(h, "visible"):
                        h.visible = False
