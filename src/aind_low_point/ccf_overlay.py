"""CCFOverlayManager — on-demand CCF region mesh overlays.

Manages its own PyVista actors directly (bypasses AssetCatalog /
RendererAdapter / Scene).  CCF regions are reference geometry, not
planned objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pyvista as pv
import SimpleITK as sitk
import trimesh
from aind_mri_utils.meshes import mask_to_trimesh

from aind_low_point.ccf_ontology import CCFOntology, CCFStructure


@dataclass
class RegionState:
    """Per-region cache and display state."""

    label_id: int
    structure: CCFStructure
    mesh: trimesh.Trimesh | None = None
    actor: pv.Actor | None = None
    visible: bool = True
    opacity: float = 1.0
    color: str = "#C8C8C8"


@dataclass
class CCFOverlayManager:
    """Manages transparent CCF region overlays in a PyVista plotter."""

    plotter: pv.Plotter
    volume_path: str | Path
    ontology: CCFOntology | None = None
    global_opacity: float = 0.3
    decimate_fraction: float = 0.25
    smooth_iters: int = 0
    flush_callback: Callable[[], None] | None = None

    _regions: dict[int, RegionState] = field(default_factory=dict)
    _volume: np.ndarray | None = field(default=None, repr=False)
    _sitk_image: sitk.Image | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.ontology is None:
            self.ontology = CCFOntology.from_bundled()

    def _load_volume(self) -> None:
        """Lazy-load the segmentation volume."""
        if self._sitk_image is not None:
            return
        self._sitk_image = sitk.ReadImage(str(self.volume_path))
        self._volume = sitk.GetArrayFromImage(self._sitk_image)

    def _extract_mesh(self, label_id: int) -> trimesh.Trimesh | None:
        """Extract a mesh for a single CCF label via marching cubes."""
        self._load_volume()
        binary = (self._volume == label_id).astype(np.uint8)
        if not binary.any():
            return None

        mask_img = sitk.GetImageFromArray(binary)
        mask_img.CopyInformation(self._sitk_image)

        mesh = mask_to_trimesh(mask_img, smooth_iters=self.smooth_iters)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_inversion(mesh)

        if len(mesh.faces) > 500:
            target = max(int(len(mesh.faces) * self.decimate_fraction), 100)
            mesh = mesh.simplify_quadric_decimation(target)

        return mesh

    def _actor_name(self, label_id: int) -> str:
        return f"ccf:{label_id}"

    def _effective_opacity(self, region: RegionState) -> float:
        return region.opacity * self.global_opacity

    def _add_actor(self, region: RegionState) -> None:
        """Add a PyVista actor for a cached mesh."""
        if region.mesh is None:
            return
        pv_mesh = pv.wrap(region.mesh)
        actor = self.plotter.add_mesh(
            pv_mesh,
            name=self._actor_name(region.label_id),
            color=region.color,
            opacity=self._effective_opacity(region),
        )
        region.actor = actor

    def _flush(self) -> None:
        if self.flush_callback is not None:
            self.flush_callback()

    def toggle(self, label_id: int) -> bool:
        """Toggle a region on/off. Returns new visibility state."""
        if label_id in self._regions:
            region = self._regions[label_id]
            if region.visible:
                self.hide(label_id)
                return False
            else:
                self.show(label_id)
                return True
        self.show(label_id)
        return True

    def show(self, label_id: int) -> None:
        """Show a CCF region (extract mesh on first call)."""
        if label_id in self._regions:
            region = self._regions[label_id]
            if region.visible:
                return
            region.visible = True
            if region.mesh is not None:
                self._add_actor(region)
                self._flush()
            return

        assert self.ontology is not None  # set in __post_init__
        structure = self.ontology.get(label_id)
        if structure is None:
            return
        color = structure.color_hex
        region = RegionState(
            label_id=label_id,
            structure=structure,
            color=color,
        )
        mesh = self._extract_mesh(label_id)
        if mesh is None:
            return
        region.mesh = mesh
        self._regions[label_id] = region
        self._add_actor(region)
        self._flush()

    def hide(self, label_id: int) -> None:
        """Hide a CCF region (keeps cached mesh)."""
        region = self._regions.get(label_id)
        if region is None or not region.visible:
            return
        region.visible = False
        self.plotter.remove_actor(self._actor_name(label_id))
        region.actor = None
        self._flush()

    def remove(self, label_id: int) -> None:
        """Remove a region entirely (actor + cached mesh)."""
        region = self._regions.pop(label_id, None)
        if region is None:
            return
        if region.actor is not None:
            self.plotter.remove_actor(self._actor_name(label_id))
        self._flush()

    def set_region_opacity(self, label_id: int, opacity: float) -> None:
        """Set per-region opacity (0-1)."""
        region = self._regions.get(label_id)
        if region is None:
            return
        region.opacity = opacity
        if region.actor is not None:
            region.actor.prop.opacity = self._effective_opacity(region)
            self._flush()

    def set_region_color(self, label_id: int, color: str) -> None:
        """Set per-region color."""
        region = self._regions.get(label_id)
        if region is None:
            return
        region.color = color
        if region.actor is not None:
            region.actor.prop.color = color
            self._flush()

    def set_global_opacity(self, opacity: float) -> None:
        """Set global opacity multiplier for all visible regions."""
        self.global_opacity = opacity
        for region in self._regions.values():
            if region.actor is not None:
                region.actor.prop.opacity = self._effective_opacity(region)
        self._flush()

    def clear_all(self) -> None:
        """Remove all CCF overlays."""
        for label_id in list(self._regions):
            self.remove(label_id)

    def visible_regions(self) -> list[RegionState]:
        """Return list of currently visible regions."""
        return [r for r in self._regions.values() if r.visible]

    def available_labels(self) -> set[int]:
        """Return the set of label ids present in the volume."""
        self._load_volume()
        vol = self._volume
        if vol is None:
            return set()
        return set(int(x) for x in np.unique(vol)) - {0}
