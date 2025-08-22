"""Things shared between the config and the run time"""

from __future__ import annotations

from enum import Enum


class Capability(str, Enum):
    RENDERABLE = "renderable"
    MOVABLE = "movable"
    COLLIDABLE = "collidable"
    SELECTABLE = "selectable"
    DEFORMABLE = "deformable"
    SAVABLE = "savable"


class Role(str, Enum):
    GEOMETRY = "geometry"
    TARGET = "target"
    LANDMARK = "landmark"
    ANATOMY = "anatomy"


class Kind(str, Enum):
    MESH = "mesh"
    POINTS = "points"
    LINES = "lines"
