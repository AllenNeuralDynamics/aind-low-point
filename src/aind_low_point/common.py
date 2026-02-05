"""Things shared between the config and the run time"""

from __future__ import annotations

from enum import Enum, IntFlag


class Capability(IntFlag):
    RENDERABLE = 1
    MOVABLE = 2
    COLLIDABLE = 4
    SELECTABLE = 8
    DEFORMABLE = 16
    SAVABLE = 32


class Role(str, Enum):
    GEOMETRY = "geometry"
    TARGET = "target"
    LANDMARK = "landmark"
    ANATOMY = "anatomy"


class Kind(str, Enum):
    MESH = "mesh"
    POINTS = "points"
    LINES = "lines"
