"""Things shared between the config and the run time"""

from __future__ import annotations

from enum import Enum, IntFlag, auto


class Capability(IntFlag):
    RENDERABLE = auto()
    MOVABLE = auto()
    COLLIDABLE = auto()
    SELECTABLE = auto()
    DEFORMABLE = auto()
    SAVABLE = auto()


class Role(str, Enum):
    GEOMETRY = "geometry"
    TARGET = "target"
    LANDMARK = "landmark"
    ANATOMY = "anatomy"


class Kind(str, Enum):
    MESH = "mesh"
    POINTS = "points"
    LINES = "lines"
