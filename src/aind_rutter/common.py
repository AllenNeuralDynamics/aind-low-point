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


class MRSignal(str, Enum):
    """Which resonance a feature's coordinates were localized from.

    Fat and water resonate about 3.5 ppm apart, so a scanner reconstructing at
    the water frequency places fat-derived signal some millimetres off along the
    readout axis. The canonical frame here is the headframe's, and the headframe
    is found from vaseline — so ``FAT`` features define the frame and ``WATER``
    features are translated into it.

    ``NONE`` is geometry the image never saw: CAD assumed rigid to the headframe
    (the cone, the well) or placed by the planner (the probes).
    """

    WATER = "water"
    FAT = "fat"
    NONE = "none"


class Role(str, Enum):
    """What an asset is, as against where its coordinates came from.

    `PROBE` and `FIXTURE` replace identifying those by key prefix: a probe was
    whatever was keyed ``probe:*``, so a subject could not name a mesh freely.
    """

    GEOMETRY = "geometry"
    TARGET = "target"
    LANDMARK = "landmark"
    ANATOMY = "anatomy"
    PROBE = "probe"
    FIXTURE = "fixture"


class Kind(str, Enum):
    MESH = "mesh"
    POINTS = "points"
    LINES = "lines"
