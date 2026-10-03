#!/usr/bin/env python3
"""Render a publication figure from a config plus an accepted plan-only YAML.

Builds the same runtime the interactive planner builds, applies a plan
through the same command path the "Load plan" button uses, then renders
offscreen with no UI chrome. The figure therefore comes from the code that
produced the plan, which is the provenance claim worth being able to make in
a methods section.

Defaults follow the PL-inputs figure contract (``FIGURE-STYLE.md`` in the
ccg-analysis monorepo) so panels drawn here compose with panels drawn
there: 57 x 42.75 mm at 600 dpi, the shared region colour key, neutral
hardware, a 0.6 pt rule on cropped panels. Constants are vendored near the
top of this file with a pointer back; diff on touch.

Two consequences are worth stating outright. Panels are sized in
millimetres, not pixels, and rendered at that size — a figure drawn large
and shrunk into a 57 mm column hides how much of it survives the
reduction. And anything VTK measures in screen pixels (line widths, point
sizes) is stated here in points and converted by ``line_px``, so it prints
at the weight it was set rather than at whatever the resolution made of
it.

Scale bars and parallel projection, which that contract also covers, are
deliberately not implemented: these panels are read for arrangement, not
measured off the page, and perspective is what makes the depth legible.

Two framings are available:

``overview``
    Whole-head view framed on the brain. Establishes the multi-probe
    geometry and the scale of the problem.

``implant``
    Tight view framed on the implant, with the implant rendered
    semi-transparent so probes threading their bores are visible. This is
    the evidence panel — it shows the constraint being satisfied.

Examples
--------
Size, colour and weight are already the defaults; only framing is per-panel::

    uv run --python 3.13 python scripts/render_plan_figure.py \\
        --config subject.yml --plan plan.plan.yml --out figures/subject

The manipulator crowding, seen dorsally and tilted forward::

    ... --panel overview --view dorsal --dorsal-tilt 1.25 --frame-on brain \\
        --hide-tags hole headframe cone --extend-up 14 --transparent

The anatomy, from the side::

    ... --panel overview --view left --frame-on brain \\
        --hide-tags hole cone headframe --zoom 2.0 --transparent

From an isometric angle the camera lands among the splayed manipulator
bodies, and the cone and headframe bury the brain, hence ``--view left`` and
``--hide-tags``. Do not reach for ``--clip-above`` instead: bodies cut
mid-shaft read as a rendering artifact; let the frame crop them.

The panel border is on by default (``--border-pt 0.6``). Both panels crop
the manipulator bodies, and without a rule the cut ends read as geometry
that simply stops rather than as geometry leaving the panel. A panel that
crops nothing is better without it: pass ``--border-pt 0``.

Each render reports how much of the panel it uses, measured before the
border goes on: ``bbox`` is the content bounding box, ``ink`` the fraction
of pixels that are content. Margin is free on a slide and expensive at
57 mm. Watch ``ink`` — with manipulators splayed to every edge, ``bbox``
reads 100% for any framing at all.

Vector output for a journal that wants editable line art::

    ... --out figures/subject --format pdf

Runs headless as-is on VTK 9.3+, which falls back to OSMesa or EGL when
``DISPLAY`` is unset. A probe render at startup fails loudly if that
fallback is missing, rather than writing a blank figure.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyvista as pv
import tyro
import yaml

from aind_rutter.build.assemble import build_runtime_from_config
from aind_rutter.config import ConfigModel, PlanningModel
from aind_rutter.plan_io.replay import apply_plan_model_to_state
from aind_rutter.render.adapter import RendererAdapter, RenderHandler
from aind_rutter.render.overlays import OverlayResolver, OverlayState
from aind_rutter.render.pyvista import PyVistaBackend
from aind_rutter.session.store import PlanStore

# Vendored from the PL-inputs figure contract: FIGURE-STYLE.md and
# src/pl_inputs_ccf_figures/render.py in the ccg-analysis monorepo. That
# document is the authority; diff against it when either side moves. Panels
# from the two repos are composed side by side, so a private palette or panel
# size here shows up as an inconsistency on the page.

#: Three panels across a 183 mm Nature double column with 6 mm gutters:
#: 3*57 + 2*6 = 183. Panels are drawn at this width and never reduced in
#: layout, which is what lets every weight below be stated in points.
PANEL_WIDTH_MM = 57.0
PANEL_ASPECT = 4 / 3
PANEL_DPI = 600.0

#: Region key, shared with the rasters and CCG traces so a colour means the
#: same structure across the whole figure. Probes here are named for their
#: target, so a probe takes its target's colour.
TARGET_COLOURS: dict[str, str] = {
    "PL": "#9467bd",
    "MD": "#17becf",
    "VM": "#e377c2",
    "CA1": "#2ca02c",
    # Ventral CA1 is CA1 to the reader, so it shares the colour.
    "vCA1": "#2ca02c",
    "BLA": "#ff7f0e",
    "RSP": "#d62728",
    "CLA": "#1f77b4",
}

_COLOUR_BY_UPPER = {k.upper(): v for k, v in TARGET_COLOURS.items()}

#: Probe shanks where they are context rather than the subject of the panel.
SHANK_COLOUR = "#7a7a7a"

#: Skull, headframe, well, cone, implant: geometry that locates the anatomy
#: without being the claim.
CONTEXT_GREY = "#dddddd"

#: Retrogradely labelled cells. Reserved by the contract — the config's own
#: #FF4444 is close enough to RSP's #d62728 to read as a region.
RETRO_COLOUR = "#ff6e00"

BORDER_COLOUR = "#222222"

# LPS axes: +x = Left, +y = Posterior, +z = Superior. Each entry is the
# camera offset direction from the focal point, plus the view-up vector.
# Anterior is -y, so the superior-left-anterior octant is (+1, -1, +1).
VIEWS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "iso": ((1.0, -1.0, 1.0), (0.0, 0.0, 1.0)),
    "left": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "right": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "anterior": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "posterior": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    # Looking straight down +z, view-up must leave the view axis: use
    # anterior-up so the reader sees the head nose-up.
    "superior": ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    # Dorsal but tilted off vertical: camera sits above and posterior, so
    # the view runs downward *and* forward toward the nose. Reads as
    # looking over the animal's head rather than flat onto it.
    "dorsal": ((0.0, 0.45, 1.0), (0.0, -1.0, 0.0)),
}

ViewName = Literal[
    "iso", "left", "right", "anterior", "posterior", "superior", "dorsal"
]
ProbeColorMode = Literal["region", "grey", "index"]
PanelName = Literal["overview", "implant", "both"]
FormatName = Literal["png", "pdf", "svg", "eps"]


@dataclass
class Args:
    """Arguments for the figure renderer."""

    config: Path
    """Full ``ConfigModel`` YAML for the subject."""

    plan: Path
    """Accepted plan-only YAML (a ``PlanningModel``, as written by
    ``rutter-emit`` or the app's "Save plan" button). An exported geometry
    report — the kind carrying ``plan_export_version`` — is not loadable
    here and is rejected with a clear message."""

    out: Path
    """Output path *stem*. The panel name and extension are appended, so
    ``figures/subject`` yields ``figures/subject-overview.png``."""

    panel: PanelName = "both"
    """Which framing to render."""

    view: ViewName = "iso"
    """Camera direction for the overview panel."""

    implant_view: ViewName = "left"
    """Camera direction for the implant panel. A lateral view reads the
    bore threading most clearly; ``superior`` shows the hole pattern."""

    size: tuple[int, int] = (1600, 1400)
    """Render size in pixels before ``scale`` is applied. Ignored when
    ``width_mm``/``height_mm`` are given."""

    width_mm: float | None = PANEL_WIDTH_MM
    """Printed panel width in millimetres. Setting this (with
    ``height_mm``) sizes the render to the space the panel actually
    occupies on the page: the viewport is the printed pixel count, so
    anti-aliasing, silhouette softness and every sub-pixel feature resolve
    at print size instead of at some larger size the reader never sees.
    ``scale`` then becomes pure supersampling — the render is downsampled
    back to the target — and the PNG carries a matching DPI so it places
    at this width without rescaling."""

    height_mm: float | None = PANEL_WIDTH_MM / PANEL_ASPECT
    """Printed panel height in millimetres. Must accompany ``width_mm``.
    Its ratio to ``width_mm`` sets the viewport aspect, which changes the
    framing: check ``zoom`` and ``pan`` after changing it."""

    dpi: int = round(PANEL_DPI)
    """Print resolution used to turn ``width_mm``/``height_mm`` into
    pixels, and written into the PNG."""

    scale: int = 1
    """Supersampling factor. Leave at 1: a window already sized to the panel
    makes one pixel a real length, which is what removes the whole class of
    bug where a quantity VTK measures in pixels prints at the wrong weight.
    Raising it is safe — ``line_px`` compensates — but it buys little past
    600 dpi and costs the guarantee. At the default size this gives a 4800x4200
    PNG, comfortably past 300 dpi for a single-column panel. Under
    ``width_mm`` it is supersampling only: the result is resampled down to
    the printed size."""

    format: FormatName = "png"
    """``png`` rasterizes; ``pdf``/``svg``/``eps`` emit vector art via
    VTK's gl2ps exporter. Vector output ignores ``scale`` and renders
    transparency less faithfully — check it before committing to it."""

    zoom: float = 1.0
    """Extra zoom applied after framing. Above 1 tightens the crop."""

    implant_zoom: float = 1.0
    """Framing zoom for the implant panel, applied on top of ``zoom``."""

    background: str = "white"
    """Background color. Ignored when ``--transparent`` is set."""

    transparent: bool = False
    """Write an alpha channel instead of a background (PNG only)."""

    implant_opacity: float = 0.35
    """Implant opacity in the implant panel. The app uses 0.2, which is
    tuned for interactive use; a print figure wants the bore walls
    slightly more legible."""

    fixture_opacity: float = 0.45
    """Opacity for headframe, well, and cone in both panels."""

    brain_opacity: float = 0.06
    """Opacity for the brain shell, matching the companion panels' root
    alpha. It is deliberately almost invisible: the brain is read from its
    outline, not its fill, which keeps the structures and shanks inside it
    unclouded. Raising it instead of using the outline works — a
    translucent shell darkens at grazing angles — but it greys everything
    the panel is about, and past ~0.4 the shanks go entirely."""

    brain_color: str | None = None
    """Override the brain color. Defaults to the config's, which is a pale
    pink chosen for the app rather than for print."""

    brain_silhouette_smooth: int = 120
    """Taubin smoothing passes applied to a *copy* of the brain before
    silhouetting. Load-bearing, not cosmetic: the filter strokes every
    silhouette edge, so the raw subject mask — faceted, and foliated at
    the cerebellum — comes out as a scribble of interior folds instead of
    a contour. Taubin is volume-preserving, so the smoothed outline still
    sits on the shell. It converges by ~120; higher costs time and changes
    nothing. The rendered shell itself is untouched."""

    brain_silhouette_pt: float = 0.5
    """Stroke width in points of the brain's outline, matching the region
    silhouettes in the companion panels. Set 0 for no outline, in which
    case ``brain_opacity`` has to carry the brain on its own."""

    brain_silhouette_color: str = BORDER_COLOUR
    """Color of the brain outline."""

    border_pt: float = 0.6
    """Stroke width in points for a rule drawn round the panel. Worth
    setting whenever the frame crops geometry: without a border the cut
    manipulator bodies read as ending in mid-air rather than as running
    out of the panel. PNG only — for vector output draw the rule in the
    layout program."""

    border_color: str = BORDER_COLOUR
    """Color of the panel border."""

    probe_opacity: float = 1.0
    """Opacity for probe bodies. From a dorsal view the manipulators sit
    directly between camera and brain and occlude it by construction, so
    ghosting them (~0.35) is the only way to put the brain in front
    without losing the probes. Lateral views need no such help."""

    clip_above: float | None = None
    """Optionally clip probe and fixture geometry above this LPS z (mm,
    +z Superior). Off by default: cutting the manipulators mid-body reads
    as a rendering artifact rather than a design choice. Frame wider
    instead (``--frame-on scene``)."""

    frame_on: Literal["scene", "brain", "implant"] = "scene"
    """What the overview panel frames on. ``scene`` fits every visible
    mesh, so the full manipulator assembly is in shot at the cost of a
    smaller brain; ``brain`` fits the brain alone, which puts the camera
    inside the probe bodies unless they are clipped."""

    extend_up: float = 0.0
    """Extend the framing box this many mm in +z (Superior) before
    fitting. The probes all rise from the head, so a box drawn round the
    brain alone centres the camera below the geometry and leaves the
    lower third of the panel empty. Extending upward recentres on the
    converging shafts."""

    dorsal_tilt: float = 0.45
    """How far the ``dorsal`` view leans off vertical, as the posterior
    (+y) component against a fixed +z of 1. Larger is more oblique: 0 is
    straight down, ~0.5 looks over the head, ~1.0 is a 45-degree
    three-quarter view. Ignored by every other view."""

    pan: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translate camera and focal point together by this LPS offset in mm
    after framing (+x Left, +y Posterior, +z Superior). Use it to push the
    subject off-centre and reclaim dead space on one side of the panel."""

    hide_tags: tuple[str, ...] = ()
    """Scene tags to hide entirely, e.g. ``--hide-tags headframe cone``
    to strip the rig hardware down to implant and probes."""

    anti_alias: bool = True
    """Supersample anti-aliasing. Leave on for print."""

    parallel: bool = False
    """Use an orthographic camera. Preferred by some journals because it
    keeps distances comparable across the panel."""

    orbit: tuple[float, float] = (0.0, 0.0)
    """Extra (azimuth, elevation) degrees applied after framing, for
    nudging the view off the preset without editing ``VIEWS``."""

    probe_colors: ProbeColorMode = "region"
    """How probes are coloured. ``region`` gives each probe its target's
    colour from :data:`TARGET_COLOURS`, so a hue means the same structure
    here as in the rasters — the probe and the region it reaches read as
    one object. ``grey`` draws them as context in ``SHANK_COLOUR``, for a
    panel whose colour carries something else. ``index`` falls back to
    ``palette``, which encodes nothing and should not go in the paper."""

    region_colors: bool = True
    """Recolour the target structure meshes from :data:`TARGET_COLOURS`.
    The config's own greens and blues were picked per subject for the app
    and collide with the paper's key."""

    fixture_color: str | None = CONTEXT_GREY
    """Colour for headframe, well, cone, and implant. Neutral by contract:
    these locate the anatomy without being the claim."""

    retro_color: str | None = RETRO_COLOUR
    """Colour for the retrograde cell cloud."""

    retro_point_pt: float = 0.35
    """Retrograde cell marker diameter in points. VTK sizes points in
    screen pixels, not world units, so the config's 3.0 means one weight
    on screen and another on paper. 0.35 pt is roughly the 22 um marker
    radius the companion panels draw, carried through this panel's
    magnification."""

    palette: tuple[str, ...] = (
        "#1b6ca8",
        "#e07b39",
        "#3f8f4a",
        "#b5384d",
        "#7b5ea7",
        "#8a6a3f",
        "#3c8f8f",
    )
    """Probe colors used when ``--probe-colors index`` is set."""

    quiet: bool = False
    """Suppress the provenance banner."""


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------


MM_PER_INCH = 25.4


def print_size(args: Args) -> tuple[int, int] | None:
    """Viewport size for a physically-sized panel, or ``None``.

    Raster panels size at ``dpi``; gl2ps writes one PostScript point per
    window pixel, so vector formats size at 72 instead and come out as a
    page of the requested millimetres.
    """
    if args.width_mm is None and args.height_mm is None:
        return None
    if args.width_mm is None or args.height_mm is None:
        raise SystemExit("--width-mm and --height-mm must be given together.")
    per_inch = args.dpi if args.format == "png" else 72
    return (
        round(args.width_mm / MM_PER_INCH * per_inch),
        round(args.height_mm / MM_PER_INCH * per_inch),
    )


def render_size(args: Args) -> tuple[int, int]:
    """Viewport size in pixels, however it was specified."""
    return print_size(args) or args.size


def resample_to(path: Path, size: tuple[int, int], dpi: int) -> None:
    """Resample a supersampled screenshot down to its printed pixel size.

    RGBA is resized premultiplied: VTK leaves the background colour in the
    RGB of fully transparent pixels, so straight-alpha filtering bleeds a
    halo of it into every silhouette. The stored DPI is what makes the
    file place at its intended width.
    """
    from PIL import Image

    with Image.open(path) as opened:
        img = opened.copy()

    if img.size != size:
        if img.mode == "RGBA":
            src = np.asarray(img, dtype=np.float64)
            alpha = src[..., 3:4]
            premul = np.concatenate([src[..., :3] * alpha / 255.0, alpha], axis=-1)
            small = np.asarray(
                Image.fromarray(premul.round().astype(np.uint8)).resize(
                    size, Image.Resampling.LANCZOS
                ),
                dtype=np.float64,
            )
            out_a = small[..., 3:4]
            rgb = np.where(
                out_a > 0, small[..., :3] * 255.0 / np.maximum(out_a, 1e-6), 255.0
            )
            img = Image.fromarray(
                np.clip(np.concatenate([rgb, out_a], axis=-1), 0, 255).astype(np.uint8)
            )
        else:
            img = img.resize(size, Image.Resampling.LANCZOS)

    img.save(path, dpi=(dpi, dpi))


def ensure_gl_context() -> None:
    """Fail loudly now if offscreen rendering would produce a blank image.

    VTK 9.3+ falls back to OSMesa or EGL on its own when ``DISPLAY`` is
    unset, so a headless box usually needs no help. When the fallback is
    missing the render does not raise — it silently writes a uniform
    image. A 32x32 probe render turns that into an error at startup
    instead of a black figure discovered later.
    """
    probe = pv.Plotter(off_screen=True, window_size=(32, 32))
    try:
        probe.set_background("white")
        probe.add_mesh(pv.Sphere(), color="red")
        img = probe.screenshot(return_img=True)
    except Exception as exc:
        raise SystemExit(f"Offscreen rendering is unavailable: {exc}") from exc
    finally:
        probe.close()

    if len(np.unique(np.asarray(img).reshape(-1, 3), axis=0)) < 5:
        raise SystemExit(
            "Offscreen rendering produced a blank image, so VTK has no usable "
            "GL context.\nInstall a virtual framebuffer (Fedora: sudo dnf "
            "install xorg-x11-server-Xvfb) and rerun under `xvfb-run -a`, or "
            "use a VTK wheel built with OSMesa."
        )


def load_plan(path: Path) -> PlanningModel:
    """Load and validate a plan-only YAML, rejecting geometry reports."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if isinstance(raw, dict) and "plan_export_version" in raw:
        raise SystemExit(
            f"{path} is an exported geometry report "
            f"(plan_export_version={raw['plan_export_version']}), not a loadable "
            "plan. Use a plan-only YAML from rutter-emit or the 'Save plan' button."
        )
    return PlanningModel.model_validate(raw)


def git_revision() -> str:
    """Short git revision, or ``unknown`` outside a repo."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{rev}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# --------------------------------------------------------------------------
# scene assembly
# --------------------------------------------------------------------------


def build_scene(args: Args) -> tuple[pv.Plotter, PyVistaBackend, RendererAdapter, dict]:
    """Build runtime, wire the renderer, and apply the plan.

    Mirrors :func:`aind_rutter.web.app.build_trame_app` minus the collision
    adapter and every widget. Collisions are skipped deliberately: they
    would recolor meshes red as an overlay, and an accepted plan has
    already passed the FCL gate.
    """
    cfg = ConfigModel.from_yaml(args.config)
    bundle = build_runtime_from_config(cfg)
    catalog, scene, plan_state = bundle.asset_catalog, bundle.scene, bundle.plan_state

    plotter = pv.Plotter(off_screen=True, window_size=list(render_size(args)))

    backend = PyVistaBackend(plotter=plotter)
    adapter = RendererAdapter(
        backend=backend,
        scene=scene,
        assets=catalog,
        overlays=OverlayResolver(overlays=OverlayState()),
    )

    store = PlanStore(plan_state)
    store.subscribe(RenderHandler(scene=scene, adapter=adapter))
    adapter.build(plan_state)

    plan = load_plan(args.plan)
    touched = apply_plan_model_to_state(plan, store)

    return plotter, backend, adapter, {"probes": touched, "catalog": catalog}


def target_colour(name: str) -> str | None:
    """``name``'s colour in :data:`TARGET_COLOURS`, ignoring case."""
    return _COLOUR_BY_UPPER.get(name.upper())


def _apply_opacity(actor, node, args: Args, implant_opacity: float) -> None:
    """Set one node's opacity from its role in the panel."""
    tags = node.tags
    if node.key == "brain":
        actor.prop.opacity = args.brain_opacity
    elif "implant" in tags:
        actor.prop.opacity = implant_opacity
    elif tags & {"fixture", "headframe", "cone", "well"}:
        actor.prop.opacity = args.fixture_opacity
    elif "probe" in tags and args.probe_opacity != 1.0:
        actor.prop.opacity = args.probe_opacity


def _apply_color(actor, node, args: Args, indexed: dict[str, str]) -> None:
    """Set one node's colour under the shared region key.

    Colour carries exactly one thing across the figure — which structure —
    so a probe takes the colour of the region it is aimed at, and anything
    that is context takes a neutral.
    """
    tags = node.tags
    key = node.key

    if key == "brain":
        if args.brain_color is not None:
            actor.prop.color = args.brain_color
        return

    if "implant" in tags or tags & {"fixture", "headframe", "cone", "well"}:
        if args.fixture_color is not None:
            actor.prop.color = args.fixture_color
        return

    if "retro-target" in tags or key == "retro-targets":
        if args.retro_color is not None:
            actor.prop.color = args.retro_color
        actor.prop.point_size = line_px(args.retro_point_pt, args)
        return

    if args.region_colors and key.startswith("structure:"):
        colour = target_colour(key.split(":", 1)[1])
        if colour is not None:
            actor.prop.color = colour
        return

    if key.startswith("probe:"):
        name = key.split(":", 1)[1]
        if args.probe_colors == "grey":
            actor.prop.color = SHANK_COLOUR
        elif args.probe_colors == "index":
            actor.prop.color = indexed[name]
        elif (colour := target_colour(name)) is not None:
            actor.prop.color = colour


def style_scene(
    backend: PyVistaBackend,
    adapter: RendererAdapter,
    args: Args,
    *,
    implant_opacity: float,
) -> None:
    """Apply figure styling: opacities, hidden tags, the region colour key.

    Pokes actor properties directly, the same way the app's
    ``apply_default_opacities`` does, because opacity here is a
    presentation choice rather than a change to the plan.
    """
    probe_names = sorted(
        n.key.split(":", 1)[1]
        for n in adapter.scene.nodes.values()
        if n.key.startswith("probe:")
    )
    indexed = {
        name: args.palette[i % len(args.palette)] for i, name in enumerate(probe_names)
    }

    for node in adapter.scene.nodes.values():
        actor = backend.actor(node.key)
        if actor is None:
            continue
        if node.tags & set(args.hide_tags):
            actor.visibility = False
            continue
        _apply_opacity(actor, node, args, implant_opacity)
        _apply_color(actor, node, args, indexed)
        # Smooth shading reads better in print than VTK's default flat normals.
        actor.prop.interpolation = "phong"
        actor.prop.specular = 0.15
        actor.prop.specular_power = 24


# --------------------------------------------------------------------------
# framing
# --------------------------------------------------------------------------


def _union(boxes: list[tuple[float, ...]]) -> tuple[float, ...] | None:
    """Union of VTK-flat bounds ``(xmin, xmax, ymin, ymax, zmin, zmax)``."""
    real = [b for b in boxes if b[1] > b[0] or b[3] > b[2] or b[5] > b[4]]
    if not real:
        return None
    arr = np.asarray(real, dtype=np.float64)
    return (
        float(arr[:, 0].min()),
        float(arr[:, 1].max()),
        float(arr[:, 2].min()),
        float(arr[:, 3].max()),
        float(arr[:, 4].min()),
        float(arr[:, 5].max()),
    )


def node_bounds(
    backend: PyVistaBackend, adapter: RendererAdapter, tag: str
) -> tuple[float, ...] | None:
    """World bounds of visible nodes carrying ``tag``, ignoring point nodes.

    Actor bounds already include ``user_matrix``, so this is the posed
    geometry rather than the asset's local mesh. Degenerate boxes are
    dropped: target nodes are single points, and a tag can land on
    nothing but points — ``brain`` does exactly that in configs whose
    brain mesh has no scene node.
    """
    return _union(
        [
            actor.GetBounds()
            for node in adapter.scene.nodes.values()
            if tag in node.tags
            and (actor := backend.actor(node.key)) is not None
            and actor.visibility
        ]
    )


def scene_bounds(
    backend: PyVistaBackend, adapter: RendererAdapter
) -> tuple[float, ...] | None:
    """Bounds of every visible mesh node, probes and fixtures included.

    Framing on this puts the whole manipulator assembly in shot. The
    brain ends up small — a 20 mm brain against 160 mm of hardware — but
    the crowding is the point, and nothing has to be cut to show it.
    """
    return _union(
        [
            actor.GetBounds()
            for node in adapter.scene.nodes.values()
            if (actor := backend.actor(node.key)) is not None
            and actor.visibility
            and backend.is_mesh(node.key)
        ]
    )


def brain_bounds(
    backend: PyVistaBackend, adapter: RendererAdapter
) -> tuple[float, ...] | None:
    """Bounds to frame the overview on: the posed brain, else structures.

    Read from the brain node's *actor*, not ``catalog.assets["brain"]``.
    An asset's ``mesh.raw`` is pre-transform, and a brain carrying
    ``transform: headframe_to_lps`` sits tens of millimetres from where
    its raw bounds claim — framing on those aims the camera at empty
    space. Structures are the fallback; the whole scene is not, because
    the cone and manipulator bodies dwarf the brain.
    """
    actor = backend.actor("brain")
    if actor is not None and actor.visibility:
        box = _union([actor.GetBounds()])
        if box is not None:
            return box
    return node_bounds(backend, adapter, "structure")


def clip_actors_above(
    backend: PyVistaBackend, adapter: RendererAdapter, z_mm: float, tags: set[str]
) -> int:
    """Clip tagged geometry above ``z_mm`` (LPS +z is Superior).

    Manipulator bodies run ~160 mm above the brain, so any camera framed
    on a 20 mm brain sits *inside* them and the panel fills with hardware.
    Cutting them at the implant leaves the shanks entering the bores,
    which is the part a reader needs to see.
    """
    clipped = 0
    for node in adapter.scene.nodes.values():
        if not (node.tags & tags):
            continue
        actor = backend.actor(node.key)
        if actor is None or not actor.visibility:
            continue
        vtk_plane = pv._vtk.vtkPlane()
        vtk_plane.SetOrigin(0.0, 0.0, z_mm)
        vtk_plane.SetNormal(0.0, 0.0, -1.0)  # keep the half-space below z
        actor.mapper.AddClippingPlane(vtk_plane)
        clipped += 1
    return clipped


def frame(
    plotter: pv.Plotter,
    bounds: tuple[float, ...],
    view: ViewName,
    *,
    zoom: float,
    orbit: tuple[float, float],
    parallel: bool,
    pan: tuple[float, float, float] = (0.0, 0.0, 0.0),
    dorsal_tilt: float = 0.45,
) -> None:
    """Point the camera at ``bounds`` from the named direction and fit."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    centroid = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0])
    diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))

    direction, up = VIEWS[view]
    if view == "dorsal":
        direction = (0.0, dorsal_tilt, 1.0)
    offset = np.asarray(direction, dtype=np.float64)
    offset = offset / np.linalg.norm(offset) * diag

    if parallel:
        plotter.enable_parallel_projection()

    cam = plotter.camera
    cam.focal_point = tuple(centroid)
    cam.position = tuple(centroid + offset)
    cam.up = up

    # Fit to the requested bounds, not the whole scene: the headframe and
    # cone stack are far bulkier than the brain and would shrink it.
    plotter.reset_camera(bounds=(xmin, xmax, ymin, ymax, zmin, zmax))

    azimuth, elevation = orbit
    if azimuth:
        cam.azimuth = azimuth
    if elevation:
        cam.elevation = elevation
    if zoom != 1.0:
        cam.zoom(zoom)
    if any(pan):
        shift = np.asarray(pan, dtype=np.float64)
        cam.position = tuple(np.asarray(cam.position) + shift)
        cam.focal_point = tuple(np.asarray(cam.focal_point) + shift)


def write(plotter: pv.Plotter, args: Args, path: Path) -> tuple[float, float] | None:
    """Write one panel, returning its frame fill where that is measurable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.format != "png":
        if args.border_pt > 0:
            print("  note: --border-pt is PNG only; no rule drawn on vector output")
        plotter.save_graphic(str(path))
        return None

    plotter.image_scale = args.scale
    plotter.screenshot(str(path), transparent_background=args.transparent)
    target = print_size(args)
    if target is not None:
        resample_to(path, target, args.dpi)
    # Measured before the border, which touches all four edges and would
    # otherwise report every panel as perfectly filled.
    fill = frame_fill(path)
    if args.border_pt > 0:
        draw_border(path, args)
    return fill


# --------------------------------------------------------------------------


def line_px(width_pt: float, args: Args) -> float:
    """Stroke width in render pixels for a width in printed points.

    ``image_scale`` supersamples by tiling the viewport, and VTK keeps line
    widths in device pixels across tiles: a stroke set to *w* comes out *w*
    wide in the supersampled image and *w*/``scale`` wide once that image is
    resampled to print size. Pre-multiplying by ``scale`` cancels it.
    """
    return width_pt / 72.0 * args.dpi * args.scale


def add_brain_silhouette(
    plotter: pv.Plotter, backend: PyVistaBackend, args: Args
) -> None:
    """Outline the brain against the camera set by :func:`frame`.

    The filter silhouettes in the polydata's own frame, so the brain's
    ``user_matrix`` has to be baked into the geometry first — the brain
    carries ``headframe_to_lps`` and the outline would otherwise be drawn
    tens of millimetres from the shell it belongs to.
    """
    import vtk

    actor = backend.actor("brain")
    if actor is None or not actor.visibility:
        return

    poly = actor.GetMapper().GetInput()
    if args.brain_silhouette_smooth:
        # The mapper input is already polydata; wrap is a view, not a copy,
        # so smooth into a new mesh and leave the rendered shell alone.
        poly = pv.wrap(poly).smooth_taubin(n_iter=args.brain_silhouette_smooth)
    matrix = actor.GetUserMatrix()
    if matrix is not None:
        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)
        baked = vtk.vtkTransformPolyDataFilter()
        baked.SetTransform(transform)
        baked.SetInputData(poly)
        baked.Update()
        poly = baked.GetOutput()

    silhouette = vtk.vtkPolyDataSilhouette()
    silhouette.SetInputData(poly)
    silhouette.SetCamera(plotter.renderer.GetActiveCamera())
    silhouette.SetEnableFeatureAngle(False)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(silhouette.GetOutputPort())
    outline = vtk.vtkActor()
    outline.SetMapper(mapper)
    prop = outline.GetProperty()
    prop.SetColor(*pv.Color(args.brain_silhouette_color).float_rgb)
    prop.SetLineWidth(line_px(args.brain_silhouette_pt, args))
    prop.SetLighting(False)
    plotter.renderer.AddActor(outline)


def draw_border(path: Path, args: Args) -> None:
    """Stroke a rule round the written panel, flush with its edge."""
    from PIL import Image, ImageDraw

    width = max(1, round(args.border_pt / 72.0 * args.dpi))
    with Image.open(path) as opened:
        img = opened.copy()
        dpi = opened.info.get("dpi")

    rgba = img.convert("RGBA")
    # PIL grows the outline inward from the box, so the box is the panel edge.
    ImageDraw.Draw(rgba).rectangle(
        (0, 0, rgba.width - 1, rgba.height - 1),
        outline=pv.Color(args.border_color).hex_rgb,
        width=width,
    )
    out = rgba if img.mode == "RGBA" else rgba.convert(img.mode)
    if dpi:
        out.save(path, dpi=dpi)
    else:
        out.save(path)


def frame_fill(path: Path) -> tuple[float, float] | None:
    """How much of the panel the content uses: bounding box, then ink.

    A pose framed against the whole scene can leave a third of the panel as
    margin, which is free on a slide and expensive at 57 mm. The bounding
    box is the contract's measure, but it saturates the moment anything
    reaches an edge — which splayed manipulators do in every framing of
    this scene — so the second number, the fraction of pixels that are
    content at all, is the one that moves. Needs a transparent render;
    alpha is what says where the content is.
    """
    from PIL import Image

    with Image.open(path) as opened:
        if opened.mode != "RGBA":
            return None
        alpha = np.asarray(opened.copy())[..., 3]

    ink = alpha > 0
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return 0.0, 0.0
    box = (rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1)
    return float(box) / float(alpha.size), float(ink.mean())


def _size_note(args: Args) -> str:
    """Describe the written size the way a layout program will see it."""
    w, h = render_size(args)
    if args.width_mm is None:
        return f"{w * args.scale}x{h * args.scale}px"
    if args.format != "png":
        return f"{w}x{h}pt {args.width_mm:g}x{args.height_mm:g}mm"
    return f"{w}x{h}px {args.width_mm:g}x{args.height_mm:g}mm@{args.dpi}dpi"


def render_panel(args: Args, panel: str) -> Path:
    """Build, style, frame, and write a single panel."""
    implant_opacity = (
        args.implant_opacity if panel == "implant" else args.implant_opacity * 0.6
    )
    plotter, backend, adapter, meta = build_scene(args)
    style_scene(backend, adapter, args, implant_opacity=implant_opacity)

    if args.clip_above is not None:
        clip_actors_above(
            backend, adapter, args.clip_above, {"probe", "fixture", "headframe", "cone"}
        )

    if args.transparent:
        plotter.set_background("white")
    else:
        plotter.set_background(args.background)
    if args.anti_alias:
        plotter.enable_anti_aliasing("ssaa")

    if panel == "implant":
        # Implant plus target structures: the bore plate alone is ~6x10 mm
        # and frames so tightly the shanks leave the panel before reaching
        # anything. Including the targets shows the threading and where it
        # ends up, which is the whole point of this panel.
        boxes = [
            b
            for b in (
                node_bounds(backend, adapter, "implant"),
                node_bounds(backend, adapter, "structure"),
            )
            if b is not None
        ]
        bounds = _union(boxes) or brain_bounds(backend, adapter)
        view, zoom = args.implant_view, args.zoom * args.implant_zoom
    else:
        if args.frame_on == "scene":
            bounds = scene_bounds(backend, adapter)
        elif args.frame_on == "implant":
            bounds = node_bounds(backend, adapter, "implant")
        else:
            bounds = brain_bounds(backend, adapter)
        bounds = bounds or brain_bounds(backend, adapter)
        view, zoom = args.view, args.zoom

    if bounds is not None and args.extend_up:
        bounds = (*bounds[:5], bounds[5] + args.extend_up)

    if bounds is None:
        plotter.reset_camera()
    else:
        frame(
            plotter,
            bounds,
            view,
            zoom=zoom,
            orbit=args.orbit,
            parallel=args.parallel,
            pan=args.pan,
            dorsal_tilt=args.dorsal_tilt,
        )

    # After framing: the silhouette is computed against the final camera.
    if args.brain_silhouette_pt > 0:
        add_brain_silhouette(plotter, backend, args)

    path = args.out.with_name(f"{args.out.name}-{panel}.{args.format}")
    fill = write(plotter, args, path)
    plotter.close()

    if not args.quiet:
        print(
            f"  {panel:<9} {path}  "
            f"{_size_note(args)} "
            f"{'' if fill is None else f'bbox={fill[0]:.0%} ink={fill[1]:.0%} '}"
            f"view={view} probes={len(meta['probes'])} bounds="
            + (
                "scene"
                if bounds is None
                else f"{bounds[1] - bounds[0]:.1f}x"
                f"{bounds[3] - bounds[2]:.1f}x{bounds[5] - bounds[4]:.1f}mm"
            )
        )
    return path


def main(args: Args) -> None:
    """Render the requested panels."""
    ensure_gl_context()

    if not args.quiet:
        print(f"config   {args.config}")
        print(f"plan     {args.plan}")
        print(f"revision {git_revision()}")

    panels = ("overview", "implant") if args.panel == "both" else (args.panel,)
    for panel in panels:
        render_panel(args, panel)


if __name__ == "__main__":
    main(tyro.cli(Args))
