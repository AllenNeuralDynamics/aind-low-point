Architecture Guide
==================

This guide describes the internal architecture of aind-rutter for developers
who want to understand or extend the codebase.

High-Level Overview
-------------------

aind-rutter follows a layered architecture with clear separation between:

1. **Configuration Layer** - Declarative YAML-based specification
2. **Build Layer** - Transform config into runtime objects
3. **Domain Layer** - Core data structures and business logic
4. **Adapter Layer** - Connect domain to external systems (rendering, collision)
5. **Controller Layer** - Handle user interaction and state changes

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────┐
    │                     Configuration (YAML)                     │
    │  ConfigModel → Templates → Bulk Expansion → Validation       │
    └──────────────────────────┬──────────────────────────────────┘
                               │ build_runtime_from_config()
                               ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      RuntimeBundle                           │
    │  ┌─────────────┐  ┌─────────┐  ┌──────────────┐            │
    │  │AssetCatalog │  │  Scene  │  │PlanningState │            │
    │  │ assets{}    │  │ nodes{} │  │ probes{}     │            │
    │  │ targets{}   │  │         │  │ kinematics   │            │
    │  └─────────────┘  └─────────┘  └──────────────┘            │
    └─────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │RendererAdapt │  │CollisionAdapt│  │  PlanStore   │
    │  (PyVista)   │  │   (FCL)      │  │  (Redux-ish) │
    └──────────────┘  └──────────────┘  └──────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Frontend  (TrameController for the web app)                 │
    └─────────────────────────────────────────────────────────────┘


Module Organization
-------------------

.. code-block:: text

    src/aind_rutter/
    ├── domain/                # the planning domain (imports no config, optimizer or UI)
    │   ├── transforms.py      # AffineTransform, TransformChain, *Transformable
    │   ├── enums.py           # Role, Kind, MRSignal, OrientationCode; KNOWN_SCENE_TAGS
    │   ├── catalog.py         # AssetSpec, TargetSpec, AssetCatalog, Material
    │   ├── scene.py           # NodeInstance, Scene
    │   ├── rig.py             # JointRange, PoseLimits, Kinematics, AP/ML_LIMIT_DEG
    │   ├── plan.py            # ProbePlan, PlanningState, resolve_target_LPS
    │   ├── pose.py            # ProbePose, PoseResolver, shank-tip detection
    │   ├── probe_kinds.py     # RecordingGeometry per probe kind
    │   └── commands.py        # planning commands + apply_planning_command
    ├── config/                # the config DSL, split by what each model describes
    │   ├── models_common.py   # sources, materials, transforms, imaging, paths
    │   ├── models_catalog.py  # asset and target specs, single and bulk
    │   ├── models_scene.py    # SceneNodeModel, SceneModel
    │   ├── models_plan.py     # arcs, probes, calibrations, head mount
    │   ├── templates.py       # the template merge
    │   ├── resolve.py         # effective canonicalization and transform
    │   └── root.py            # ConfigModel and its cross-reference validation
    ├── build/                 # config → runtime
    │   ├── assemble.py        # build_runtime_from_config
    │   ├── loaders.py         # the loader registry
    │   ├── reducers.py        # the reducer registry
    │   ├── canonicalize.py    # orientation, scale and inline transform
    │   ├── chem_shift.py      # the per-ppm correction
    │   ├── calibration.py     # the bank plus the NewScale frame
    │   ├── queries.py         # head_pitch_deg_*, fixture sets, brain_world_mesh
    │   ├── transforms.py      # transform-reference resolution
    │   └── probe_context.py   # the per-probe view the optimizer reads
    ├── plan_io/               # reading a plan back out
    │   ├── roundtrip.py       # planning_state_to_plan_model, save_plan_to_config
    │   ├── replay.py          # apply_plan_model_to_state
    │   ├── rig_export.py      # export_plan_geometry, reorder_plan_for_rig
    │   └── cli_csv.py         # the rutter-plan-csv entry point
    ├── session/
    │   ├── store.py           # PlanStore: one planning state, edited by command
    │   └── worker.py          # AsyncLatestWorker
    ├── render/
    │   ├── overlays.py        # OverlaySpec, OverlayState, OverlayResolver
    │   ├── adapter.py         # RendererAdapter, RenderBackend protocol
    │   └── pyvista.py         # PyVistaBackend + DebouncedFlush
    ├── collision/
    │   ├── geometry.py        # FCL BVH and transform construction, with guards
    │   ├── adapter.py         # CollisionAdapter, pair_bits (the pair rule)
    │   ├── fcl.py             # FCLBackend (per-pair callback)
    │   └── worker.py          # CollisionHandler (sync + async paths)
    ├── web/
    │   ├── cli.py             # the rutter-plan entry point
    │   ├── app.py             # build_trame_app() factory
    │   ├── controller.py      # TrameController: state, wiring, intent → command
    │   ├── layout.py          # the Vuetify page: tabs, sliders, dialogs
    │   ├── readouts.py        # tip, depth, over-insertion, collisions, NewScale
    │   ├── materials.py       # warning overlays, visibility groups, opacities
    │   ├── camera.py          # framing, click-to-pick, probe highlight
    │   ├── keybindings.py     # which key means which action, and how fast
    │   └── ccf_regions.py     # CCFOverlayManager (lazy region meshes)
    ├── ccf/
    │   └── ontology.py        # Allen CCF structures + search
    └── optimization/          # the placement optimizer; see the optimizer guide
        ├── assignment/        # which probe goes where
        ├── geometry/          # holes, headstages, probes, kinematics
        ├── clearance/         # poses, smooth math, SDF grids, boxes, pair terms
        ├── objectives/        # the soft and constrained objectives
        ├── search/            # spin restore and the batched minimizers
        ├── validation/        # the FCL ground-truth check
        └── pipeline/          # the offline batch flow: phase1, phase2, emit

``build/__init__`` exports only ``build_runtime_from_config`` and
``RuntimeBundle``; everything else comes from its submodule.


Layer rules
-----------

Five rules hold over that tree, and ``tests/architecture/`` enforces each
against the import graph rather than against a convention. Every rule carries a
baseline of today's exceptions that may only shrink: fixing one means deleting
its line, and a rule fails when a listed exception stops violating it, so the
lists cannot rot quietly.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Rule
     - Enforced by
   * - No import cycles between modules.
     - ``test_layers.py``, ``CYCLES_BASELINE``
   * - The solver never imports the pipeline that drives it.
     - ``test_layers.py``, ``SOLVER_TO_PIPELINE_BASELINE``
   * - No module reaches into another module's private names.
     - ``test_layers.py``, ``PRIVATE_IMPORT_BASELINE``
   * - Nothing below the CLI reads the environment at import time, apart from
       the JAX platform and allocator variables, which must precede the JAX
       import.
     - ``test_environment.py``, ``ENV_READ_BASELINE``
   * - The planner imports without JAX installed, so the viewer runs without
       the optimizer stack.
     - ``test_jax_free.py``
   * - All five console scripts import and resolve.
     - ``test_console_scripts.py``

``domain/`` is the load-bearing one. It imports no config, no optimizer and no
UI, which is what lets the same probe geometry serve the app and the solver
without the app pulling in JAX.


Core Abstractions
-----------------

Transforms (``domain/transforms.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transform system provides immutable, composable 3D transforms.

**AffineTransform**

Represents a single rigid transform (rotation + translation):

.. code-block:: python

    @dataclass(frozen=True)
    class AffineTransform:
        rotation: Float3x3      # (3,3) rotation matrix
        translation: Float3     # (3,) translation vector
        inverted: bool = False  # lazily invert on application

        def apply_to(self, pts: FloatNx3) -> FloatNx3: ...
        def invert(self) -> AffineTransform: ...

        @classmethod
        def from_sitk_path(cls, path: Path) -> AffineTransform: ...

**TransformChain**

Composes multiple transforms with lazy evaluation:

.. code-block:: python

    @dataclass(frozen=True)
    class TransformChain:
        elements: Tuple[AffineTransform, ...]

        @cached_property
        def composed_transform(self) -> Tuple[Float3x3, Float3]:
            # Compose all elements into single R, t
            ...

        def apply_to(self, pts: FloatNx3) -> FloatNx3: ...
        def invert(self) -> TransformChain: ...

Geometry Wrappers (``domain/transforms.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Protocol-based polymorphism for transformable geometry:

.. code-block:: python

    @runtime_checkable
    class SupportsRigidTransform(Protocol[RawT_co]):
        @property
        def raw(self) -> RawT_co: ...
        def transformed(self, R, t) -> RawT_co: ...

    @dataclass(frozen=True, slots=True)
    class MeshTransformable(SupportsRigidTransform[trimesh.Trimesh]):
        _raw: trimesh.Trimesh
        ...

    @dataclass(frozen=True, slots=True)
    class PointsTransformable(SupportsRigidTransform[FloatNx3]):
        _raw: FloatNx3  # (N, 3)
        ...

**Transformed[W, RawT_co]**

Binds geometry to a transform chain:

.. code-block:: python

    @dataclass(frozen=True)
    class Transformed(Generic[W, RawT_co]):
        original: W  # e.g., MeshTransformable
        chain: TransformChain

        @cached_property
        def raw(self) -> RawT_co:
            # Returns transformed geometry
            R, t = self.chain.composed_transform
            return self.original.transformed(R, t)

Material (``domain/catalog.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple value object for rendering properties:

.. code-block:: python

    @dataclass(frozen=True, slots=True)
    class Material:
        name: str
        color_hex_str: str = "#C8C8C8"
        opacity: float = 1.0
        wireframe: bool = False
        visible: bool = True


Asset Catalog (``domain/catalog.py``)
-------------------------------------

The catalog holds all loaded geometry and metadata.

BaseSpec
~~~~~~~~

Common fields for all catalog items:

.. code-block:: python

    @dataclass(frozen=True)
    class BaseSpec:
        key: str                     # Unique identifier
        kind: Literal["mesh", "points", "lines"]
        role: Role                   # ANATOMY, TARGET, LANDMARK, GEOMETRY
        default_material: Material
        metadata: dict[str, Any]
        tags: set[str]

        # Collision
        collidable: bool             # Whether it gets an FCL body

        # UI hints
        pivot_LPS: Optional[Float3]  # Rotation center
        bbox_hint: Optional[FloatAABB]

AssetSpec
~~~~~~~~~

Extends ``BaseSpec`` with concrete geometry:

.. code-block:: python

    @dataclass(frozen=True)
    class AssetSpec(BaseSpec):
        source_path: Optional[Path]
        loader: Optional[str]

        # Loaded geometry (post-canonicalization, in LPS mm)
        mesh: Optional[MeshTransformable]
        points: Optional[PointsTransformable]

TargetSpec
~~~~~~~~~~

Specialized for probe insertion targets:

.. code-block:: python

    @dataclass(frozen=True)
    class TargetSpec(BaseSpec):
        kind: Literal["points"] = "points"
        role: Role = Role.TARGET

        source_path: Optional[Path]      # Explicit file
        source_key: Optional[str]        # Derive from another asset
        reducer: Optional[str]           # e.g., "centroid"

        points: Optional[PointsTransformable]
        approach_vector: Optional[Float3]
        uncertainty_mm: Optional[float]

AssetCatalog
~~~~~~~~~~~~

Container for all assets and targets:

.. code-block:: python

    @dataclass(frozen=True, slots=True)
    class AssetCatalog:
        assets: dict[str, AssetSpec]
        targets: dict[str, TargetSpec]

        def get_spec(self, key: str) -> Union[AssetSpec, TargetSpec]: ...
        def get_geometry(self, key: str) -> Union[MeshTransformable, PointsTransformable]: ...


Scene Graph (``domain/scene.py``)
---------------------------------

The scene graph places catalog items in 3D space.

NodeInstance
~~~~~~~~~~~~

A placed instance of an asset:

.. code-block:: python

    @dataclass(slots=True)
    class NodeInstance:
        key: str                    # Unique node ID (e.g., "probe:probe_A")
        asset_key: str              # Reference to catalog
        transform: TransformChain   # World placement
        tags: Set[str]              # Filtering/grouping
        material_override: Optional[Material]
        enabled: bool

        # Per-instance state
        locked_axes: Set[str]       # {"ap_tilt", "ml_tilt", ...}
        extras: dict[str, Any]      # e.g., {"pose_source_probe": "probe_A"}

Scene
~~~~~

Container for all nodes:

.. code-block:: python

    @dataclass(slots=True)
    class Scene:
        nodes: dict[str, NodeInstance]

        def upsert(self, node: NodeInstance): ...
        def remove(self, node_id: str): ...
        def by_tag(self, tag: str) -> list[NodeInstance]: ...


Planning Domain (``domain/plan.py``, ``pose.py``, ``rig.py``)
-------------------------------------------------------------

The planning domain handles probe kinematics and targeting.

ProbePlan
~~~~~~~~~

Declarative specification of a probe's intended state:

.. code-block:: python

    @dataclass(slots=True)
    class ProbePlan:
        kind: str              # e.g., "neuropixels"
        arc_id: Optional[str]  # Which arc this probe belongs to

        # Angle control
        bind_ap_to_arc: bool   # AP comes from arc angle?
        ap_local: float        # Per-probe AP override
        ml_local: float        # Per-probe ML angle
        spin: float            # Axial rotation

        # Targeting
        target_key: Optional[str]
        target_point_RAS: Optional[Tuple[float, float, float]]
        past_target_mm: float
        offsets_RA: Tuple[float, float]

        calibrated: bool       # Use calibration for angles?

ProbePose
~~~~~~~~~

Resolved runtime pose (angles + tip position):

.. code-block:: python

    @dataclass(slots=True)
    class ProbePose:
        ap: float              # Resolved AP angle (deg)
        ml: float              # Resolved ML angle (deg)
        spin: float            # Spin angle (deg)
        tip: NDArray           # (3,) tip position in LPS

        def transform(self) -> AffineTransform: ...
        def chain(self) -> TransformChain: ...

        @classmethod
        def from_planning_state(cls, ps: PlanningState, probe_name: str) -> ProbePose:
            # Resolves angles from calibration/arc/local
            # Computes tip from target + offsets + past_target_mm
            ...

Kinematics
~~~~~~~~~~

Rig-wide parameters shared by all probes:

.. code-block:: python

    @dataclass(slots=True)
    class Kinematics:
        arc_angles: dict[str, float]  # arc_id → AP angle
        limits: PoseLimits            # Joint limits
        coupled_axes: Set[str]        # Which DOFs are arc-coupled

PlanningState
~~~~~~~~~~~~~

Complete planning state container:

.. code-block:: python

    @dataclass(slots=True)
    class PlanningState:
        kinematics: Kinematics
        probes: dict[str, ProbePlan]
        calibrations: dict[str, AffineTransform]
        target_index: dict[str, Float3]  # key → LPS position

PoseResolver
~~~~~~~~~~~~

Computes world transforms for scene nodes:

.. code-block:: python

    @dataclass
    class PoseResolver:
        scene: Scene
        plan: PlanningState
        get_pivot_for_asset: Callable[[str], Optional[np.ndarray]]

        def world_chain_for_node(self, node: NodeInstance) -> TransformChain:
            # Composes: base transform + dynamic probe pose
            base = node.transform
            dyn = self._dynamic_chain_for_node(node)
            return TransformChain.new([*base.elements, *dyn.elements])


Build Runtime (``build/assemble.py``)
-------------------------------------

Transforms configuration into runtime objects.

Loader Registry
~~~~~~~~~~~~~~~

Extensible geometry loading system:

.. code-block:: python

    @register_loader
    def trimesh(path: Path) -> trimesh.Trimesh:
        return trimesh.load(str(path))

    @register_loader
    def sitk_volume(path: Path) -> trimesh.Trimesh:
        # Load SITK image, extract surface mesh
        ...

    @register_loader
    def numpy_points(path: Path) -> np.ndarray:
        return np.load(path)

    # Usage
    geometry = load_geometry("/data/brain.obj", "trimesh")

Reducer Registry
~~~~~~~~~~~~~~~~

Target point reduction:

.. code-block:: python

    @register_reducer
    def mesh_centroid(source: SourceGeo) -> np.ndarray:
        if isinstance(source, trimesh.Trimesh):
            return np.array(source.centroid)
        raise TypeError(...)

    @register_reducer
    def mesh_center_mass(source: SourceGeo) -> np.ndarray:
        ...

Canonicalization Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

Transforms geometry from file coordinates to canonical LPS:

.. code-block:: python

    @dataclass(frozen=True)
    class CanonicalizationRuntime:
        source_space: SourceSpace     # e.g., "RAS", "ASR", "FILE_NATIVE"
        scale_to_mm: float            # Unit conversion
        transform_file_to_canonical: Optional[AffineTransform]

Chemical Shift Correction
~~~~~~~~~~~~~~~~~~~~~~~~~

MRI-specific coordinate correction:

.. code-block:: python

    @dataclass(frozen=True)
    class ChemShiftContext:
        enabled: bool
        magnet_MHz: float
        default_ppm: float
        apply_by_role: set[Role]
        image: Optional[sitk.Image]

        def pt_transform_for_ppm(self, ppm: float) -> AffineTransform:
            # Returns correction transform for given chemical shift
            ...

RuntimeBundle
~~~~~~~~~~~~~

The complete built runtime:

.. code-block:: python

    @dataclass
    class RuntimeBundle:
        catalog: AssetCatalog
        scene: Scene
        plan: PlanningState
        label_index: CollisionLabelIndex

    def build_runtime_from_config(cfg: ConfigModel) -> RuntimeBundle:
        # 1. Compile collision labels
        # 2. Build assets (load, canonicalize, chem-shift)
        # 3. Load resources
        # 4. Build targets (derive or load)
        # 5. Build scene nodes
        # 6. Build planning state (kinematics, calibrations, probes)
        return RuntimeBundle(...)


Adapters
--------

Adapters connect the domain to external systems.

RendererAdapter (``render/adapter.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bridges domain objects to a render backend. Pushes a 4×4
``model_matrix`` per node so the renderer applies the transform on the GPU
side; the underlying vertex buffers stay in canonical LPS layout:

.. code-block:: python

    @dataclass
    class RendererAdapter:
        plot: pv.Plotter
        catalog: AssetCatalog
        scene: Scene
        overlays: Optional[OverlayResolver]

        def rebuild(self, plan: PlanningState) -> None:
            # Rebuild all nodes in the plot
            resolver = self._make_resolver(plan)
            for node in self.scene.nodes.values():
                self._upsert_node(node, resolver)

        def on_store_change(self, plan: PlanningState, changed: List[str]) -> None:
            # Update only changed probe nodes
            resolver = self._make_resolver(plan)
            for probe_name in changed:
                node = self.scene.nodes.get(f"probe:{probe_name}")
                if node:
                    self._upsert_node(node, resolver)

**Overlay System**

Overlays modify node appearance (e.g., collision highlighting):

.. code-block:: python

    @dataclass(frozen=True)
    class OverlaySpec:
        color: int          # 0xRRGGBB
        alpha: float        # Blend strength
        blend: BlendMode    # "replace", "alpha_over", ...
        priority: int       # Higher wins
        source: str         # "collision", "hover", "selection"

    @dataclass(slots=True)
    class OverlayState:
        by_node: dict[str, List[OverlaySpec]]

        def set_for_source(self, node_ids: list[str], spec: OverlaySpec): ...
        def clear_source(self, source: str): ...

CollisionAdapter (``collision/adapter.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bridges domain to FCL collision detection:

.. code-block:: python

    class CollisionBackend(Protocol):
        def rebuild(self, specs: Iterable[ObjSpec]) -> None: ...
        def sync(self, specs: Iterable[ObjSpec]) -> None: ...
        def collide_internal(self, **kwargs) -> List[CollisionPair]: ...

    @dataclass
    class CollisionAdapter:
        backend: CollisionBackend
        scene: Scene
        assets: AssetCatalog
        include: Callable[[NodeInstance, AssetCatalog], bool]

        def rebuild(self, plan: PlanningState) -> None:
            # Build collision geometry for all collidable nodes
            ...

        def on_store_change(self, plan: PlanningState, changed: List[str]) -> None:
            # Update collision geometry for changed probes
            ...

        def collide_internal(self, **kwargs) -> List[CollisionPair]:
            # Run collision detection, return pairs
            ...


State Management (``session/store.py``)
---------------------------------------

Redux-inspired unidirectional data flow.

PlanStore
~~~~~~~~~

Central state container with subscription:

.. code-block:: python

    Subscriber = Callable[[PlanningState, List[str]], None]

    class PlanStore:
        def __init__(self, initial: PlanningState): ...

        @property
        def state(self) -> PlanningState: ...

        def subscribe(self, fn: Subscriber) -> Callable[[], None]:
            # Returns unsubscribe function
            ...

        def dispatch(self, cmd: PlanningCommand) -> None:
            # Apply command, notify subscribers
            changed = apply_planning_command(self._state, cmd)
            self._notify(changed)

AsyncLatestWorker
~~~~~~~~~~~~~~~~~

Runs an expensive subscriber (collision detection) off the main thread with
latest-only semantics — rapid updates collapse into a single in-flight
request:

.. code-block:: python

    class AsyncLatestWorker:
        def __init__(
            self,
            prepare: Callable[[PlanningState, List[str]], Any],
            work: Callable[[Any], Any],
            deliver: Callable[[Any], None],
            post_to_main: Callable[[Callable[[], None]], None],
        ) -> None: ...

        def __call__(self, plan: PlanningState, changed_ids: List[str]) -> None:
            # main thread: capture latest state, kick worker if idle
            ...

        def shutdown(self) -> None: ...

``prepare`` runs on the main thread (so it can read PlanningState safely),
``work`` runs on a dedicated worker thread (the FCL update + collision
query), and ``deliver`` is posted back to the main thread via
``post_to_main`` (typically ``asyncio.AbstractEventLoop.call_soon_threadsafe``
under trame).

Command Pattern (``domain/commands.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Commands encapsulate state mutations. They are frozen dataclasses; a single
``Union`` (``PlanningCommand``) covers all of them:

.. code-block:: python

    @dataclass(frozen=True)
    class SetProbeLocalAngles:
        name: str
        ap_local: Optional[float] = None
        ml_local: Optional[float] = None
        spin: Optional[float] = None

    @dataclass(frozen=True)
    class SetArcAngle:
        arc_id: str
        ap_deg: float

    @dataclass(frozen=True)
    class SetProbePastTarget:
        name: str
        past_target_mm: float

    PlanningCommand = Union[
        SetProbeLocalAngles, SetProbeOffsetsRA, NudgeProbeOffsetsRA,
        SetProbePastTarget, NudgeProbePastTarget, SetProbeTarget,
        SetArcAngle, AssignProbeArc, BindProbeAPToArc, SetProbeCalibrated,
    ]

    def apply_planning_command(
        state: PlanningState, cmd: PlanningCommand
    ) -> List[str]:
        # Mutate state, return list of affected probe names
        ...


Frontend
--------

Trame web app (``web/controller.py`` + ``web/app.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TrameController`` — Vuetify3 layout with PyVista 3D view via
``pyvista.trame.ui.plotter_ui``. Includes optional Save-YAML and CCF region
overlay controls.

``build_trame_app(cfg, *, ccf_volume=None, save_path=None) -> Server`` is the
factory: wires runtime, store, render adapter, async collision worker,
overlay state, debounced flush, optional CCF overlay manager, and returns a
ready-to-``server.start()`` trame server.

Event flow (both frontends)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. User moves slider → controller handler
2. Controller dispatches a command → ``store.dispatch(SetProbeLocalAngles(...))``
3. Store applies command, notifies subscribers synchronously
4. ``RenderHandler`` (sync) → ``RendererAdapter`` updates the model_matrix
   on changed nodes
5. ``AsyncLatestWorker`` (off-thread) wraps ``CollisionHandler``
   prepare/work/deliver — recomputes collisions and posts overlay updates
   back to the main thread
6. ``OverlayResolver`` folds collision overlays into the next render pass


Data Flow Summary
-----------------

.. code-block:: text

    Configuration File (YAML)
            │
            ▼
    ┌───────────────┐
    │  ConfigModel  │  Pydantic parsing & validation
    │   (config/)   │  Template expansion, auto-inference
    └───────┬───────┘
            │ build_runtime_from_config()
            ▼
    ┌───────────────┐
    │RuntimeBundle  │  Loaded geometry, resolved transforms
    │   (build/)    │  AssetCatalog, Scene, PlanningState
    └───────┬───────┘
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
    ┌─────────┐  ┌─────────┐
    │PlanStore│  │Adapters │
    │ (state) │  │(render, │
    │         │  │ collide)│
    └────┬────┘  └────┬────┘
         │            │
         │ dispatch() │ on_store_change()
         │            │
         ▼            ▼
    ┌─────────────────────┐
    │     Controller      │
    │  (widgets, events)  │
    └─────────────────────┘


Extension Points
----------------

Adding a New Loader
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from aind_rutter.build.loaders import register_loader

    @register_loader("my_custom_loader")
    def my_loader(path: Path, **kwargs) -> trimesh.Trimesh:
        # Custom loading logic
        return my_mesh

Then use in config:

.. code-block:: yaml

    assets:
      - key: custom
        src: /data/file.custom
        loader: my_custom_loader

Adding a New Reducer
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from aind_rutter.build.reducers import register_reducer

    @register_reducer("bbox_center")
    def bbox_center(source: SourceGeo) -> np.ndarray:
        if isinstance(source, trimesh.Trimesh):
            return source.bounds.mean(axis=0)
        raise TypeError(...)

Then use in config:

.. code-block:: yaml

    targets:
      - key: target:center
        source_key: my_mesh
        reducer: bbox_center

Adding a New Command
~~~~~~~~~~~~~~~~~~~~

1. Define the command in ``commands.py``:

.. code-block:: python

    @dataclass
    class MyNewCommand:
        probe_name: str
        new_value: float

2. Handle it in ``apply_planning_command()``:

.. code-block:: python

    def apply_planning_command(state, cmd) -> List[str]:
        if isinstance(cmd, MyNewCommand):
            probe = state.probes[cmd.probe_name]
            probe.some_field = cmd.new_value
            return [cmd.probe_name]
        ...


Testing Strategy
----------------

The codebase uses pytest with factory helpers:

- ``tests/config_factories.py`` - Generate test configurations
- ``tests/test_config_*.py`` - Configuration parsing/validation
- ``tests/conftest.py`` - Shared fixtures

Example test pattern:

.. code-block:: python

    def test_asset_loading():
        config_data = ConfigFactory.minimal_config()
        config_data["assets"] = [
            AssetFactory.mesh_asset(key="test", src="/data/test.obj")
        ]

        config = ConfigModel.model_validate(config_data)
        bundle = build_runtime_from_config(config)

        assert "test" in bundle.catalog.assets
        assert bundle.catalog.assets["test"].mesh is not None


Key Design Decisions
--------------------

**Immutable Core Objects**

``AffineTransform``, ``TransformChain``, ``AssetSpec``, etc. are frozen dataclasses.
This enables safe sharing, caching, and simpler reasoning.

**Protocol-Based Polymorphism**

``SupportsRigidTransform`` allows any geometry type to participate in the
transform system without inheritance.

**Registry Pattern for Extensibility**

Loaders and reducers use function registries, allowing users to add custom
implementations without modifying core code.

**Unidirectional Data Flow**

State changes flow: Command → Store → Subscribers (Adapters, UI).
This prevents feedback loops and makes debugging easier.

**Lazy Evaluation**

``@cached_property`` on ``TransformChain.composed_transform`` and
``Transformed.raw`` defers expensive computations until needed.

**Separation of Concerns**

- Config models handle parsing/validation (Pydantic)
- Runtime specs hold loaded data (dataclasses)
- Adapters translate between domain and external systems
- Controllers manage UI state
