Configuration Guide
===================

This guide covers the configuration system for aind-low-point. The configuration
defines assets, targets, transforms, and planning parameters for probe placement
experiments.

Overview
--------

Configuration files are YAML documents parsed into a ``ConfigModel``. The system
supports hierarchical configuration with templates, bulk declarations, and
automatic inference to reduce boilerplate.

A minimal configuration:

.. code-block:: yaml

    version: 1

    assets:
      - key: brain
        src: /data/brain.obj

    targets:
      - key: target:PL
        source_key: brain

This minimal example leverages auto-inference:

- ``kind: mesh`` and ``loader: trimesh`` are inferred from the ``.obj`` extension
- ``role: anatomy`` is inferred from the ``brain`` key prefix
- ``role: target`` is inferred from the ``target:`` prefix

Configuration Structure
-----------------------

The root ``ConfigModel`` contains these sections:

.. code-block:: yaml

    version: 1                    # Schema version (required)

    # Path interpolation helpers
    paths:
      data_root: /path/to/data
      mouse_id: "12345"

    # Imaging parameters (MRI-specific)
    imaging:
      magnet_frequency_MHz: 9.4
      chem_shift_ppm_default: 3.7

    # Reusable material definitions
    materials:
      brain_material:
        color: "#FFB6C1"
        opacity: 0.8

    # Coordinate canonicalization definitions
    canonicalizations:
      headframe:
        source_space: RAS
        scale_to_mm: 1.0

    # Named transform recipes
    transforms:
      headframe_to_lps:
        sequence:
          - kind: sitk_file
            path: /transforms/headframe.h5

    # Geometry templates (reduce repetition)
    asset_templates:
      "structure:*":
        role: anatomy
        caps: [RENDERABLE]

    target_templates:
      default_target:
        reducer: centroid

    # Resources (multi-geometry containers)
    resources: []

    # Asset catalog (meshes, points, etc.)
    assets: []

    # Target catalog (probe insertion points)
    targets: []

    # Scene graph (instances with transforms)
    scene:
      nodes: []

    # Planning domain (probes, arcs, calibrations)
    plan:
      arcs: {}
      probes: {}
      reticles: {}
      calibrations:
        files: {}
        probe_to_ref: {}

    # Rendering options
    options:
      color_map: rainbow


Assets
------

Assets are geometry objects (meshes, point clouds, lines) loaded from files or
resources.

Basic Asset Definition
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    assets:
      - key: brain_mesh
        kind: mesh
        role: anatomy
        src: /data/brain.obj
        loader: trimesh
        caps: [RENDERABLE, COLLIDABLE]

Asset Fields
~~~~~~~~~~~~

==================== ============ ================================================
Field                Required     Description
==================== ============ ================================================
``key``              Yes          Unique identifier
``kind``             Inferred     ``mesh``, ``points``, or ``lines``
``role``             Inferred     ``anatomy``, ``target``, ``landmark``, ``geometry``
``src``              Conditional  Path to geometry file
``loader``           Inferred     Loader name (``trimesh``, ``numpy_points``, etc.)
``loader_kwargs``    No           Additional loader arguments
``caps``             No           Capabilities (default: ``[RENDERABLE]``)
``material_ref``     No           Reference to materials bank
``material``         No           Inline material definition
``tags``             No           Arbitrary string tags
``metadata``         No           Arbitrary key-value metadata
``templates``        No           Template names to apply
``transform``        No           Scene placement transform
``scene_tags``       No           Tags for auto-generated scene node
``auto_scene``       No           Auto-create scene node (default: true)
==================== ============ ================================================

Source Modes
~~~~~~~~~~~~

Assets support three mutually exclusive source modes:

**File source** (most common):

.. code-block:: yaml

    - key: brain
      src: /data/brain.obj
      loader: trimesh

**Resource reference** (for multi-geometry files):

.. code-block:: yaml

    - key: left_hemisphere
      from_resource: brain_atlas
      selector:
        kind: label
        label: 1

**Neither** (geometry injected programmatically):

.. code-block:: yaml

    - key: dynamic_points
      kind: points

Bulk Asset Declarations
~~~~~~~~~~~~~~~~~~~~~~~

Declare multiple similar assets with shared configuration:

.. code-block:: yaml

    assets:
      - keys: [structure:PL, structure:MD, structure:CLA]
        src: /data/{name}-Mask.nrrd
        templates: [structure]

Placeholders:

- ``{name}``: Suffix after the last ``:`` (e.g., ``structure:PL`` → ``PL``)
- ``{key}``: Full key (e.g., ``structure:PL``)


Targets
-------

Targets represent probe insertion points. They are always ``kind: points``.

Target Source Modes
~~~~~~~~~~~~~~~~~~~

**Explicit file**:

.. code-block:: yaml

    targets:
      - key: target1
        src: /targets/target1.npy
        loader: numpy_points

**Derived from asset** (most common):

.. code-block:: yaml

    targets:
      - key: target:PL
        source_key: structure:PL
        reducer: centroid

**Resource reference**:

.. code-block:: yaml

    targets:
      - key: target:holes
        from_resource: hole_positions
        selector:
          kind: name
          name: insertion_points

Target-Specific Fields
~~~~~~~~~~~~~~~~~~~~~~

==================== ============ ================================================
Field                Required     Description
==================== ============ ================================================
``source_key``       Conditional  Asset key to derive target from
``reducer``          No           Reduction method (``centroid``, ``mesh_centroid``, ``mesh_center_mass``, ``hemisphere_center_mass``, …)
``reducer_kwargs``   No           Extra args for the reducer (e.g. ``{hemisphere: left}``)
``approach_vector``  No           [x, y, z] preferred insertion direction (advisory)
``uncertainty_mm``   No           Position uncertainty radius (advisory)
``chem_shift_policy`` No          MR chem-shift correction: ``auto`` (default, applies if `imaging` is configured), ``on``, ``off``
==================== ============ ================================================

Range Target Declarations
~~~~~~~~~~~~~~~~~~~~~~~~~

Generate numbered targets:

.. code-block:: yaml

    targets:
      - key_pattern: "target:hole:{n}"
        range: [1, 12]
        src: /holes/Hole{n}.npy

Derived Target Declarations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create targets from multiple assets:

.. code-block:: yaml

    targets:
      - derive_from: [structure:PL, structure:MD, structure:CLA]
        key_prefix: "target:"
        reducer: centroid


Auto-Inference
--------------

The configuration system automatically infers values to reduce boilerplate.

Kind and Loader from File Extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``src`` is provided, ``kind`` and ``loader`` are inferred from the extension:

=============== ========== ================
Extension       Kind       Loader
=============== ========== ================
``.obj``        mesh       trimesh
``.stl``        mesh       trimesh
``.ply``        mesh       trimesh
``.nrrd``       mesh       sitk_volume
``.nii``        mesh       sitk_volume
``.nii.gz``     mesh       sitk_volume
``.npy``        points     numpy_points
=============== ========== ================

**Example**: No need to specify kind/loader:

.. code-block:: yaml

    assets:
      - key: brain
        src: /data/brain.obj
        # kind: mesh and loader: trimesh are inferred

Explicit values always override inference:

.. code-block:: yaml

    assets:
      - key: special
        src: /data/special.npy
        kind: lines        # Override inferred 'points'
        loader: custom     # Override inferred 'numpy_points'

Role from Key Prefix
~~~~~~~~~~~~~~~~~~~~

When ``role`` is not specified, it's inferred from the key prefix:

=============== ===========
Prefix          Role
=============== ===========
``structure:*`` anatomy
``brain*``      anatomy
``target:*``    target
``landmark:*``  landmark
*(default)*     geometry
=============== ===========

**Example**:

.. code-block:: yaml

    assets:
      - key: structure:PL
        src: /data/PL.obj
        # role: anatomy is inferred from "structure:" prefix

      - key: brain_mesh
        src: /data/brain.obj
        # role: anatomy is inferred from "brain" prefix

      - key: landmark:bregma
        src: /data/bregma.npy
        # role: landmark is inferred from "landmark:" prefix


Templates
---------

Templates define reusable defaults that can be applied to multiple assets or
targets.

Defining Templates
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    asset_templates:
      structure:
        kind: mesh
        role: anatomy
        caps: [RENDERABLE]
        material_ref: anatomy_material

      transparent:
        material:
          opacity: 0.5

    target_templates:
      default:
        reducer: centroid
        caps: [RENDERABLE]

Applying Templates
~~~~~~~~~~~~~~~~~~

**Explicit template reference**:

.. code-block:: yaml

    assets:
      - key: brain
        src: /data/brain.obj
        templates: [structure, transparent]

Templates are merged left-to-right, with the asset's own values taking priority.

**Glob pattern auto-matching**:

Templates with glob patterns automatically match assets by key:

.. code-block:: yaml

    asset_templates:
      "structure:*":
        role: anatomy
        caps: [RENDERABLE]

    assets:
      - key: structure:PL
        src: /data/PL.obj
        # Automatically matches "structure:*" template

Matching rules:

1. Explicit ``templates: [...]`` always takes precedence
2. Exact template names have priority over glob patterns
3. Multiple glob matches apply in template order

Template Merge Behavior
~~~~~~~~~~~~~~~~~~~~~~~

When templates are applied:

- Scalar values: Later values override earlier
- Lists (``tags``, ``caps``): Union of all values
- Dicts (``metadata``, ``loader_kwargs``): Shallow merge
- ``material``, ``collision``: Nested merge


Materials
---------

Define reusable materials in the ``materials`` bank:

.. code-block:: yaml

    materials:
      default:
        name: default
        color: "#C8C8C8"
        opacity: 1.0
        wireframe: false
        visible: true

      transparent_red:
        color: "#FF0000"
        opacity: 0.3

      wireframe_blue:
        color: "#0000FF"
        wireframe: true

Reference materials in assets/targets:

.. code-block:: yaml

    assets:
      - key: brain
        src: /data/brain.obj
        material_ref: transparent_red

Or define inline:

.. code-block:: yaml

    assets:
      - key: brain
        src: /data/brain.obj
        material:
          color: "#FFB6C1"
          opacity: 0.8


Transforms
----------

Named Transform Recipes
~~~~~~~~~~~~~~~~~~~~~~~

Define reusable transform sequences:

.. code-block:: yaml

    transforms:
      headframe_to_lps:
        sequence:
          - kind: sitk_file
            path: /transforms/headframe.h5
          - kind: translate_mm
            delta: [10.0, 0.0, 0.0]

      rotate_90:
        kind: rotate_euler_deg
        order: ZYX
        angles_deg: [0, 0, 90]

Transform Operations
~~~~~~~~~~~~~~~~~~~~

**Translation**:

.. code-block:: yaml

    - kind: translate_mm
      delta: [x, y, z]
      invert: false

**Rotation (Euler angles)**:

.. code-block:: yaml

    - kind: rotate_euler_deg
      order: ZYX
      angles_deg: [rx, ry, rz]
      invert: false

**SITK transform file**:

.. code-block:: yaml

    - kind: sitk_file
      path: /path/to/transform.h5
      invert: false

Transform References
~~~~~~~~~~~~~~~~~~~~

Reference transforms by key or inline:

.. code-block:: yaml

    # By key
    assets:
      - key: brain
        transform: headframe_to_lps

    # Inline
    assets:
      - key: skull
        transform:
          inline:
            sequence:
              - kind: translate_mm
                delta: [0, 0, 10]


Canonicalization
----------------

Canonicalization converts geometry from file coordinates to a standard space.

.. code-block:: yaml

    canonicalizations:
      ras_1mm:
        source_space: RAS
        scale_to_mm: 1.0
        version: canon-v1

      file_native:
        source_space: FILE_NATIVE
        transform:
          key: headframe_to_lps

Reference in assets:

.. code-block:: yaml

    assets:
      - key: brain
        src: /data/brain.obj
        canonicalization_ref: ras_1mm
        canonicalization_override:
          scale_to_mm: 0.001  # Override just the scale

Source Space Options
~~~~~~~~~~~~~~~~~~~~

- ``RAS``: Right-Anterior-Superior (neuroimaging standard)
- ``LPS``: Left-Posterior-Superior (DICOM/internal canonical)
- Other orientation codes (see ``orientation_codes.py``)
- ``FILE_NATIVE``: No assumed orientation; requires explicit transform


.. _config_tags_vs_scene_tags:

``tags`` vs ``scene_tags``
--------------------------

Two fields with almost-identical names but different scopes — easy to confuse,
worth getting straight before authoring a config.

================ ====================================================
``tags``         Lives on the **asset spec**. Catalog-only metadata —
                 used for queries like "give me every CCF region" or
                 "is this asset a target?". Doesn't reach the rendered
                 scene. Default: empty list.
``scene_tags``   Lives on the **scene node** (auto-created from the asset).
                 What the UI, collision adapter, and visibility toggles
                 filter on. Default: empty list (which suppresses
                 auto-scene-node creation unless ``transform`` is set).
================ ====================================================

A node is auto-created from an asset when **either** ``transform`` is set
**or** ``scene_tags`` is non-empty (controlled by ``auto_scene``, default
``true``). Probes are an exception — they always have ``scene_tags=
["probe", "dynamic"]`` by default.

Well-known ``scene_tags`` values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are tags the existing runtime + UI actively look for. Adding new
ones is fine; they just won't trigger any behaviour unless someone wires
them up.

==================== =====================================================
Tag                  What it does
==================== =====================================================
``static``           Doesn't move with probe state. Affects collision-group
                     inclusion.
``dynamic``          Repositioned on every probe state change (probes only).
``probe``            Identifies probe meshes. Drives the *Probes* visibility
                     switch + opacity slider on the Display tab.
``brain``            Drives the *Brain outline* visibility group; also what
                     the *Recenter on brain* button uses to find the focal
                     point.
``structure``        CCF-region meshes; drives the *CCF regions* group.
``fixture``          Generic rig hardware (well, probe-guard, etc.). Drives
                     the *Other fixtures* group; gets a default 60%
                     transparency at startup.
``implant``          The implant body. Drives the *Implant* group; gets a
                     default 80% transparency at startup. The implant
                     typically carries **both** ``fixture`` and ``implant``;
                     the visibility-group exclusion column keeps the implant
                     slider distinct from "Other fixtures".
``headframe``        Headframe mesh. Subject to fixture-group defaults.
``target``           Visualised target points.
``hole``             Per-bore points on the implant (used by hole extraction).
==================== =====================================================

When in doubt: ``scene_tags`` is what controls how the user *sees* the
asset; ``tags`` is what controls how the *code* finds it.


Scene Graph
-----------

The scene graph defines instances of assets with transforms for rendering.

Auto-Generated Nodes
~~~~~~~~~~~~~~~~~~~~

Assets and targets with ``transform`` or ``scene_tags`` automatically create
scene nodes:

.. code-block:: yaml

    assets:
      - key: brain
        src: /data/brain.obj
        transform: headframe_to_lps
        scene_tags: [static, anatomy]
        # Auto-creates: scene.nodes[key=brain, asset=brain]

Suppress with ``auto_scene: false``.

Explicit Scene Nodes
~~~~~~~~~~~~~~~~~~~~

Override or define additional nodes:

.. code-block:: yaml

    scene:
      nodes:
        - key: brain_instance
          asset: brain
          transform: headframe_to_lps
          tags: [static]

        - key: probe_node
          asset: probe:neuropixels
          pose_source_probe: probe1
          tags: [dynamic]

Node Fields
~~~~~~~~~~~

======================== ============ ================================================
Field                    Required     Description
======================== ============ ================================================
``key``                  Yes          Unique node identifier
``asset``                Yes          Asset or target key from catalog
``transform``            No           Transform reference
``tags``                 No           Arbitrary tags for filtering
``pose_source_probe``    No           Link to planning probe for dynamic pose
======================== ============ ================================================


Planning
--------

The planning section defines probe placement parameters.

Arcs
~~~~

Define arc angles:

.. code-block:: yaml

    plan:
      arcs:
        arc_left: 15.0    # AP angle in degrees
        arc_right: -15.0

Probes
~~~~~~

.. code-block:: yaml

    plan:
      probes:
        probe_A:
          kind: neuropixels
          arc: arc_left
          target: target:PL
          slider_ml: 5.0
          spin: 0.0
          past_target_mm: 2.0
          offsets_RA: [0.0, 0.0]
          calibrated: false
          auto_scene: true
          scene_tags: [probe, dynamic]

Probe Fields
~~~~~~~~~~~~

========================= ============ ================================================
Field                     Required     Description
========================= ============ ================================================
``kind``                  Yes          Probe type (matches ``probe:{kind}`` asset)
``arc``                   No           Arc key reference; ``null`` for off-arc probes
``target``                Yes          Target reference (key string, ``{kind: catalog, key: ...}``, ``{kind: node, key: ...}``, ``{kind: inline, point_RAS: [x, y, z]}``, or a bare ``[x, y, z]`` list which is coerced to ``inline``)
``ap_local``              No           Per-probe AP angle override (deg). When ``bind_ap_to_arc`` is true this is ignored at runtime; kept as a fallback when you unbind.
``bind_ap_to_arc``        No           If ``true`` (default), AP comes from ``arcs[arc]``; if ``false``, AP comes from ``ap_local``. Requires ``arc`` to be set when true.
``slider_ml``             No           ML slider angle (deg, default 0)
``spin``                  No           Spin angle around the probe shaft (deg, default 0)
``past_target_mm``        No           Distance past target along the shaft (mm, default 0). Positive = deeper into the brain. For multi-shank probes this is measured from the named shank's tip — see ``position_bearing_shank``.
``offsets_RA``            No           [R, A] entry-offset in mm relative to the bore (default [0, 0])
``position_bearing_shank`` No          1-indexed shank index whose tip is the "named" reference: it's the one tip-RAS / brain-depth readouts report and the one ``past_target_mm`` measures from. Default 1. Single-shank probes ignore the value.
``calibrated``            No           Lock the AP/ML angles to a pre-recorded calibration (default ``false``). Calibration data lives under ``plan.calibrations``.
``auto_scene``            No           Auto-create the ``probe:{name}`` scene node (default ``true``)
``scene_tags``            No           Tags for the auto-created scene node. **Default ``["probe", "dynamic"]``** — don't drop these unless you know what you're doing; ``probe`` drives the Display-tab Probes group + opacity slider, and ``dynamic`` is what tells the renderer + collision adapter to refresh on probe state changes.
========================= ============ ================================================

Target References
~~~~~~~~~~~~~~~~~

Probes can reference targets two ways:

**Catalog target** (most common):

.. code-block:: yaml

    probes:
      probe_A:
        target: target:PL  # Short form
        # Or explicit:
        target:
          kind: catalog
          key: target:PL

**Scene node target**:

.. code-block:: yaml

    probes:
      probe_A:
        target:
          kind: node
          key: target_node_key

Calibrations
~~~~~~~~~~~~

.. code-block:: yaml

    plan:
      reticles:
        reticle_A:
          offset_RAS: [0.0, 0.0, 0.0]
          rotation_z: 0.0

      calibrations:
        files:
          cal_2024:
            directory: /calibrations/2024
            reticle: reticle_A
          cal_xlsx:
            file: /calibrations/probes.xlsx

        probe_to_ref:
          probe_A: "cal_2024:12345"
          probe_B:
            cal_id: cal_2024
            probe_code: "67890"


Plan-Only YAML
--------------

The trame UI's *Save plan* button writes a separate, smaller YAML
containing only the ``plan:`` block of the full config — i.e. a
serialized ``PlanningModel``. The *Load plan* button (a browser file
picker) reads the same shape.

This format intentionally **omits the asset list, targets, transforms,
materials, and scene**: it's portable across any full config that shares
the same probe roster. Use it to checkpoint a working plan and reload
later, or to ship a plan to a colleague running a different mouse-specific
config with the same probes.

Top-level shape:

.. code-block:: yaml

    arcs:
      a: 13.0
      b: -10.0
      c: -43.0
    probes:
      P1:
        kind: quadbase
        arc: a
        slider_ml: -12.0
        spin: 141.0
        ap_local: null
        bind_ap_to_arc: true
        past_target_mm: 0.0675
        offsets_RA: [0.0, 0.0]
        position_bearing_shank: 1
        target:
          kind: catalog
          key: target:MD
      # ... more probes

Probes named in the loaded plan that aren't in the running session's
state are skipped (with a stdout warning) — adding/removing probes is a
full-config concern, not a plan concern.

The geometric **Export plan** button is a *different* format
(``plan_export_version: 1``, fields like ``tip_RAS_mm`` and
``depth_from_brain_surface_mm``) intended for hand-off to physical
execution. It's read-only — there's no loader for that variant.


Capabilities
------------

Capabilities control what systems process an asset:

============== ===============================================
Capability     Description
============== ===============================================
RENDERABLE     Asset appears in rendered scene
MOVABLE        Asset can be moved interactively
COLLIDABLE     Asset participates in collision detection
SELECTABLE     Asset can be selected in UI
DEFORMABLE     Asset supports deformation
SAVABLE        Asset state is persisted
============== ===============================================

Specify as list:

.. code-block:: yaml

    caps: [RENDERABLE, COLLIDABLE]


Complete Example
----------------

.. code-block:: yaml

    version: 1

    paths:
      data_root: /data/mouse_123

    materials:
      brain:
        color: "#FFB6C1"
        opacity: 0.7
      target:
        color: "#FF0000"

    transforms:
      headframe_to_lps:
        kind: sitk_file
        path: ${paths.data_root}/transforms/headframe.h5

    asset_templates:
      "structure:*":
        role: anatomy
        material_ref: brain
        transform: headframe_to_lps

    target_templates:
      "target:*":
        reducer: centroid
        material_ref: target

    assets:
      - keys: [structure:PL, structure:MD, structure:CLA]
        src: ${paths.data_root}/masks/{name}-Mask.nrrd

      - key: probe:neuropixels
        src: /probes/neuropixels.obj
        role: geometry

    targets:
      - derive_from: [structure:PL, structure:MD, structure:CLA]
        key_prefix: "target:"

    plan:
      arcs:
        left: 15.0
        right: -15.0

      probes:
        probe_A:
          kind: neuropixels
          arc: left
          target: target:PL
          slider_ml: 5.0

        probe_B:
          kind: neuropixels
          arc: right
          target: target:MD
          slider_ml: 3.0


Validation and Errors
---------------------

The configuration is validated at parse time. Common errors:

**Cross-reference errors**:

- Asset/target referenced in scene but not defined
- Template referenced but not defined
- Transform key not found
- Calibration references invalid reticle

**Source mode errors**:

- Asset has ``src`` without ``loader`` (and extension not recognized)
- Asset has both file source and resource source
- Target missing source (no ``src``, ``source_key``, or ``from_resource``)

**Constraint violations**:

- Targets cannot have ``COLLIDABLE`` capability
- ``FILE_NATIVE`` source space requires a transform
- Duplicate keys in asset/target catalog


API Usage
---------

Load and validate configuration:

.. code-block:: python

    from aind_low_point.config import ConfigModel, expand_config
    import yaml

    # Load from YAML
    with open("config.yaml") as f:
        data = yaml.safe_load(f)

    # Parse and validate
    config = ConfigModel.model_validate(data)

    # Access expanded data
    print(f"Assets: {len(config.assets)}")
    print(f"Targets: {len(config.targets)}")
    print(f"Scene nodes: {len(config.scene.nodes)}")

    # Export fully-expanded config (no templates, bulk specs resolved)
    explicit = config.to_explicit_dict()

    # Or use convenience function
    explicit = expand_config(data)
