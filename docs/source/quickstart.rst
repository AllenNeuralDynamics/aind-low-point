Quickstart Guide
================

This guide helps you create your first aind-low-point configuration.

Installation
------------

.. code-block:: bash

    pip install aind-low-point

Or with uv:

.. code-block:: bash

    uv add aind-low-point


Your First Configuration
------------------------

Create a file named ``config.yaml``:

.. code-block:: yaml

    version: 1

    # Define brain anatomy assets
    assets:
      - key: brain
        src: /path/to/brain.obj

      - key: structure:PL
        src: /path/to/PL-Mask.nrrd

    # Define insertion targets derived from structures
    targets:
      - key: target:PL
        source_key: structure:PL
        reducer: centroid

    # Define probe placement
    plan:
      arcs:
        left: 15.0

      probes:
        probe_A:
          kind: neuropixels
          arc: left
          target: target:PL

That's it! The configuration system automatically infers:

- ``kind: mesh`` and ``loader: trimesh`` for ``.obj`` files
- ``kind: mesh`` and ``loader: sitk_volume`` for ``.nrrd`` files
- ``role: anatomy`` for keys starting with ``brain`` or ``structure:``
- ``role: target`` for keys starting with ``target:``


Loading Configuration
---------------------

.. code-block:: python

    import yaml
    from aind_low_point.config import ConfigModel

    with open("config.yaml") as f:
        data = yaml.safe_load(f)

    config = ConfigModel.model_validate(data)

    # Access your data
    print(f"Loaded {len(config.assets)} assets")
    print(f"Loaded {len(config.targets)} targets")
    print(f"Configured {len(config.plan.probes)} probes")


Reducing Repetition with Templates
----------------------------------

For multiple similar structures, use templates and bulk declarations:

.. code-block:: yaml

    version: 1

    # Define a template that matches all structure:* keys
    asset_templates:
      "structure:*":
        caps: [RENDERABLE]
        transform: headframe_to_lps

    target_templates:
      "target:*":
        reducer: centroid

    # Define the transform
    transforms:
      headframe_to_lps:
        kind: sitk_file
        path: /transforms/headframe.h5

    # Bulk declare multiple structures at once
    assets:
      - keys: [structure:PL, structure:MD, structure:CLA]
        src: /data/{name}-Mask.nrrd

    # Bulk derive targets from structures
    targets:
      - derive_from: [structure:PL, structure:MD, structure:CLA]
        key_prefix: "target:"

This creates:

- 3 assets: ``structure:PL``, ``structure:MD``, ``structure:CLA``
- 3 targets: ``target:PL``, ``target:MD``, ``target:CLA``

All assets automatically:

- Match the ``structure:*`` template
- Get ``role: anatomy`` inferred
- Get ``kind: mesh``, ``loader: sitk_volume`` from ``.nrrd``


Adding Visual Customization
---------------------------

.. code-block:: yaml

    version: 1

    # Define reusable materials
    materials:
      anatomy:
        color: "#FFB6C1"
        opacity: 0.7

      target:
        color: "#FF0000"

    # Templates reference materials
    asset_templates:
      "structure:*":
        material_ref: anatomy

    target_templates:
      "target:*":
        material_ref: target
        reducer: centroid

    assets:
      - key: structure:PL
        src: /data/PL.nrrd

    targets:
      - key: target:PL
        source_key: structure:PL


Multi-Probe Setup
-----------------

.. code-block:: yaml

    version: 1

    assets:
      - key: structure:PL
        src: /data/PL.nrrd

      - key: structure:MD
        src: /data/MD.nrrd

      # Probe geometry (the visual representation)
      - key: probe:neuropixels
        src: /probes/neuropixels.obj
        role: geometry

    targets:
      - key: target:PL
        source_key: structure:PL
        reducer: centroid

      - key: target:MD
        source_key: structure:MD
        reducer: centroid

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

Each probe automatically creates a scene node (``probe:probe_A``, ``probe:probe_B``)
using the ``probe:neuropixels`` asset geometry.


Scene Tags Cheat Sheet
----------------------

Assets and probes both have a ``scene_tags`` field that drives how the UI
treats them (visibility toggles, default opacity, recenter target). It's
distinct from the catalog-only ``tags`` field — see
:ref:`config_tags_vs_scene_tags` for the full distinction.

Most-used values when hand-authoring:

============= =========================================================
Tag           Use it on…
============= =========================================================
``brain``     The brain-surface mesh. Drives *Recenter on brain* and
              the *Brain outline* visibility toggle.
``structure`` CCF region meshes. Drives the *CCF regions* group.
``implant``   The implant body. Default opacity 0.2 (80% transparent)
              so probes threading the holes stay visible.
``fixture``   Headframe / well / probe-guard etc. Default opacity 0.6.
``probe``     Probe meshes. Auto-added by ``ProbeDeclModel``; leave it.
``dynamic``   Anything that moves with probe state. Auto-added for
              probes; leave it.
``static``    Anything that doesn't move with probe state.
============= =========================================================

You generally don't need to tag every asset — only the ones whose
behaviour the UI / runtime needs to differentiate. A bare structure mesh
with no scene_tags is still rendered (assets default to ``auto_scene:
true`` if any of ``transform`` / ``scene_tags`` are set); it just won't
appear in the Display tab's visibility toggles.


Common File Extensions
----------------------

The loader is automatically inferred from file extensions:

============== ========== ================
Extension      Kind       Loader
============== ========== ================
``.obj``       mesh       trimesh
``.stl``       mesh       trimesh
``.ply``       mesh       trimesh
``.nrrd``      mesh       sitk_volume
``.nii``       mesh       sitk_volume
``.nii.gz``    mesh       sitk_volume
``.npy``       points     numpy_points
============== ========== ================


Next Steps
----------

- :doc:`configuration` - Complete configuration reference
- API documentation - Python API for building runtime objects
