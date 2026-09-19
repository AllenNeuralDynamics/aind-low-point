"""The planning domain: geometry, catalog, scene, rig and plan.

Nothing here imports the config layer, the optimizer or a frontend, so a viewer
can be built without pulling in jax and the optimizer can run without trame.
"""
