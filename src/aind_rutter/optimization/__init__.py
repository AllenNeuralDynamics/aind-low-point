"""Placement-optimizer subpackage.

The current production optimizer is the offline ``optimization.pipeline`` flow:
``rutter-phase1`` builds the MRV/RProp pool, ``rutter-phase2`` polishes and ranks the
handoff, and ``rutter-emit`` writes plan-only YAML files. Import the
subpackages directly; re-exporting here would load jax to launch the viewer.
"""
