"""Offline placement-optimizer pipeline (Phase 1 → Phase 2 → emit).

The overnight driver `run_subject_overnight.sh` runs three stages:

1. :mod:`phase1` (``rutter-phase1``) — MRV enumerate + spin restore +
   RProp/coarse-fine soft objective (no FCL cull), producing a pose pool.
2. :mod:`phase2` (``rutter-phase2``) — IPOPT + thick-well polish of the
   top-K by min clearance, FCL-validated at the end, producing a handoff.
3. :mod:`emit` (``rutter-emit``) — emit trame plan configs + tree + manifest.

Shared building blocks: :mod:`candidates` (the enumerator and its atlas cache),
:mod:`subject` (the runtime a stage optimizes over), :mod:`stage_setup`,
:mod:`fixtures`, :mod:`thick_well`, :mod:`probe_setup`, :mod:`settings`,
:mod:`records` and :mod:`payloads`.
"""
