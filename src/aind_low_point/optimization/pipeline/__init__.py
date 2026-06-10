"""Offline placement-optimizer pipeline (Phase 1 → Phase 2 → emit).

The overnight driver `run_subject_overnight.sh` runs three stages:

1. :mod:`phase1_pool` (``alp-phase1``) — MRV enumerate + spin restore +
   RProp/coarse-fine Phase-1 objective (no FCL cull), producing a pose pool.
2. :mod:`phase2_ipopt` (``alp-phase2``) — IPOPT + thick-well polish of the
   top-K by min clearance, FCL-validated at the end, producing a handoff.
3. :mod:`emit` (``alp-emit``) — emit trame plan configs + tree + manifest.

Shared building blocks: :mod:`enumeration`, :mod:`phase1_build`,
:mod:`restore`, :mod:`thick_well`, :mod:`probe_setup`, :mod:`phase1_geometry`.
"""
