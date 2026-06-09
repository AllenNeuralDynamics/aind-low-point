#!/usr/bin/env bash
# Full placement pipeline for ONE subject, config-driven and unattended-safe:
#   Phase 1 (MRV enumerate 3-arc/4-probe → 16-spin restore+thick well →
#            RProp coarse/fine Phase-1, NO intermediate FCL)
#   → rank top-N by soft min_clear
#   → Phase 2 (IPOPT limited-memory + thick well, GPU thread-shared)
#   → FCL at the end (ground-truth gate)
#   → emit per-plan trame configs from the feasible set.
#
# Memory-safe: Phase-1 is chunked; Phase-2 is one process / one GPU context with
# threads sharing the SDF grids in HBM (no process-pool RAM blow-up). All outputs
# are subject-keyed so multiple subjects never collide.
#
# Usage:
#   CONFIG=examples/837229-config.yml scripts/run_subject_overnight.sh
# Optional env (with defaults):
#   HOLES=scratch/0283-300-04.holes.yml  TOPK=200  P2_ITER=1000
#   COARSE_N=1000  REDUCED_FINE=50  FULL_FINE=50  WORKERS=4  EMIT_N=15
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:?set CONFIG=examples/<subject>-config.yml}"
HOLES="${HOLES:-scratch/0283-300-04.holes.yml}"
STEM="$(basename "${CONFIG%.yml}")"
POOL="scratch/${STEM}_pool.pkl"
HANDOFF="scratch/${STEM}_phase2_handoff.pkl"
PLANDIR="scratch/${STEM}_plans"
# WORKERS=4: GPU thread-shared knee (re-benched on real 837229 candidates after
# the kernel vmapping). Faster GPU evals make Phase-2 more host/IPOPT-bound, so
# the knee moved in from >4 to 4: W=2 is -18%, W=4 == W=8 (flat). 4 = full
# throughput with less GIL/stream contention + HBM than 8.
TOPK="${TOPK:-200}"; P2_ITER="${P2_ITER:-1000}"; WORKERS="${WORKERS:-4}"
COARSE_N="${COARSE_N:-1000}"; REDUCED_FINE="${REDUCED_FINE:-50}"; FULL_FINE="${FULL_FINE:-50}"
EMIT_N="${EMIT_N:-15}"

echo "[$(date +%H:%M)] === subject=${STEM} ==="

echo "[$(date +%H:%M)] Phase 1: MRV enumerate + restore + RProp/coarse-fine (no FCL cull) → ${POOL}"
CONFIG="$CONFIG" HOLES="$HOLES" \
  MAX_ARCS=3 MAX_PROBES_PER_ARC=4 FCL_TOPK=0 \
  MINIMIZER=rprop WELL=thick N_SPINS=16 \
  COARSE_N="$COARSE_N" REDUCED_FINE="$REDUCED_FINE" FULL_FINE="$FULL_FINE" \
  OUT="$POOL" JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  uv run --python 3.13 -m scripts.mrv_pool_run

echo "[$(date +%H:%M)] Phase 2: IPOPT + thick well on top-${TOPK} by min_clear (FCL at end) → ${HANDOFF}"
SOLVER=ipopt CONFIG="$CONFIG" HOLES="$HOLES" \
  POSES="$POOL" OUT="$HANDOFF" SELECT_BY=min_clear \
  WELL=thick POOL=thread PLATFORM=gpu GPU_MEM_FRACTION=0.9 WORKERS="$WORKERS" \
  TOPK="$TOPK" P2_ITER="$P2_ITER" \
  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.phase2_parallel

echo "[$(date +%H:%M)] Emit top-${EMIT_N} trame configs → ${PLANDIR}/"
CONFIG="$CONFIG" HOLES="$HOLES" HANDOFF="$HANDOFF" N="$EMIT_N" OUTDIR="$PLANDIR" \
  JAX_PLATFORMS=cpu uv run --python 3.13 -m scripts.emit_plan_configs

echo "[$(date +%H:%M)] === DONE: pool=${POOL} handoff=${HANDOFF} plans=${PLANDIR}/ ==="
