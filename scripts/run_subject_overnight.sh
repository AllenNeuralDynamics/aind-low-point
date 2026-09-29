#!/usr/bin/env bash
# Full placement pipeline for ONE subject, config-driven and unattended-safe:
#   Phase 1 (MRV enumerate 3-arc/4-probe → 16-spin restore+thick well →
#            one RProp pass, NO intermediate FCL)
#   → rank: the objective's first RUTTER_RANK_HEAD, then the learned model
#   → Phase 2 (IPOPT limited-memory + thick well, process pool under MPS)
#   → FCL at the end (ground-truth gate)
#   → emit per-plan trame configs from the feasible set.
#
# Phase-1 is chunked and runs one process per arc-count group. Phase 2's three
# worker processes share the card through an MPS daemon this script starts and
# stops. All outputs are subject-keyed so multiple subjects never collide.
#
# Coverage is the package default (normalized, alpha 0.2, gain 7). A subject with
# a retro cloud can add RUTTER_RETRO_DENSITY=1; every RUTTER_ variable set here
# reaches both phases.
#
# Every variable the pipeline reads is RUTTER_-prefixed; the bare spellings
# (CONFIG, OUT, LIMIT, N, WORKERS, POOL, PLATFORM) are set for unrelated reasons
# in ordinary shells and are no longer read.
#
# Usage:
#   RUTTER_CONFIG=examples/837229-config.yml scripts/run_subject_overnight.sh
# Optional env (with defaults):
#   RUTTER_HOLES=scratch/0283-300-04.holes.yml  RUTTER_TOPK=2000 (about an hour)
#   RUTTER_WORKERS=3  RUTTER_GPU_MEM_FRACTION=0.28  RUTTER_PLANS=<all feasible>
#   FRESH=1   force-clear the geometry caches + pool before running
set -euo pipefail
cd "$(dirname "$0")/.."

# Gotcha 2: XLA preallocation must be DISABLED in the *launch* env (setting it
# from inside Python is too late — JAX then defaults to preallocate=true and
# grabs the whole mem-fraction up front, which starves extra Phase-2 workers /
# any process-pool or MPS run). Exporting it here puts it in the env every
# `uv run` child inherits; without it only one Phase-2 worker fits.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

CONFIG="${RUTTER_CONFIG:?set RUTTER_CONFIG=examples/<subject>-config.yml}"
HOLES="${RUTTER_HOLES:-scratch/0283-300-04.holes.yml}"
STEM="$(basename "${CONFIG%.yml}")"
# TAG: optional output suffix so variant runs don't clobber each other
# (e.g. TAG=density → scratch/<stem>_density_pool.json.gz). Seed cache is untagged
# (geometry-only) so it's reused across coverage variants.
TAG="${TAG:+_${TAG}}"
POOL="scratch/${STEM}${TAG}_pool.json.gz"
HANDOFF="scratch/${STEM}${TAG}_phase2_handoff.json"
PLANDIR="scratch/${STEM}${TAG}_plans"
# Three processes under MPS run about 2000 candidates an hour on a 10 GB card;
# threads in one process are GIL-bound.
TOPK="${RUTTER_TOPK:-2000}"
WORKERS="${RUTTER_WORKERS:-3}"; GPU_FRACTION="${RUTTER_GPU_MEM_FRACTION:-0.28}"
MAX_ARCS="${RUTTER_MAX_ARCS:-3}"  # Phase 1 loops n_arcs down to 1, one process each
MAX_PROBES_PER_ARC="${RUTTER_MAX_PROBES_PER_ARC:-4}"
N_SPINS="${RUTTER_N_SPINS:-16}"
PLANS="${RUTTER_PLANS:-}"

# Gotcha 1: the atlas + seed caches are keyed only by the config stem, but their
# CONTENTS depend on probe geometry (meshes, kinds), the implant holes, and the
# enumeration caps. Editing any of those without clearing the caches silently
# reuses stale geometry. Stamp a fingerprint and nuke atlas + seeds + pool (the
# pool RESUMES, so it accumulates stale records too) whenever it changes.
# Coverage settings (RUTTER_RETRO_DENSITY / RUTTER_COV_*) are NOT in the
# fingerprint, so coverage variants still share the geometry caches.
# NOT detected: editing a probe .obj in place (config text unchanged) → FRESH=1.
export RUTTER_ATLAS_CACHE="${RUTTER_ATLAS_CACHE:-scratch/atlas_${STEM}.json.gz}"
export RUTTER_SEED_CACHE="${RUTTER_SEED_CACHE:-scratch/mrv_seeds_${STEM}.json.gz}"
STAMP="scratch/geom_${STEM}.stamp"
FRESH="${FRESH:-0}"
geom_fingerprint() {
  {
    cat "$CONFIG"
    if [ -f "$HOLES" ]; then cat "$HOLES"; fi
    printf 'caps MAX_ARCS=%s MPA=%s N_SPINS=%s\n' \
      "$MAX_ARCS" "$MAX_PROBES_PER_ARC" "$N_SPINS"
  } | sha256sum | cut -d' ' -f1
}
FP="$(geom_fingerprint)"
if [ "$FRESH" = "1" ]; then
  echo "[$(date +%H:%M)] FRESH=1 → clearing geometry caches + pool"
  rm -f "$RUTTER_ATLAS_CACHE" "$RUTTER_SEED_CACHE" "$POOL"; echo "$FP" >"$STAMP"
elif [ -f "$STAMP" ] && [ "$(cat "$STAMP")" != "$FP" ]; then
  echo "[$(date +%H:%M)] config/holes/caps changed → clearing stale geometry caches + pool"
  echo "    atlas=$RUTTER_ATLAS_CACHE seeds=$RUTTER_SEED_CACHE pool=$POOL"
  rm -f "$RUTTER_ATLAS_CACHE" "$RUTTER_SEED_CACHE" "$POOL"; echo "$FP" >"$STAMP"
elif [ ! -f "$STAMP" ]; then
  echo "[$(date +%H:%M)] arming geometry-cache fingerprint (keeping existing caches);"
  echo "    if you changed probe geometry/holes since they were built, re-run FRESH=1."
  echo "$FP" >"$STAMP"
else
  echo "[$(date +%H:%M)] geometry caches current (fingerprint matched)."
fi

echo "[$(date +%H:%M)] === subject=${STEM} ==="

echo "[$(date +%H:%M)] Phase 1: MRV enumerate + restore + RProp/coarse-fine (no FCL cull) → ${POOL}"
# One mrv_pool_run PROCESS per n_arcs group (ONLY_NARCS), largest first. Each
# group gets a fresh GPU context so the BFC allocator can't fragment across
# groups (the cause of the cross-group 5.9GiB restore OOM); the per-group resume
# logic accumulates records into $POOL. A group with no candidates no-ops.
for NA in $(seq "$MAX_ARCS" -1 1); do
  echo "[$(date +%H:%M)]   Phase-1 group n_arcs=${NA}"
  RUTTER_CONFIG="$CONFIG" RUTTER_HOLES="$HOLES" \
    RUTTER_MAX_ARCS="$MAX_ARCS" RUTTER_MAX_PROBES_PER_ARC="$MAX_PROBES_PER_ARC" \
    RUTTER_ONLY_NARCS="$NA" RUTTER_WELL=thick RUTTER_N_SPINS="$N_SPINS" \
    RUTTER_OUT="$POOL" \
    JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
    uv run --python 3.13 rutter-phase1
done

# A private MPS pipe directory, so this daemon neither finds nor stops another's.
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps-rutter}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-rutter-log}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
stop_mps() { echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true; }
if command -v nvidia-cuda-mps-control >/dev/null; then
  stop_mps; sleep 1
  nvidia-cuda-mps-control -d && sleep 2
  trap stop_mps EXIT
else
  echo "[$(date +%H:%M)] no nvidia-cuda-mps-control: the workers time-slice the card"
fi

echo "[$(date +%H:%M)] Phase 2: IPOPT + thick well on the top ${TOPK} (FCL at end) → ${HANDOFF}"
RUTTER_SOLVER=ipopt RUTTER_CONFIG="$CONFIG" RUTTER_HOLES="$HOLES" \
  RUTTER_POSES="$POOL" RUTTER_OUT="$HANDOFF" \
  RUTTER_WELL=thick RUTTER_POOL=process RUTTER_PLATFORM=gpu \
  RUTTER_GPU_MEM_FRACTION="$GPU_FRACTION" RUTTER_WORKERS="$WORKERS" \
  RUTTER_TOPK="$TOPK" \
  JAX_PLATFORMS=cuda uv run --python 3.13 --extra ipopt rutter-phase2
stop_mps

echo "[$(date +%H:%M)] Emit ${PLANS:-all feasible} trame configs → ${PLANDIR}/"
env RUTTER_CONFIG="$CONFIG" RUTTER_HOLES="$HOLES" RUTTER_HANDOFF="$HANDOFF" \
  ${PLANS:+RUTTER_PLANS="$PLANS"} RUTTER_OUTDIR="$PLANDIR" \
  JAX_PLATFORMS=cpu uv run --python 3.13 rutter-emit

echo "[$(date +%H:%M)] === DONE: pool=${POOL} handoff=${HANDOFF} plans=${PLANDIR}/ ==="
