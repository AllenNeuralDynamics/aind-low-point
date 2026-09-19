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
# Every variable the pipeline reads is RUTTER_-prefixed; the bare spellings
# (CONFIG, OUT, LIMIT, N, WORKERS, POOL, PLATFORM) are set for unrelated reasons
# in ordinary shells and are no longer read.
#
# Usage:
#   RUTTER_CONFIG=examples/837229-config.yml scripts/run_subject_overnight.sh
# Optional env (with defaults):
#   RUTTER_HOLES=scratch/0283-300-04.holes.yml  RUTTER_TOPK=200
#   RUTTER_P2_ITER=1000  RUTTER_COARSE_N=1000  RUTTER_REDUCED_FINE=50
#   RUTTER_FULL_FINE=50  RUTTER_WORKERS=4  RUTTER_PLANS=15
#   FRESH=1   force-clear the geometry caches + pool before running
set -euo pipefail
cd "$(dirname "$0")/.."

# Gotcha 2: XLA preallocation must be DISABLED in the *launch* env (setting it
# from inside Python is too late — JAX then defaults to preallocate=true and
# grabs the whole mem-fraction up front, which starves extra Phase-2 workers /
# any process-pool or MPS run). Exporting it here puts it in the env every
# `uv run` child inherits. Harmless for the default thread-shared Phase-2;
# load-bearing the moment you raise RUTTER_WORKERS or switch RUTTER_POOL=process.
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
# Coverage objective (off by default → legacy plain-sum). Density run:
# RUTTER_RETRO_DENSITY=1 RUTTER_COV_NORM=1 RUTTER_COV_ALPHA=0.2
# RUTTER_COV_WEIGHT=7.
RETRO_DENSITY="${RUTTER_RETRO_DENSITY:-0}"; COV_NORM="${RUTTER_COV_NORM:-0}"
COV_ALPHA="${RUTTER_COV_ALPHA:-0.2}"; COV_WEIGHT="${RUTTER_COV_WEIGHT:-1.0}"
# RUTTER_WORKERS=4: GPU thread-shared knee (re-benched on real 837229 candidates after
# the kernel vmapping). Faster GPU evals make Phase-2 more host/IPOPT-bound, so
# the knee moved in from >4 to 4: W=2 is -18%, W=4 == W=8 (flat). 4 = full
# throughput with less GIL/stream contention + HBM than 8.
TOPK="${RUTTER_TOPK:-200}"; P2_ITER="${RUTTER_P2_ITER:-1000}"
WORKERS="${RUTTER_WORKERS:-4}"
MAX_ARCS="${RUTTER_MAX_ARCS:-3}"  # Phase 1 loops n_arcs down to 1, one process each
MAX_PROBES_PER_ARC="${RUTTER_MAX_PROBES_PER_ARC:-4}"
N_SPINS="${RUTTER_N_SPINS:-16}"
COARSE_N="${RUTTER_COARSE_N:-1000}"; REDUCED_FINE="${RUTTER_REDUCED_FINE:-50}"
FULL_FINE="${RUTTER_FULL_FINE:-50}"
PLANS="${RUTTER_PLANS:-15}"

# Gotcha 1: the atlas + seed caches are keyed only by the config stem, but their
# CONTENTS depend on probe geometry (meshes, kinds), the implant holes, and the
# enumeration caps. Editing any of those without clearing the caches silently
# reuses stale geometry. Stamp a fingerprint and nuke atlas + seeds + pool (the
# pool RESUMES, so it accumulates stale records too) whenever it changes.
# Coverage tuning is env-driven (RUTTER_RETRO_DENSITY / RUTTER_COV_*) and NOT in
# the fingerprint, so coverage variants still share the geometry caches.
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
    RUTTER_COARSE_N="$COARSE_N" RUTTER_REDUCED_FINE="$REDUCED_FINE" \
    RUTTER_FULL_FINE="$FULL_FINE" RUTTER_RETRO_DENSITY="$RETRO_DENSITY" \
    RUTTER_COV_NORM="$COV_NORM" RUTTER_COV_ALPHA="$COV_ALPHA" \
    RUTTER_COV_WEIGHT="$COV_WEIGHT" RUTTER_OUT="$POOL" \
    JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
    uv run --python 3.13 rutter-phase1
done

echo "[$(date +%H:%M)] Phase 2: IPOPT + thick well on top-${TOPK} by min_clear (FCL at end) → ${HANDOFF}"
RUTTER_SOLVER=ipopt RUTTER_CONFIG="$CONFIG" RUTTER_HOLES="$HOLES" \
  RUTTER_POSES="$POOL" RUTTER_OUT="$HANDOFF" RUTTER_SELECT_BY=min_clear \
  RUTTER_WELL=thick RUTTER_POOL=thread RUTTER_PLATFORM=gpu \
  RUTTER_GPU_MEM_FRACTION=0.9 RUTTER_WORKERS="$WORKERS" \
  RUTTER_RETRO_DENSITY="$RETRO_DENSITY" RUTTER_COV_NORM="$COV_NORM" \
  RUTTER_COV_ALPHA="$COV_ALPHA" RUTTER_COV_WEIGHT="$COV_WEIGHT" \
  RUTTER_TOPK="$TOPK" RUTTER_P2_ITER="$P2_ITER" \
  JAX_PLATFORMS=cuda uv run --python 3.13 rutter-phase2

echo "[$(date +%H:%M)] Emit top-${PLANS} trame configs → ${PLANDIR}/"
RUTTER_CONFIG="$CONFIG" RUTTER_HOLES="$HOLES" RUTTER_HANDOFF="$HANDOFF" \
  RUTTER_PLANS="$PLANS" RUTTER_OUTDIR="$PLANDIR" \
  JAX_PLATFORMS=cpu uv run --python 3.13 rutter-emit

echo "[$(date +%H:%M)] === DONE: pool=${POOL} handoff=${HANDOFF} plans=${PLANDIR}/ ==="
