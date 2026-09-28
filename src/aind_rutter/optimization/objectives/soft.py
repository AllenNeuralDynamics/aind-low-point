"""Phase 1 soft-penalty JAX objective, the warm start for Phase 2.

Phase 1 minimizes Phase 2's objective,
:func:`~aind_rutter.optimization.objectives.terms.objective_terms` under the
weights' :class:`ObjectiveSpec`, plus the squared shortfall of the slacks Phase 2
constrains under the weights' :class:`FeasibilitySpec`. At a pose Phase 2 finds
feasible the two objectives are equal, the fixtures aside: Phase 1 checks only
those it is given.

x layout: ``(arc_aps, (ml, sx, sy, off_R, off_A, depth) × P)`` — 6 DOFs
per probe.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Callable, Hashable

import jax
import jax.numpy as jnp
import numpy as np

from aind_rutter.optimization.clearance.aggregate import (
    fixture_slack_gains,
    probe_pair_slack_gains,
)
from aind_rutter.optimization.clearance.boxes import SAT_SMOOTH_BETA
from aind_rutter.optimization.clearance.grids import trilinear_sdf
from aind_rutter.optimization.clearance.poses import (
    pose_from_optimizer_vars,
    spin_deg_from_sxy,
)
from aind_rutter.optimization.clearance.smooth import smooth_abs
from aind_rutter.optimization.clearance.sweep import (
    build_padded_fixture_table,
    build_padded_probe_tables,
    swept_fixture_clearances,
    swept_pair_clearances,
)
from aind_rutter.optimization.geometry.holes import MAX_WALLS_PAD, NO_WALL_OFFSET_MM
from aind_rutter.optimization.objectives.cache_keys import weights_cache_key
from aind_rutter.optimization.objectives.coverage import (
    CoverageData,
    coverage_total_over_probes,
)
from aind_rutter.optimization.objectives.feasibility import FeasibilitySpec
from aind_rutter.optimization.objectives.layout import PHASE1_PER_PROBE_VARS
from aind_rutter.optimization.objectives.objective_spec import ObjectiveSpec
from aind_rutter.optimization.objectives.sdf_inputs import (
    BrainSDFData,
    FixtureSDFData,
)
from aind_rutter.optimization.objectives.slacks import (
    arc_separation_slack,
    brain_slack,
    fixture_clearance_slack,
    ml_separation_slack,
    pair_clearance_slack,
    shortfall,
    threading_slack,
)
from aind_rutter.optimization.objectives.terms import (
    arc_angles_of,
    ml_of,
    ml_pair_separation,
    normalized_coverage,
    objective_terms,
    spin_components_of,
)
from aind_rutter.optimization.objectives.threading import (
    MAX_SECTIONS_PAD,
    MAX_SHANKS_PAD,
    threading_g_matrix,
)

# ---------------------------------------------------------------------------
# Phase 1 weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase1Weights:
    """``objective`` plus penalties on the shortfall of ``spec``'s slacks.

    The penalties are zero exactly where Phase 2's constraints hold, and sized to
    dominate coverage (order 100s against ~17) where they do not.
    """

    spec: FeasibilitySpec
    objective: ObjectiveSpec
    lambda_thread: float = 100.0
    # Per category (body-body, body-shank, shank-shank): deep body overlap and
    # grazing shank contact are different failures, and each needs a signal.
    lambda_clearance: float = 100.0
    lambda_kinematic: float = 100.0
    lambda_clearance_fixture: float = 100.0
    # Near-barrier on every shank tip staying inside the brain, against the
    # coverage gain from inserting deeper. Read only with a brain SDF.
    lambda_brain: float = 500.0


# ---------------------------------------------------------------------------
# JIT-built objective
# ---------------------------------------------------------------------------


_JIT_CACHE: dict[Hashable, tuple[Callable, Callable]] = {}
_CACHE_STATS = {"hits": 0, "misses": 0}

# _objective's positional arg order after x. The batched phase-1 wrapper uses
# this to convert pack_statics' keyword dict into a stable positional arglist.
PACKED_ARG_ORDER: tuple[str, ...] = (
    "target_LPS",
    "pivot_local",
    "arc_idx",
    "tips_local",
    "shank_mask",
    "s_axes",
    "s_centers",
    "s_e1",
    "s_e2",
    "s_cos",
    "s_sin",
    "s_a",
    "s_b",
    "section_mask",
    "same_arc_mask",
    "sdf_grids",
    "sdf_origins",
    "sdf_spacings",
    "sdf_surfaces",
    "shank_obb_centers",
    "shank_obb_halves",
    "sdf_table",
    "w_normals",
    "w_offsets",
    "shank_half_xy",
)

# Keys that vary per candidate. Everything else in PACKED_ARG_ORDER is fixed for
# a probe set and can be shared across a batched chunk.
PACKED_PER_CAND_KEYS: frozenset[str] = frozenset(
    {
        "arc_idx",
        "s_axes",
        "s_centers",
        "s_e1",
        "s_e2",
        "s_cos",
        "s_sin",
        "s_a",
        "s_b",
        "section_mask",
        "same_arc_mask",
        "w_normals",
        "w_offsets",
    }
)


_weights_key = weights_cache_key


def _signature(statics, n_arcs: int, weights: Phase1Weights) -> tuple:
    """Cache key over per-probe SDF/shank-OBB shape info."""
    has_sdf = any(s.sdf_data is not None for s in statics)
    per_probe_sdf_shapes: tuple = ()
    per_probe_shank_counts: tuple = ()
    n_surf = 0
    if has_sdf:
        shapes = []
        counts = []
        for s in statics:
            if s.sdf_data is None:
                shapes.append(None)
                counts.append(0)
            else:
                shapes.append(
                    tuple(int(x) for x in np.asarray(s.sdf_data["grid"]).shape)
                )
                if n_surf == 0:
                    n_surf = int(np.asarray(s.sdf_data["surface"]).shape[0])
                centers = s.sdf_data.get("shank_centers")
                counts.append(
                    int(np.asarray(centers).shape[0]) if centers is not None else 0
                )
        per_probe_sdf_shapes = tuple(shapes)
        per_probe_shank_counts = tuple(counts)
    return (
        len(statics),
        int(n_arcs),
        MAX_SHANKS_PAD,
        MAX_SECTIONS_PAD,
        has_sdf,
        per_probe_sdf_shapes,
        per_probe_shank_counts,
        n_surf,
        _weights_key(weights),
    )


def _build_jit(  # noqa: C901
    signature: tuple,
    weights: Phase1Weights,
    coverage_data: tuple[CoverageData, ...] | None = None,
    fixtures: tuple[FixtureSDFData, ...] = (),
    coverage_n_samples: int = 41,
    brain_sdf: "BrainSDFData | None" = None,
    coverage_ceilings: "tuple[float, ...] | None" = None,
    coverage_weights: "tuple[float, ...] | None" = None,
) -> tuple[Callable, Callable]:
    """Build the (fn, grad) pair for one signature.

    All terms — coverage AND probe-fixture clearance — run inside JAX
    when their respective data is provided. ``coverage_data`` has one
    entry per probe (mixed Gaussian / KDE modes supported). ``fixtures``
    is a tuple of static-in-world fixture SDFs; each contributes a
    probe-vs-fixture clearance penalty and, to the clearance reward, its
    P pairs.

    ``coverage_ceilings`` (one per probe) switches coverage to the
    normalised [0,1] blend ``(1 - cov_alpha)·mean(norm) + cov_alpha·worst(norm)``
    (``weights.objective.cov_alpha``); when ``None`` coverage is the plain
    sum over probes.
    """
    (
        n_probes,
        n_arcs,
        _max_shanks,
        _max_sections,
        has_sdf,
        per_probe_sdf_shapes,
        _per_probe_shank_counts,
        _n_surf,
        _w_key,
    ) = signature

    sdf_pair_list: list[tuple[int, int]] = []
    if has_sdf:
        for i in range(n_probes):
            if per_probe_sdf_shapes[i] is None:
                continue
            for j in range(i + 1, n_probes):
                if per_probe_sdf_shapes[j] is None:
                    continue
                sdf_pair_list.append((i, j))

    lt = float(weights.lambda_thread)
    lc = float(weights.lambda_clearance)
    lk = float(weights.lambda_kinematic)
    lcf = float(weights.lambda_clearance_fixture)
    lbrain = float(weights.lambda_brain)
    spec, objective = weights.spec, weights.objective
    shaft_len = float(spec.shaft_length_mm)
    cov_alpha = float(objective.cov_alpha)
    beta_cov = float(objective.softmin_beta_cov)
    cov_ceilings = (
        jnp.asarray(coverage_ceilings, dtype=jnp.float32)
        if coverage_ceilings is not None
        else None
    )
    cov_weights = (
        jnp.asarray(coverage_weights, dtype=jnp.float32)
        if coverage_weights is not None
        else None
    )
    if brain_sdf is not None:
        brain_grid = jnp.asarray(brain_sdf.grid)
        brain_origin = jnp.asarray(brain_sdf.origin)
        brain_spacing = jnp.asarray(brain_sdf.spacing)

    arc_pairs = jnp.asarray(
        [(a, b) for a in range(n_arcs) for b in range(a + 1, n_arcs)],
        dtype=jnp.int32,
    ).reshape(-1, 2)

    # Padded fixture table (stacked edge-padded grids + n_real_f), built ONCE
    # from the closure-captured fixtures so the fixture × probe loop is a single
    # fused vmap rather than n_fixtures unrolled probe-vmaps.
    fix_table = build_padded_fixture_table(fixtures)

    def _objective(
        x,
        target_LPS,
        pivot_local,
        arc_idx,
        tips_local,
        shank_mask,
        s_axes,
        s_centers,
        s_e1,
        s_e2,
        s_cos,
        s_sin,
        s_a,
        s_b,
        section_mask,
        same_arc_mask,
        sdf_grids,
        sdf_origins,
        sdf_spacings,
        sdf_surfaces,
        shank_obb_centers,
        shank_obb_halves,
        coverage_scale=1.0,
        sdf_table=None,
        w_normals=None,
        w_offsets=None,
        shank_half_xy=None,
    ):
        # ``coverage_scale`` multiplies the coverage gain at run time: 0 for the
        # clearance-first reduced stage, 1 for the full one, so a traced value
        # shares one compiled kernel between them.
        arc_aps = arc_angles_of(x, n_arcs)

        # Pose + threading per probe, vmapped over probes — one
        # pose_from_optimizer_vars + threading_g_matrix subgraph instead of
        # n_probes unrolled copies. x after the arcs is laid out
        # (ml, sx, sy, off_R, off_A, depth) × P (stride PHASE1_PER_PROBE_VARS).
        xp = x[n_arcs : n_arcs + PHASE1_PER_PROBE_VARS * n_probes].reshape(
            n_probes, PHASE1_PER_PROBE_VARS
        )
        aps = arc_aps[arc_idx]  # (P,) per-probe arc AP

        def _pose_thread(
            xp6,
            ap,
            target,
            pivot,
            tips,
            sax,
            scen,
            se1,
            se2,
            scos,
            ssin,
            sa,
            sb,
            sec_m,
            sh_m,
            wn,
            wo,
            sh_xy,
        ):
            R, t = pose_from_optimizer_vars(
                target_LPS=target,
                ap_deg=ap,
                ml_deg=xp6[0],
                spin_deg=spin_deg_from_sxy(xp6[1], xp6[2]),
                offset_R_mm=xp6[3],
                offset_A_mm=xp6[4],
                past_target_mm=xp6[5],
                recording_center_local=pivot,
            )
            g = threading_g_matrix(
                R,
                t,
                tips,
                sax,
                scen,
                se1,
                se2,
                scos,
                ssin,
                sa,
                sb,
                shaft_length_mm=shaft_len,
                w_normals=wn,
                w_offsets=wo,
                shank_half_xy=sh_xy,
            )  # (S, SH)
            valid_g = sec_m[:, None] * sh_m[None, :]
            slack = threading_slack(g, spec)
            excess = shortfall(slack)
            jt = jnp.sum(valid_g * excess * excess)
            # The threading reward reads the same slack.
            return R, t, jt, slack.reshape(-1), valid_g.reshape(-1)

        Rs, ts, _jts, _thread_slacks, _thread_masks = jax.vmap(_pose_thread)(
            xp,
            aps,
            target_LPS,
            pivot_local,
            tips_local,
            s_axes,
            s_centers,
            s_e1,
            s_e2,
            s_cos,
            s_sin,
            s_a,
            s_b,
            section_mask,
            shank_mask,
            w_normals,
            w_offsets,
            shank_half_xy,
        )
        j_thread = jnp.sum(_jts)

        # AP separation (smooth_abs over arc-pair differences)
        if arc_pairs.shape[0] > 0:
            ap_diffs = smooth_abs(arc_aps[arc_pairs[:, 0]] - arc_aps[arc_pairs[:, 1]])
            short_ap = shortfall(arc_separation_slack(ap_diffs, spec))
            j_arc_ap = jnp.sum(short_ap * short_ap)
        else:
            j_arc_ap = jnp.float32(0.0)

        # Intra-arc ML separation
        ml_vals = ml_of(x, n_arcs, n_probes)
        ml_diff = ml_pair_separation(ml_vals)
        short_ml = shortfall(ml_separation_slack(ml_diff, spec))
        j_ml = jnp.sum(same_arc_mask * short_ml * short_ml)

        # Probe-probe clearance, vmapped over the static pair list — ONE dual-rep
        # subgraph instead of C(P,2) Python-unrolled copies (the unrolled loop is
        # ~90% of the autodiff compile; see clearance_sweep + T0 measurement). The
        # padded grid/OBB table is built from the loop-invariant per-probe tuples,
        # so XLA hoists it out of the optimiser's fori_loop. Bit-parity gated.
        clearances = []
        if sdf_pair_list and sdf_table is not None:
            _pa = jnp.asarray([a for a, _ in sdf_pair_list], jnp.int32)
            _pb = jnp.asarray([b for _, b in sdf_pair_list], jnp.int32)
            # The straight-through SAT changes the gradient, not the value.
            _hard, _soft = swept_pair_clearances(
                Rs,
                ts,
                sdf_table,
                _pa,
                _pb,
                **spec.pair_softmin_kwargs(),
                sat_beta=SAT_SMOOTH_BETA,
            )
            _gains = jnp.asarray(probe_pair_slack_gains(), jnp.float32)
            _short = shortfall(pair_clearance_slack(_soft, spec)) * _gains
            j_clear = jnp.sum(_short * _short)
            clearances.append((_hard, _soft))
        else:
            j_clear = jnp.float32(0.0)

        # Probe-vs-fixture clearance, double-vmapped over (fixture × probe) — one
        # dual-rep subgraph in one fused kernel instead of n_fixtures × P unrolled
        # copies (see clearance_sweep.swept_fixture_clearances). Fixtures gather
        # from the padded fix_table; probes from the per-kind sdf_table.
        if fixtures and sdf_table is not None:
            _fidx = [
                i
                for i in range(n_probes)
                if (not has_sdf) or per_probe_sdf_shapes[i] is not None
            ]
            _fh, _fs = swept_fixture_clearances(
                Rs,
                ts,
                sdf_table,
                fix_table,
                _fidx,
                **spec.fixture_softmin_kwargs(),
            )  # (n_fix, n_sdf, 2) in (body, obb)
            _fgains = jnp.asarray(fixture_slack_gains(), jnp.float32)
            _fshort = shortfall(fixture_clearance_slack(_fs, spec)) * _fgains
            j_clear_fixture = jnp.sum(_fshort * _fshort)
            clearances.append((_fh, _fs))
        else:
            j_clear_fixture = jnp.float32(0.0)

        # Brain containment: every shank tip must stay inside the brain
        # (don't puncture through the bottom). Point query of the brain SDF
        # (negative inside) at the world-frame tips; penalize any tip that is
        # not at least ``brain_margin`` inside. One-sided (ReLU²) ⇒ zero cost
        # well inside, so it doesn't perturb already-contained plans. Opposes
        # the depth-greedy coverage cheat (the term is a function of depth).
        j_brain = jnp.float32(0.0)
        if brain_sdf is not None:
            # Batched over probes: (P, max_shanks, 3) world tips in one matmul →
            # one trilinear gather (P × max_shanks points) instead of P calls.
            world_tips = (
                jnp.matmul(tips_local, jnp.transpose(Rs, (0, 2, 1))) + ts[:, None, :]
            )
            d = trilinear_sdf(
                brain_grid, brain_origin, brain_spacing, world_tips
            )  # (P, max_shanks) signed distance, negative inside
            viol = shortfall(brain_slack(d, spec))
            j_brain = jnp.sum(shank_mask * viol * viol)

        # Coverage. ``coverage_data`` is a Python tuple closed over the
        # trace; per-probe mode (Gaussian vs KDE) is fixed at JIT-build time.
        # With per-probe ceilings, each region is normalised to a fraction-
        # of-achievable and a soft-min floor protects the worst region; else
        # it's the legacy plain sum across probes.
        if coverage_data is not None and cov_ceilings is not None:
            coverage_total = normalized_coverage(
                Rs,
                ts,
                tips_local,
                shank_mask,
                coverage_data,
                cov_ceilings,
                n_samples=coverage_n_samples,
                alpha=cov_alpha,
                softmin_beta=beta_cov,
                weights=cov_weights,
            )
        elif coverage_data is not None:
            # vmapped when all probes share a coverage mode (all-Gaussian /
            # all-KDE uniform grid); unrolled loop otherwise. Same result.
            coverage_total = coverage_total_over_probes(
                Rs,
                ts,
                tips_local,
                shank_mask,
                coverage_data,
                n_samples=coverage_n_samples,
            )
        else:
            coverage_total = jnp.float32(0.0)

        sx_arr, sy_arr = spin_components_of(x, n_arcs, n_probes)
        # _thread_slacks/_thread_masks are (P, S*SH) from the vmap, so the reward
        # sees each probe's tightest shank.
        return objective_terms(
            objective,
            coverage_total,
            arc_aps,
            ml_vals,
            sx_arr,
            sy_arr,
            tuple(clearances),
            _thread_slacks,
            _thread_masks,
            coverage_scale=coverage_scale,
        ) + (
            lt * j_thread
            + lc * j_clear
            + lcf * j_clear_fixture
            + lk * (j_arc_ap + j_ml)
            + lbrain * j_brain
        )

    jit_obj = jax.jit(_objective)
    jit_grad = jax.jit(jax.grad(_objective))
    return jit_obj, jit_grad


# Per-probe-fixed SDF tuples + swept-pair table, cached by probe sdf-data
# identity so they're built ONCE per probe set, not per candidate (see
# pack_statics). Keyed by id() of each probe's SDFPayload.
_SDF_PACK_CACHE: dict = {}


def _evict_pack_entry_with_its_payloads(statics) -> None:
    """Arrange for this entry to go when any payload its key names is collected.

    A probe with no SDF contributes ``id(None)``, a constant that outlives
    everything, so there is nothing to finalize for it.
    """
    for s in statics:
        if s.sdf_data is not None:
            weakref.finalize(s.sdf_data, _drop_pack_entries_for, id(s.sdf_data))


def _drop_pack_entries_for(payload_id: int) -> None:
    """Forget every packed table built from a payload that has been collected.

    The key is a tuple of payload ids, so an entry outliving its payloads would
    be read by whatever object next lands on one of those addresses.
    """
    for key in [k for k in _SDF_PACK_CACHE if payload_id in k[0]]:
        _SDF_PACK_CACHE.pop(key, None)


def pack_statics(
    statics, n_arcs: int, build_sdf: bool = True, build_table: bool = True
) -> dict:
    """Pack per-candidate static data into padded jnp tensors. Mirrors
    The reduced-objective packer convention is preserved so per-probe data
    lines up.

    ``build_sdf`` / ``build_table`` control whether the per-probe SDF/OBB tuples
    and the padded swept-pair ``sdf_table`` are built. They are per-probe-fixed
    (identical across candidates of the same probe set), so the batched
    ``build_arglist`` — which reuses a single shared template for those and only
    needs the per-candidate keys (arc_idx, hole sections) — passes
    ``build_sdf=False`` to skip the per-chunk grid→device conversion + table
    build. ``build_table`` requires ``build_sdf``.
    """
    P = len(statics)
    max_shanks, max_sections = MAX_SHANKS_PAD, MAX_SECTIONS_PAD
    target_LPS = np.zeros((P, 3), dtype=np.float32)
    pivot_local = np.zeros((P, 3), dtype=np.float32)
    arc_idx = np.zeros(P, dtype=np.int32)
    tips_local = np.zeros((P, max_shanks, 3), dtype=np.float32)
    shank_mask = np.zeros((P, max_shanks), dtype=np.float32)
    s_axes = np.zeros((P, max_sections, 3), dtype=np.float32)
    s_axes[:, :, 2] = 1.0
    s_centers = np.zeros((P, max_sections, 3), dtype=np.float32)
    s_e1 = np.zeros((P, max_sections, 3), dtype=np.float32)
    s_e1[:, :, 0] = 1.0
    s_e2 = np.zeros((P, max_sections, 3), dtype=np.float32)
    s_e2[:, :, 1] = 1.0
    s_cos = np.ones((P, max_sections), dtype=np.float32)
    s_sin = np.zeros((P, max_sections), dtype=np.float32)
    s_a = np.ones((P, max_sections), dtype=np.float32)
    s_b = np.ones((P, max_sections), dtype=np.float32)
    section_mask = np.zeros((P, max_sections), dtype=np.float32)
    w_normals = np.zeros((P, MAX_WALLS_PAD, 3), dtype=np.float32)
    w_offsets = np.full((P, MAX_WALLS_PAD), NO_WALL_OFFSET_MM, dtype=np.float32)
    shank_half_xy = np.zeros((P, 2), dtype=np.float32)
    any_corners = any(s.shank_half_xy is not None for s in statics)
    for i, s in enumerate(statics):
        target_LPS[i] = s.target_LPS
        pivot_local[i] = s.pivot_local
        arc_idx[i] = s.arc_idx
        ns = min(int(s.shank_tips_local.shape[0]), max_shanks)
        if ns > 0:
            tips_local[i, :ns] = s.shank_tips_local[:ns]
            shank_mask[i, :ns] = 1.0
        nsec = min(int(s.section_axes.shape[0]), max_sections)
        if nsec > 0:
            s_axes[i, :nsec] = s.section_axes[:nsec]
            s_centers[i, :nsec] = s.section_centers[:nsec]
            s_e1[i, :nsec] = s.section_e1[:nsec]
            s_e2[i, :nsec] = s.section_e2[:nsec]
            s_cos[i, :nsec] = s.section_cos_theta[:nsec]
            s_sin[i, :nsec] = s.section_sin_theta[:nsec]
            s_a[i, :nsec] = s.section_a[:nsec]
            s_b[i, :nsec] = s.section_b[:nsec]
            section_mask[i, :nsec] = 1.0
        w_normals[i] = s.wall_normals
        w_offsets[i] = s.wall_offsets
        if s.shank_half_xy is not None:
            shank_half_xy[i] = s.shank_half_xy

    same_arc_mask = np.zeros((P, P), dtype=np.float32)
    for i in range(P):
        for j in range(i + 1, P):
            if statics[i].arc_idx == statics[j].arc_idx:
                same_arc_mask[i, j] = 1.0

    out = dict(
        target_LPS=jnp.asarray(target_LPS),
        pivot_local=jnp.asarray(pivot_local),
        arc_idx=jnp.asarray(arc_idx),
        tips_local=jnp.asarray(tips_local),
        shank_mask=jnp.asarray(shank_mask),
        s_axes=jnp.asarray(s_axes),
        s_centers=jnp.asarray(s_centers),
        s_e1=jnp.asarray(s_e1),
        s_e2=jnp.asarray(s_e2),
        s_cos=jnp.asarray(s_cos),
        s_sin=jnp.asarray(s_sin),
        s_a=jnp.asarray(s_a),
        s_b=jnp.asarray(s_b),
        section_mask=jnp.asarray(section_mask),
        same_arc_mask=jnp.asarray(same_arc_mask),
        w_normals=jnp.asarray(w_normals),
        w_offsets=jnp.asarray(w_offsets),
        # None keeps a centreline-only problem on the kernel's centreline path.
        shank_half_xy=jnp.asarray(shank_half_xy) if any_corners else None,
    )
    # SDF + shank-OBB tuples + the padded swept-pair table. Per-probe-FIXED
    # (identical across candidates of the same probe set): skipped when
    # build_sdf=False, and otherwise cached by per-probe sdf-data identity so the
    # grid→device conversion + the (edge-pad + stack) table build happen ONCE per
    # probe set rather than per candidate. The grids are edge-padded so trilinear
    # is bit-exact with the unpadded grids (real extent in ``real_shapes``); see
    # clearance_sweep. The unrolled fixture loop uses the per-probe tuples.
    if not build_sdf:
        return out
    key = (tuple(id(s.sdf_data) for s in statics), bool(build_table))
    sdf_part = _SDF_PACK_CACHE.get(key)
    if sdf_part is None:
        _evict_pack_entry_with_its_payloads(statics)
        sdf_grids, sdf_origins, sdf_spacings, sdf_surfaces = [], [], [], []
        shank_obb_centers, shank_obb_halves, sdf_clearance = [], [], []
        # Dedup the grid→device conversion by probe KIND: the SDF/OBB/surface are
        # a deterministic function of the kind geometry (canonical-local), so each
        # kind's jnp arrays are built ONCE from the first occurrence and the refs
        # are shared. The swept-pair table then dedups by object identity to
        # N_kinds distinct grids (vs P) — less HBM, less padding, fewer host→
        # device conversions. (The per-probe ProbeSDF objects are otherwise all
        # distinct, so id-based dedup alone would never fire.)
        by_kind: dict = {}
        for s in statics:
            if s.sdf_data is not None:
                ent = by_kind.get(s.kind)
                if ent is None:
                    ent = (
                        jnp.asarray(s.sdf_data["grid"], dtype=jnp.float32),
                        jnp.asarray(s.sdf_data["origin"], dtype=jnp.float32),
                        jnp.asarray(s.sdf_data["spacing"], dtype=jnp.float32),
                        jnp.asarray(s.sdf_data["surface"], dtype=jnp.float32),
                        jnp.asarray(
                            s.sdf_data.get(
                                "shank_centers", np.zeros((0, 3), dtype=np.float32)
                            ),
                            dtype=jnp.float32,
                        ),
                        jnp.asarray(
                            s.sdf_data.get(
                                "shank_halves", np.zeros((0, 3), dtype=np.float32)
                            ),
                            dtype=jnp.float32,
                        ),
                        s.sdf_data.get("clearance"),
                    )
                    by_kind[s.kind] = ent
                sdf_grids.append(ent[0])
                sdf_origins.append(ent[1])
                sdf_spacings.append(ent[2])
                sdf_surfaces.append(ent[3])
                shank_obb_centers.append(ent[4])
                shank_obb_halves.append(ent[5])
                sdf_clearance.append(ent[6])
            else:
                sdf_grids.append(jnp.zeros((2, 2, 2), dtype=jnp.float32))
                sdf_origins.append(jnp.zeros(3, dtype=jnp.float32))
                sdf_spacings.append(jnp.float32(1.0))
                sdf_surfaces.append(jnp.zeros((1, 3), dtype=jnp.float32))
                shank_obb_centers.append(jnp.zeros((0, 3), dtype=jnp.float32))
                shank_obb_halves.append(jnp.zeros((0, 3), dtype=jnp.float32))
                sdf_clearance.append(None)
        sdf_part = {
            "sdf_grids": tuple(sdf_grids),
            "sdf_origins": tuple(sdf_origins),
            "sdf_spacings": tuple(sdf_spacings),
            "sdf_surfaces": tuple(sdf_surfaces),
            "shank_obb_centers": tuple(shank_obb_centers),
            "shank_obb_halves": tuple(shank_obb_halves),
        }
        if build_table:
            sdf_part["sdf_table"] = build_padded_probe_tables(
                sdf_part["sdf_grids"],
                sdf_part["sdf_origins"],
                sdf_part["sdf_spacings"],
                sdf_part["sdf_surfaces"],
                sdf_part["shank_obb_centers"],
                sdf_part["shank_obb_halves"],
                # Not a packed argument: only the table carries these samples.
                clearance=tuple(sdf_clearance),
            )
        _SDF_PACK_CACHE[key] = sdf_part
    out.update(sdf_part)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cache_stats() -> dict:
    return {**_CACHE_STATS, "entries": len(_JIT_CACHE)}
