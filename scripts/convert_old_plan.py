"""Convert an old-style GUI insertion-plan CSV to a low-point plan
fragment (YAML).

The old CSV (e.g. ``836656_YB_GuiInsertionPlan_2026-05-05T11-26-30.csv``)
holds one probe row with columns:

    structure, probe_type, ap_arc_id, ap_angle, ap_rig_angle, ml_angle,
    spin, target_pt_R/A/S, ideal_pt_R/A/S, hole, distance_past_target

This script emits the ``plan:`` block (arcs + probes) of a low-point
config. By default it writes only that fragment to stdout, suitable
for pasting into a hand-written subject-config YAML; with ``--base``
it splices the plan into a copy of an existing config (e.g.
``examples/786864-config.yml``) and writes a full standalone YAML.

Per-kind active-recording centers (mm from tip) are subtracted from
``distance_past_target`` so the new value matches the
recording-center-past-target semantic introduced by the pivot redesign.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

# Map CSV's ``probe_type`` strings (with bank suffixes) to the kind
# strings recognised by ``RECORDING_GEOMETRY``. Bank info is dropped at
# this layer; the user can re-introduce it once we extend the kind
# registry to per-bank entries.
KIND_MAP: dict[str, str] = {
    "2.1": "2.1",
    "2.4": "2.4",
    # Custom-holder (alpha) and dovetail variants now map to distinct
    # kinds so the catalog can render the right holder mesh per probe.
    "quadbase0": "quadbase-alpha",
    "quadbase1": "quadbase-alpha",
    "quadbase2": "quadbase-alpha",
    "quadbase3": "quadbase-alpha",
    "quadbase_dovetail0": "quadbase-dovetail",
    "quadbase_dovetail1": "quadbase-dovetail",
    "quadbase_dovetail2": "quadbase-dovetail",
    "quadbase_dovetail3": "quadbase-dovetail",
}


# Index of the shank that the OLD CSV producer's probe model had at its
# local origin, **expressed in our canonical orientation** (shank-0 at
# +y=0, shank-3 at +y=0.75 mm). The reference script
# (LoadTransform_PlanPointInsertion_*) loads
# ``Quadbase_*_centeredOnShank{N}.obj`` per variant and places that
# OBJ's local origin (= shank-N tip in *its* labeling) at ``ideal_pt``.
# The dovetail OBJs respect our rig convention (shank-0 at -y in the
# OBJ's native frame, which after reference's column-permute lands at
# +y=0), so ``quadbase_dovetailN`` → canonical shank N. The
# *customHolder* OBJs have a mirroring issue along the shank-row
# direction — what's labeled "centeredOnShank0" actually has shank-3 at
# origin in our canonical frame. So ``quadbaseN`` (customHolder, no
# ``_dovetail`` suffix) → canonical shank ``3 − N``.
OLD_CENTERED_SHANK_INDEX: dict[str, int] = {
    "2.1": 0,  # single shank
    "2.4": 0,
    "quadbase": 0,  # default (no suffix → assume shank-0)
    # customHolder variants are row-flipped: N → 3−N.
    "quadbase0": 3,
    "quadbase1": 2,
    "quadbase2": 1,
    "quadbase3": 0,
    # Dovetail variants line up with our canonical shank order.
    "quadbase_dovetail0": 0,
    "quadbase_dovetail1": 1,
    "quadbase_dovetail2": 2,
    "quadbase_dovetail3": 3,
}


# Number of shanks per kind (after KIND_MAP normalisation), used to
# compute the canonical row centroid ``(N_shanks − 1)·pitch/2``.
N_SHANKS_BY_KIND: dict[str, int] = {
    "2.1": 1,
    "2.4": 4,
    "quadbase": 4,
    "quadbase-alpha": 4,
    "quadbase-dovetail": 4,
}
SHANK_PITCH_MM = 0.25  # AIND probes use 250 µm shank-row pitch.


# Default recording-array center along the shaft (mm) per kind. Matches
# ``aind_low_point.optimization.recording.RECORDING_GEOMETRY``.
# Subtracted from CSV's ``distance_past_target`` to convert the OLD
# tip-past-target semantic to the NEW recording-center-past-target one.
ACTIVE_CENTER_MM: dict[str, float] = {
    "2.1": 1.6325,
    "2.4": 0.5525,
    "quadbase": 1.6325,
    "quadbase-alpha": 1.6325,
    "quadbase-dovetail": 1.6325,
}


# Spin offset (degrees) added to the CSV's raw spin per probe_type.
#
# Reference's QB_Centering customHolder + dovetail OBJs have their shank
# row extending in raw +z; our main-dir variants
# (``_moreacuratewidth.obj``) have it in raw −z. After canonicalization
# the two meshes are mirrored about the shaft axis in canonical local
# y. Adding 180° about the shaft compensates: with the *same* CSV ap/ml
# and a 180°-shifted spin, our pipeline produces the same physical
# placement for ALL four shanks (not just the named one — the named
# shank already matched under the OLD_CENTERED_SHANK_INDEX flip alone,
# because the converter shifts ``target_pt`` to land that shank at
# the right spot; the OTHER shanks were on the wrong side of it).
#
# The companion ``target_lps`` shift in :func:`build_probes` is rotated
# by the post-offset spin, so it self-consistently re-anchors the
# named shank at the CSV's ``target_pt`` even after the 180° rotation.
#
# Default: 0° for everything. Quadbase variants are flipped on the
# assumption that both reference mesh families share the raw +z row
# convention. If a future CSV references a probe model that does NOT
# follow this pattern, override here.
SPIN_OFFSET_BY_PROBE_TYPE: dict[str, float] = {
    "quadbase0": 180.0,
    "quadbase1": 180.0,
    "quadbase2": 180.0,
    "quadbase3": 180.0,
    "quadbase_dovetail0": 180.0,
    "quadbase_dovetail1": 180.0,
    "quadbase_dovetail2": 180.0,
    "quadbase_dovetail3": 180.0,
}


ARC_LETTERS = list("abcdefghijklmnop")


def normalize_kind(probe_type: str) -> str:
    return KIND_MAP.get(probe_type, probe_type)


def parse_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def build_arcs(
    rows: list[dict[str, str]],
) -> tuple[dict[str, float], dict[float, str]]:
    """Group rows by ``ap_arc_id`` and assign each a letter (a, b, ...).

    Returns ``(arcs_dict, arc_id_to_letter)`` where ``arcs_dict`` maps
    letter→AP angle (used in ``plan.arcs``) and ``arc_id_to_letter``
    maps the CSV's float arc id to the assigned letter.
    """
    seen: dict[float, str] = {}
    arcs: dict[str, float] = {}
    # Sort by arc_id so letter assignment is stable across runs.
    for row in sorted(rows, key=lambda r: float(r["ap_arc_id"])):
        arc_id = float(row["ap_arc_id"])
        if arc_id in seen:
            continue
        letter = ARC_LETTERS[len(seen)]
        seen[arc_id] = letter
        arcs[letter] = float(row["ap_angle"])
    return arcs, seen


def build_probes(
    rows: list[dict[str, str]],
    arc_id_to_letter: dict[float, str],
    *,
    target_mode: str = "inline",
) -> tuple[dict[str, dict], list[str]]:
    """Build the ``plan.probes`` dict and a list of comments noting
    each probe's hole assignment from the CSV.

    Two conversions matter for preserving the OLD CSV's *physical*
    probe pose under the new pivot semantics:

    1. ``past_target_mm = old_dpt − active_center_z``. Old depth was
       "shank-0 tip past target"; new depth is "recording center past
       target". They differ by the active-region center along the
       shaft.

    2. ``target_point_RAS`` is shifted by the row-centroid term
       ``R @ (centroid_x, centroid_y, 0)`` (in world). Without this
       shift the new pivot puts the *row centroid* at the CSV's
       ``ideal_pt`` instead of *shank-0 tip*, which presents as an
       ML offset (and a visible orbit when spin changes — the row
       centroid stays put but shank-0 traces a circle of radius
       ``centroid_x`` around it). The shift cancels that.

    ``target_mode``:
      - ``"inline"`` (default): use ``ideal_pt + centroid-shift`` as a
        literal ``point_RAS`` per probe. ``offsets_RA = (0, 0)``.
      - ``"node"``: target ``target:{HEMI}:{structure}``, with
        ``offsets_RA = ideal − target_pt`` (RAS xy). NB: the resolved
        hemisphere centroid will rarely match the CSV producer's
        exactly, so the probe lands at *our centroid + offset*, not
        at CSV's ``ideal_pt``. Use only when faithful reproduction of
        the CSV pose is not the goal.
    """
    if target_mode not in ("inline", "node"):
        raise ValueError(f"unknown target_mode: {target_mode!r}")

    probes: dict[str, dict] = {}
    seen_structures: dict[str, int] = defaultdict(int)
    hole_notes: list[str] = []
    for row in rows:
        structure = row["structure"]
        kind = normalize_kind(row["probe_type"])
        arc_letter = arc_id_to_letter[float(row["ap_arc_id"])]

        # Pivot redesign (746864b): NEW past_target_mm = tip-past-target
        # − recording_center_z. ``ACTIVE_CENTER_MM`` is the local-frame
        # active-region centre along the shaft.
        old_dpt = float(row["distance_past_target"])
        active_z = ACTIVE_CENTER_MM.get(kind, 0.0)
        past_target_mm = round(old_dpt - active_z, 4)

        # The CSV variant suffix records which shank the rig was
        # anchored on. Lifted onto ``position_bearing_shank``
        # (1-indexed) so the runtime knows which shank to report in
        # tip-RAS / brain-depth readouts.
        n_centered = OLD_CENTERED_SHANK_INDEX.get(row["probe_type"], 0)
        position_bearing_shank = int(n_centered) + 1

        from aind_anatomical_utils.coordinate_systems import (
            convert_coordinate_system,
        )
        from aind_mri_utils.arc_angles import arc_angles_to_affine

        ap_deg = float(row["ap_angle"])
        ml_deg = float(row["ml_angle"])
        # Use the *same* spin the YAML writes (raw + per-variant offset,
        # wrapped to [-180, 180]). The shift vector lives in the probe's
        # local frame, so it has to be rotated by the same R the runtime
        # will use when it reads back the YAML. ``SPIN_OFFSET_BY_PROBE_TYPE``
        # accounts for the 180° row-direction mismatch between the
        # reference QB_Centering meshes and our main-dir variants.
        raw_spin_deg = float(row["spin"])
        spin_offset = SPIN_OFFSET_BY_PROBE_TYPE.get(row["probe_type"], 0.0)
        wrapped_spin = ((raw_spin_deg + spin_offset + 180.0) % 360.0) - 180.0
        R = arc_angles_to_affine(ap_deg, ml_deg, wrapped_spin)
        n_shanks = N_SHANKS_BY_KIND.get(kind, 1)
        centroid_y = (n_shanks - 1) * SHANK_PITCH_MM / 2.0
        # Centroid shift: the runtime puts the recording-array centre
        # (= ``(0, centroid_y, active_z)`` in canonical local) at the
        # inline target. The CSV's ``target_pt`` is where shank-N's
        # tip should land (N = ``OLD_CENTERED_SHANK_INDEX[probe_type]``).
        # Shifting the inline target by ``R @ (0, centroid_y −
        # N·pitch, 0)`` makes shank-N (not the row centroid) coincide
        # with target_pt at past_target_mm + active_z = old_dpt.
        shift_local = np.array(
            [0.0, centroid_y - n_centered * SHANK_PITCH_MM, 0.0],
            dtype=np.float64,
        )
        shift_lps = R @ shift_local

        # Anchor on ``target_pt`` (the rig's executed landing).
        # Reference k3d builds the same anchor via ``[-ideal_R, -ideal_A,
        # ideal_S] + (x_offset, y_offset, 0)`` where the xy offsets
        # cancel ideal back to target_R/A; the S column is identical
        # between target_pt and ideal_pt in the rig's CSV format, so
        # using target_pt for all three axes matches reference exactly.
        target_pt_lps = convert_coordinate_system(
            np.array(
                [
                    [
                        float(row["target_pt_R"]),
                        float(row["target_pt_A"]),
                        float(row["target_pt_S"]),
                    ]
                ],
                dtype=np.float64,
            ),
            "RAS",
            "LPS",
        ).ravel()
        target_lps = target_pt_lps + shift_lps
        target_ras = convert_coordinate_system(
            target_lps.reshape(1, 3), "LPS", "RAS"
        ).ravel()

        n = seen_structures[structure]
        seen_structures[structure] += 1
        probe_name = structure if n == 0 else f"{structure}_{n + 1}"
        hole_id = row.get("hole", "?").rstrip("0").rstrip(".")
        hole_notes.append(
            f"#   {probe_name}: hole {hole_id}  "
            f"(probe_type={row['probe_type']}, "
            f"position_bearing_shank={position_bearing_shank}, "
            f"past_target_mm={past_target_mm:.3f})"
        )

        if target_mode == "inline":
            target_ref = {
                "kind": "inline",
                "point_RAS": [
                    round(float(target_ras[0]), 4),
                    round(float(target_ras[1]), 4),
                    round(float(target_ras[2]), 4),
                ],
            }
            offsets_RA = [0.0, 0.0]
        else:
            hemi = "L" if float(row["ideal_pt_R"]) < 0 else "R"
            target_ref = {
                "kind": "node",
                "key": f"target:{hemi}:{structure}",
            }
            offsets_RA = [
                round(float(row["ideal_pt_R"]) - float(row["target_pt_R"]), 4),
                round(float(row["ideal_pt_A"]) - float(row["target_pt_A"]), 4),
            ]
        # ``wrapped_spin`` already includes ``SPIN_OFFSET_BY_PROBE_TYPE``
        # (180° for quadbase variants) so that all four shanks — not
        # just the named one — land in their physically-correct
        # positions vs the reference k3d rendering. The variant suffix
        # is still lifted onto ``position_bearing_shank`` via
        # ``OLD_CENTERED_SHANK_INDEX`` for the GUI's tip-RAS readout.
        probes[probe_name] = {
            "kind": kind,
            "arc": arc_letter,
            "spin": int(round(wrapped_spin)),
            "slider_ml": float(row["ml_angle"]),
            "past_target_mm": past_target_mm,
            "offsets_RA": offsets_RA,
            "target": target_ref,
            "position_bearing_shank": position_bearing_shank,
        }
    return probes, hole_notes


def header_comments(
    csv_path: Path,
    arcs: dict[str, float],
    arc_id_to_letter: dict[float, str],
    structures: list[str],
    hole_notes: list[str],
) -> str:
    arc_lines = "\n".join(
        f"#   {letter}: {arcs[letter]:+.1f}°  (CSV ap_arc_id={arc_id})"
        for arc_id, letter in arc_id_to_letter.items()
    )
    note_block = "\n".join(hole_notes)
    return (
        f"# Plan converted from {csv_path.name}\n"
        f"#\n"
        f"# Structures targeted: {', '.join(structures)}\n"
        f"#\n"
        f"# Arc assignment (letter ← CSV ap_arc_id):\n"
        f"{arc_lines}\n"
        f"#\n"
        f"# Per-probe notes (hole assignment + depth conversion):\n"
        f"{note_block}\n"
        f"#\n"
        f"# Past-target depths converted from tip-past-target (old)\n"
        f"# to recording-center-past-target (new) by subtracting the\n"
        f"# active-region center per kind.\n"
        f"#\n"
    )


def emit_plan_only(
    csv_path: Path,
    output: Path | None = None,
    *,
    target_mode: str = "inline",
) -> str:
    rows = parse_csv(csv_path)
    arcs, arc_id_to_letter = build_arcs(rows)
    probes, hole_notes = build_probes(rows, arc_id_to_letter, target_mode=target_mode)
    structures = sorted({r["structure"] for r in rows})
    body = yaml.safe_dump(
        {"plan": {"arcs": arcs, "probes": probes}},
        sort_keys=False,
        default_flow_style=False,
        width=120,
    )
    out = (
        header_comments(csv_path, arcs, arc_id_to_letter, structures, hole_notes) + body
    )
    if output is not None:
        output.write_text(out)
        print(f"Wrote plan fragment to {output}", file=sys.stderr)
    return out


def emit_full_config(  # noqa: C901
    csv_path: Path,
    base_config_path: Path,
    output: Path,
    *,
    mouse: str | None = None,
    target_mode: str = "inline",
) -> None:
    """Splice the converted plan into a copy of ``base_config_path`` and
    write a full standalone config to ``output``.

    ``mouse`` overrides ``paths.mouse`` in the base config; if not
    given, we infer it from the CSV filename's leading numeric token.

    ``target_mode``: see :func:`build_probes`. ``inline`` (default)
    drops structure assets / structure targets from the spliced
    config, since each probe carries its own RAS target point and no
    CCF mask files are required.
    """
    rows = parse_csv(csv_path)
    arcs, arc_id_to_letter = build_arcs(rows)
    structures = sorted({r["structure"] for r in rows})

    if mouse is None:
        m = re.match(r"(\d+)", csv_path.name)
        if m is None:
            raise SystemExit(
                f"Could not infer mouse id from CSV filename "
                f"{csv_path.name!r}; pass --mouse explicitly."
            )
        mouse = m.group(1)

    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    if "paths" in config and "mouse" in config["paths"]:
        config["paths"]["mouse"] = str(mouse)

    # Patch the older "com_plane.h5" path convention to the modern
    # "${paths.mouse}_com_plane.h5" naming. The 786864 config predates
    # mouse-prefixed transform files; newer subjects (e.g. 836656) use
    # the prefixed form. Walk the transforms tree and rewrite.
    transforms = config.get("transforms")
    if isinstance(transforms, dict):
        for tspec in transforms.values():
            seq = tspec.get("sequence") if isinstance(tspec, dict) else None
            if not isinstance(seq, list):
                continue
            for step in seq:
                if not isinstance(step, dict):
                    continue
                p = step.get("path")
                if isinstance(p, str) and p.endswith("/com_plane.h5"):
                    step["path"] = p.replace(
                        "/com_plane.h5", "/${paths.mouse}_com_plane.h5"
                    )

    # Resolve OmegaConf-style ``${paths.KEY}`` tokens by interpolating
    # from ``config["paths"]``. Used for both file-existence checks
    # and detection of subject-specific data (rabies CSV, CCF
    # annotation volume).
    paths_dict = config.get("paths", {}) or {}

    def resolve_token(s: str) -> str:
        if not isinstance(s, str):
            return s
        for _ in range(5):
            new_s = s
            for k, v in paths_dict.items():
                if not isinstance(v, str):
                    continue
                new_s = new_s.replace("${paths." + str(k) + "}", v)
            if new_s == s:
                return s
            s = new_s
        return s

    def resolve_path_value(key: str) -> str:
        """Resolve a paths.KEY value, walking through ``${.OTHER}`` refs."""
        v = paths_dict.get(key)
        if not isinstance(v, str):
            return ""
        # Local refs (${.X}) point to siblings in the same paths dict.
        for _ in range(5):
            new_v = v
            for k2, v2 in paths_dict.items():
                if not isinstance(v2, str):
                    continue
                new_v = new_v.replace("${." + str(k2) + "}", v2)
            if new_v == v:
                break
            v = new_v
        return v

    annotations_path = resolve_path_value("annotations_path")

    # Patch the rabies-tracing CSV path: 786864 nested it under
    # ``OldRabiesTracing/``, but newer subjects (e.g. 836656) carry
    # the file at ``${annotations_path}/${mouse}_rabies_*.csv``.
    # Rewrite if the modern variant exists.
    if annotations_path:
        modern_rabies = (
            Path(annotations_path) / f"{mouse}_rabies_pts_from_698928_LPS.csv"
        )
        if modern_rabies.exists():
            for asset in config.get("assets", []):
                if not isinstance(asset, dict):
                    continue
                src = asset.get("src")
                if (
                    isinstance(src, str)
                    and "OldRabiesTracing/${paths.mouse}_rabies_pts" in src
                ):
                    asset["src"] = src.replace(
                        "OldRabiesTracing/${paths.mouse}_rabies_pts",
                        "${paths.mouse}_rabies_pts",
                    )

    def asset_src_exists(asset: dict) -> bool:
        src = asset.get("src")
        if not isinstance(src, str):
            return True  # nothing to check (e.g. atlas_dir entries)
        resolved = resolve_token(src)
        if "${" in resolved:
            return True  # incomplete resolution — let runtime decide
        if "{" in resolved:
            return True  # bulk pattern with {name} slots
        return Path(resolved).exists()

    config["assets"] = [
        a
        for a in config.get("assets", [])
        if not isinstance(a, dict) or asset_src_exists(a)
    ]

    # If the subject has an ANTs-warped CCF annotation volume,
    # auto-add per-structure asset entries that mesh each requested
    # CCF region out of it via the ``ccf_annotation_region`` loader.
    # This is strictly nicer than relying on pre-extracted per-region
    # mask NRRDs (which 836656 doesn't have).
    ccf_anno_rel = "ccfv3/ccf_annotation_in_subject.nii.gz"
    ccf_anno_abs = Path(annotations_path) / ccf_anno_rel if annotations_path else None
    has_ccf_annotation = ccf_anno_abs is not None and ccf_anno_abs.exists()

    # ``auto`` defaults to inline so each probe lands at exactly the
    # CSV's ``ideal_pt_RAS``. Node mode reroutes the probe through our
    # ``target:L:STRUCTURE`` (from ``hemisphere_center_mass``) plus
    # ``offsets_RA = ideal − target_pt``, which silently introduces a
    # bias whenever our resolved centroid differs from the CSV
    # producer's by even a fraction of a mm — almost always the case
    # in practice. inline preserves CSV positions exactly.
    if target_mode == "auto":
        target_mode = "inline"

    if target_mode == "node" and not has_ccf_annotation:
        raise SystemExit(
            f"--targets node requires the warped CCF annotation volume "
            f"at {ccf_anno_abs}, which does not exist. Use --targets "
            f"inline (or auto) instead, or generate the warped volume."
        )

    # Build probes with the resolved target mode.
    probes, hole_notes = build_probes(rows, arc_id_to_letter, target_mode=target_mode)

    # Drop the base config's structure-mask asset entries and
    # structure-derived target groups before adding new ones —
    # 786864 used pre-extracted subject-space NRRD masks (which
    # 836656 doesn't have), so the mask-based entries inherited from
    # the base would break at runtime. The CCF-annotation block below
    # re-adds them in ``node`` target mode when the warped annotation
    # volume exists.
    config["assets"] = [
        a
        for a in config.get("assets", [])
        if not (
            isinstance(a, dict)
            and (
                # Multi-key form: ``keys: [structure:A, structure:B, ...]``
                (
                    "keys" in a
                    and any(
                        isinstance(k, str) and k.startswith("structure:")
                        for k in a["keys"]
                    )
                )
                # Single-key form (what we emit): ``key: structure:A``
                or (isinstance(a.get("key"), str) and a["key"].startswith("structure:"))
            )
        )
    ]
    config["targets"] = [
        t
        for t in config.get("targets", [])
        if not (
            isinstance(t, dict)
            and isinstance(t.get("derive_from"), list)
            and any(
                isinstance(k, str) and k.startswith("structure:")
                for k in t["derive_from"]
            )
        )
    ]

    # When a warped CCF annotation volume is available we always emit
    # per-structure assets (for rendering) AND hemisphere-specific
    # target nodes (for retargeting via the trame UI's dropdown),
    # *regardless* of the probe's initial-target mode. Inline-mode
    # probes can still be retargeted to ``target:L:STRUCTURE`` or
    # ``target:R:STRUCTURE`` from the UI; they just don't *start*
    # there.
    if has_ccf_annotation:
        config["paths"]["ccf_annotation_path"] = "${.annotations_path}/" + ccf_anno_rel
        # Look up the per-region CCF colors from the bundled ontology
        # so each structure renders in its Allen colour (matching
        # ``use_ccf_color: true`` on the atlas-mesh-pack spec).
        from aind_low_point.ccf_ontology import CCFOntology

        ontology = CCFOntology.from_bundled()
        for s in structures:
            ccf_struct = ontology.find_by_acronym(s)
            material: dict = {}
            if ccf_struct is not None:
                material["color"] = ccf_struct.color_hex
            material["opacity"] = 0.15
            config.setdefault("assets", []).append(
                {
                    "key": f"structure:{s}",
                    "src": "${paths.ccf_annotation_path}",
                    "loader": "ccf_annotation_region",
                    "loader_kwargs": {"acronym": s},
                    "templates": ["structure"],
                    "transform": "headframe_to_lps",
                    "scene_tags": ["static", "structure"],
                    "metadata": {"ccf_acronym": s},
                    "material": material,
                }
            )
        # Two ``derive_from`` target groups — one per hemisphere — so
        # probes can reference ``target:L:STRUCTURE`` or
        # ``target:R:STRUCTURE`` and land on the appropriate side
        # rather than the bilateral midline centroid.
        for hemi_key, hemi_kw in (("L", "left"), ("R", "right")):
            config.setdefault("targets", []).append(
                {
                    "derive_from": [f"structure:{s}" for s in structures],
                    "key_prefix": f"target:{hemi_key}:",
                    "reducer": "hemisphere_center_mass",
                    "reducer_kwargs": {"hemisphere": hemi_kw},
                    "templates": ["structure"],
                    "transform": "headframe_to_lps",
                    "scene_tags": ["static", "target", "brain"],
                }
            )

    # Replace plan section.
    config.setdefault("plan", {})
    config["plan"]["arcs"] = arcs
    config["plan"]["probes"] = probes

    body = yaml.safe_dump(config, sort_keys=False, default_flow_style=False, width=120)
    header = (
        f"# Auto-generated from {base_config_path.name} + "
        f"{csv_path.name} for mouse {mouse}.\n"
        f"# Re-run `scripts/convert_old_plan.py` to regenerate.\n"
    )
    plan_notes = header_comments(
        csv_path, arcs, arc_id_to_letter, structures, hole_notes
    )
    output.write_text(header + plan_notes + body)
    print(f"Wrote full config to {output}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_path", type=Path, help="Old-style insertion-plan CSV")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output YAML (default: stdout for fragment mode, required for --base mode)"
        ),
    )
    p.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Base config to splice the plan into (e.g. examples/786864-config.yml). "
        "Without --base, only the plan fragment is emitted.",
    )
    p.add_argument(
        "--mouse",
        type=str,
        default=None,
        help="Override mouse id (default: inferred from CSV filename's leading digits)",
    )
    p.add_argument(
        "--targets",
        choices=["auto", "inline", "node"],
        default="auto",
        help=(
            "How each probe's *initial* target is encoded. "
            "``auto`` (default) → ``inline`` (uses CSV's ``ideal_pt_RAS``"
            " literally — preserves CSV positions exactly). ``node`` "
            "routes through ``target:HEMI:STRUCTURE`` plus an offset "
            "(=ideal − target_pt) which silently introduces a bias "
            "whenever the converter's hemisphere-centroid disagrees "
            "with the CSV producer's. CCF region assets and "
            "hemisphere-specific target nodes are emitted *regardless* "
            "(when the warped annotation volume exists), so inline-mode "
            "probes can still be retargeted to a structure from the UI."
        ),
    )
    args = p.parse_args()

    if args.base is None:
        text = emit_plan_only(
            args.csv_path,
            output=args.output,
            target_mode=args.targets,
        )
        if args.output is None:
            sys.stdout.write(text)
    else:
        if args.output is None:
            raise SystemExit("--base mode requires --output PATH")
        emit_full_config(
            args.csv_path,
            args.base,
            args.output,
            mouse=args.mouse,
            target_mode=args.targets,
        )


if __name__ == "__main__":
    main()
