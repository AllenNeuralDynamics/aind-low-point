"""The root config model and its cross-reference validation."""

from __future__ import annotations

import difflib
import fnmatch
import warnings
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from aind_rutter.config.models_catalog import (
    AssetSpecModel,
    AssetSpecUnion,
    AssetTemplateModel,
    AtlasMeshPackSpecModel,
    BulkAssetSpecModel,
    DerivedTargetSpecModel,
    RangeTargetSpecModel,
    TargetSpecModel,
    TargetSpecUnion,
    TargetTemplateModel,
    infer_from_extension,
    infer_role_from_key,
)
from aind_rutter.config.models_common import (
    NO_RECORDING_ARRAY,
    RETIRED_FIELDS,
    UPGRADE_FROM_COMMIT,
    CanonicalizationDefModel,
    ImagingModel,
    MaterialModel,
    PathsModel,
    ResourceModel,
    TransformRecipeModel,
    TransformRefModel,
    find_matching_templates,
    retired_fields_in,
)
from aind_rutter.config.models_plan import PlanningModel
from aind_rutter.config.models_scene import SceneModel, SceneNodeModel
from aind_rutter.config.resolve import effective_canon_for_spec, has_transform
from aind_rutter.config.templates import (
    apply_templates_generic,
    merge_asset_template_model_dumps,
    merge_target_template_model_dumps,
)
from aind_rutter.domain.enums import KNOWN_SCENE_TAGS, Kind, MRSignal, Role
from aind_rutter.domain.plan import probe_asset_key

# -----------------------------------------------------------------------------
# Root config (everything together) + cross-reference validation
# -----------------------------------------------------------------------------


class ConfigModel(BaseModel):
    model_config = {"extra": "forbid"}

    version: int = 1

    paths: PathsModel = Field(default_factory=PathsModel)
    imaging: Optional[ImagingModel] = None
    resources: list[ResourceModel] = Field(default_factory=list)
    materials: dict[str, MaterialModel] = Field(default_factory=dict)

    # Catalog
    asset_templates: dict[str, AssetTemplateModel] = Field(default_factory=dict)
    target_templates: dict[str, TargetTemplateModel] = Field(default_factory=dict)
    assets: list[AssetSpecUnion] = Field(default_factory=list)
    targets: list[TargetSpecUnion] = Field(default_factory=list)

    # Scene
    scene: SceneModel = Field(default_factory=SceneModel)

    # Domain
    plan: PlanningModel = Field(default_factory=PlanningModel)

    # Named transforms & misc
    transforms: dict[str, TransformRecipeModel] = Field(default_factory=dict)
    canonicalizations: dict[str, CanonicalizationDefModel] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: "str | Path") -> "ConfigModel":
        """Load a ConfigModel from a YAML file with OmegaConf interpolation."""
        from omegaconf import OmegaConf

        raw = OmegaConf.load(path)
        resolved = OmegaConf.to_container(raw, resolve=True)
        retired = retired_fields_in(resolved)
        if retired:
            says = "; ".join(f"`{k}` → {RETIRED_FIELDS[k]}" for k in sorted(retired))
            raise ValueError(
                f"{path}: written against a retired schema ({says}). Migrate it "
                f"with `scripts/upgrade_config.py` as of commit "
                f"{UPGRADE_FROM_COMMIT}, which is the last one that can read "
                f"these fields."
            )
        return cls.model_validate(resolved)

    # ---------- Cross-file integrity checks ----------
    @model_validator(mode="after")
    def _xref_and_expand_templates(self):  # noqa: C901
        errors: list[str] = []

        # ---------- Expand bulk specs first ----------
        expanded_assets: list[AssetSpecModel] = []
        for item in self.assets:
            if isinstance(item, (BulkAssetSpecModel, AtlasMeshPackSpecModel)):
                expanded_assets.extend(item.expand())
            else:
                expanded_assets.append(item)
        self.assets = expanded_assets

        # Resolve any string-form ``derive_from`` (fnmatch glob) entries
        # against the now-expanded asset list before we run target
        # expansion, so a single AtlasMeshPack can drive both the asset
        # roster and the derived targets without duplicating the
        # acronym list.
        asset_keys = [a.key for a in self.assets]
        for item in self.targets:
            if not isinstance(item, DerivedTargetSpecModel):
                continue
            if not isinstance(item.derive_from, str):
                continue
            pattern = item.derive_from
            matched = sorted(k for k in asset_keys if fnmatch.fnmatch(k, pattern))
            if not matched:
                errors.append(
                    f"derive_target glob '{pattern}' matched no assets after expansion"
                )
                item.derive_from = []
                continue
            item.derive_from = matched

        asset_by_key = {a.key: a for a in self.assets}
        expanded_targets: list[TargetSpecModel] = []
        for item in self.targets:
            if isinstance(item, DerivedTargetSpecModel):
                if not item.derive_from:
                    continue  # glob match was empty; already reported
                expanded_targets.extend(item.expand(asset_by_key))
            elif isinstance(item, RangeTargetSpecModel):
                expanded_targets.extend(item.expand())
            else:
                expanded_targets.append(item)
        self.targets = expanded_targets

        # ---------- Auto-match templates by key glob ----------
        for asset in self.assets:
            if not asset.templates:  # no explicit templates
                auto = find_matching_templates(asset.key, self.asset_templates)
                if auto:
                    object.__setattr__(asset, "templates", auto)

        for target in self.targets:
            if not target.templates:
                auto = find_matching_templates(target.key, self.target_templates)
                if auto:
                    object.__setattr__(target, "templates", auto)

        # ---------- sets for quick membership ----------
        asset_keys = {a.key for a in self.assets}
        target_keys = {t.key for t in self.targets}
        node_keys = {n.key for n in self.scene.nodes}
        transform_keys = set(self.transforms.keys())
        arc_keys = set(self.plan.arcs.keys())
        probe_names = set(self.plan.probes.keys())
        reticle_names = set(self.plan.reticles.keys())
        cal_files = self.plan.calibrations.files  # dict[id -> CalFileDecl]

        node_idx_by_key = {k: i for i, k in enumerate(node_keys)}

        # ---------- helpers ----------
        def err(msg: str) -> None:
            errors.append(msg)

        def _where_key(obj) -> str:
            return getattr(obj, "key", "?")

        def _check_template_ref(spec, templates, where_prefix: str):
            trefs = getattr(spec, "templates", [])
            for tref in trefs:
                if tref not in templates:
                    err(
                        f"{where_prefix} '{_where_key(spec)}': "
                        f"template '{tref}' not found"
                    )

        for a in self.assets:
            _check_template_ref(a, self.asset_templates, "asset")
        for t in self.targets:
            _check_template_ref(t, self.target_templates, "target")
        # Expand templates into concrete specs
        if self.asset_templates and self.assets:
            assets = []
            for a in self.assets:
                assets.append(
                    apply_templates_generic(
                        merge_asset_template_model_dumps,
                        a,
                        a.templates,
                        self.asset_templates,
                    )
                )
            self.assets = assets
        if self.target_templates and self.targets:
            targets = []
            for t in self.targets:
                targets.append(
                    apply_templates_generic(
                        merge_target_template_model_dumps,
                        t,
                        t.templates,
                        self.target_templates,
                    )
                )
            self.targets = targets

        # ---------- Infer kind/loader from extension and role from key ----------
        for asset in self.assets:
            infer_from_extension(asset)
            infer_role_from_key(asset)
        for target in self.targets:
            infer_from_extension(target)
            infer_role_from_key(target)

        # ---------- Auto-generate scene nodes ----------
        generated_nodes: list[SceneNodeModel] = []
        explicit_keys = {n.key for n in self.scene.nodes}
        self.scene._explicit_node_keys = set(explicit_keys)

        # From assets with transform or scene_tags
        for asset in self.assets:
            if not asset.auto_scene:
                continue
            if asset.key in explicit_keys:
                continue  # explicit node takes precedence
            if asset.transform or asset.scene_tags or asset.tags:
                generated_nodes.append(
                    SceneNodeModel(
                        key=asset.key,
                        asset=asset.key,
                        transform=asset.transform,
                        # `tags` describes the thing, `scene_tags` the placement;
                        # the node carries both, so a label on an asset is
                        # something a filter can actually find.
                        tags=sorted({*asset.scene_tags, *asset.tags}),
                    )
                )

        # From targets with transform or scene_tags
        for target in self.targets:
            if not target.auto_scene:
                continue
            if target.key in explicit_keys:
                continue
            if target.transform or target.scene_tags or target.tags:
                generated_nodes.append(
                    SceneNodeModel(
                        key=target.key,
                        asset=target.key,
                        transform=target.transform,
                        # `tags` describes the thing, `scene_tags` the placement;
                        # the node carries both, so a label on an asset is
                        # something a filter can actually find.
                        tags=sorted({*target.scene_tags, *target.tags}),
                    )
                )

        # From plan.probes (auto-generate probe:{name} nodes)
        for probe_name, probe_decl in self.plan.probes.items():
            node_key = f"probe:{probe_name}"
            if not probe_decl.auto_scene:
                continue
            if node_key in explicit_keys:
                continue
            asset_key = f"probe:{probe_decl.kind}"
            generated_nodes.append(
                SceneNodeModel(
                    key=node_key,
                    asset=asset_key,
                    tags=probe_decl.scene_tags,
                    pose_source_probe=probe_name,
                )
            )

        # Prepend generated nodes (explicit nodes are already in self.scene.nodes)
        if generated_nodes:
            self.scene.nodes = generated_nodes + list(self.scene.nodes)

        # Update node_keys after generation
        node_keys = {n.key for n in self.scene.nodes}
        node_idx_by_key = {n.key: i for i, n in enumerate(self.scene.nodes)}

        def _check_spec_kind(spec, where_prefix: str, allowable=None):
            kind = getattr(spec, "kind", None)
            if not kind:
                err(f"{where_prefix} '{_where_key(spec)}': kind not set")
            if allowable and kind not in allowable:
                err(f"{where_prefix} '{_where_key(spec)}': kind '{kind}' not allowed")

        def _check_spec_role(spec, where_prefix: str, allowable=None):
            role = getattr(spec, "role", None)
            if not role:
                err(f"{where_prefix} '{_where_key(spec)}': role not set")
            if allowable and role not in allowable:
                err(f"{where_prefix} '{_where_key(spec)}': role '{role}' not allowed")

        def _check_asset_spec_src_loader(spec: AssetSpecModel):
            if (spec.src is None) ^ (spec.loader is None):
                err(
                    f"Asset '{_where_key(spec)}' must provide "
                    "both 'src' and 'loader', or neither (if injected elsewhere)."
                )
            if (spec.src and spec.loader) and (spec.from_resource or spec.selector):
                err(
                    f"Asset '{_where_key(spec)}': Choose either "
                    "(src+loader) or (from_resource+selector), not both."
                )
            if (spec.from_resource is None) ^ (spec.selector is None):
                # only one given
                err(
                    f"Asset '{_where_key(spec)}': When using "
                    "from_resource, you must also provide a selector."
                )

        def _check_target_spec_single_source(spec: TargetSpecModel):
            explicit = spec.src is not None and spec.loader is not None
            derived = spec.source_key is not None
            from_res = (spec.from_resource is not None) and (spec.selector is not None)
            paths = sum([explicit, derived, from_res])
            if paths != 1:
                err(
                    f"Target '{_where_key(spec)}': provide exactly one of "
                    "(src+loader) | (source_key+reducer) | (from_resource+selector)"
                )
            if spec.collidable:
                err(f"Target '{_where_key(spec)}': targets should not be collidable.")

        def _check_material_ref(spec, where_prefix: str):
            mref = getattr(spec, "material_ref", None)
            if mref and mref not in self.materials:
                err(
                    f"{where_prefix} '{_where_key(spec)}': "
                    f"material_ref '{mref}' not found"
                )

        def _check_probe_recording():
            """Every probe kind a plan uses must resolve to a recording array.

            Unset falls back to the built-in table; a kind that is in neither
            the table nor the config is a typo, and used to resolve silently to
            tip-on-target — the same answer a pipette gets deliberately.
            """
            from aind_rutter.domain.probe_kinds import RECORDING_GEOMETRY

            by_key = {str(a.key): a for a in self.assets}
            for probe_name, decl in self.plan.probes.items():
                spec = by_key.get(probe_asset_key(decl.kind))
                if spec is None:
                    continue  # the missing-asset check reports this already
                if spec.recording is not None:
                    continue  # explicit geometry, or explicitly array-less
                if decl.kind not in RECORDING_GEOMETRY:
                    err(
                        f"plan.probes['{probe_name}']: probe kind "
                        f"'{decl.kind}' has no recording geometry — give the "
                        f"asset a `recording:` block, or "
                        f"`recording: {NO_RECORDING_ARRAY}` if it has no array"
                    )

        def _check_scene_tags():
            """Nudge on a tag that is a near-miss for one the code acts on.

            Silent otherwise: an unfamiliar tag is how a subject groups its own
            nodes, and warning on those would train the reader to ignore this.
            """
            for spec in [*self.assets, *self.targets]:
                for tag in {*spec.scene_tags, *spec.tags}:
                    if tag in KNOWN_SCENE_TAGS:
                        continue
                    close = difflib.get_close_matches(
                        tag, sorted(KNOWN_SCENE_TAGS), n=1, cutoff=0.8
                    )
                    if close:
                        warnings.warn(
                            f"{_where_key(spec)}: tag '{tag}' is not one the code "
                            f"acts on — did you mean '{close[0]}'?",
                            stacklevel=2,
                        )

        _check_scene_tags()
        _check_probe_recording()

        def _check_mr_signal(spec, where_prefix: str):
            # Chemical shift displaces geometry by millimetres and nothing
            # downstream notices, so a config that has an image must say, per
            # feature, which resonance localized it. A config without an
            # `imaging` block — an atlas plan — has no image and says nothing.
            if self.imaging is None or spec.mr_signal is not None:
                return
            err(
                f"{where_prefix} '{_where_key(spec)}': mr_signal is required "
                f"when `imaging` is set — one of "
                f"{', '.join(m.value for m in MRSignal)}. See dev/VOCABULARY.md."
            )

        for a in self.assets:
            _check_material_ref(a, "asset")
            _check_spec_kind(a, "asset")
            _check_spec_role(a, "asset")
            _check_asset_spec_src_loader(a)
            _check_mr_signal(a, "asset")
        for t in self.targets:
            _check_material_ref(t, "target")
            _check_spec_kind(t, "target", allowable={Kind.POINTS})
            _check_spec_role(t, "target", allowable={Role.TARGET})
            _check_target_spec_single_source(t)
            _check_mr_signal(t, "target")

        for name, tmpl in self.asset_templates.items():
            if tmpl.material_ref and tmpl.material_ref not in self.materials:
                err(
                    f"asset_templates['{name}']: "
                    f"material_ref '{tmpl.material_ref}' not found"
                )
        for name, tmpl in self.target_templates.items():
            if tmpl.material_ref and tmpl.material_ref not in self.materials:
                err(
                    f"target_templates['{name}']: "
                    f"material_ref '{tmpl.material_ref}' not found"
                )

        def _check_transform_ref(
            ref: Optional["TransformRefModel"], where: str
        ) -> None:
            # Only key references need validation; inline is self-contained.
            if ref and ref.key and ref.key not in transform_keys:
                err(f"{where}: transform key '{ref.key}' not found in transforms")

        def _check_canon_def(
            cdef: Optional["CanonicalizationDefModel"], where: str
        ) -> None:
            if not cdef:
                return
            _check_transform_ref(cdef.transform, f"{where}.transform")

        def _check_canon_fields(spec, where_prefix: str) -> None:
            # canonicalization_ref must exist; if it does, validate its transform ref
            cref = getattr(spec, "canonicalization_ref", None)
            if cref:
                cdef = self.canonicalizations.get(cref)
                if cdef is None:
                    err(
                        f"{where_prefix} '{_where_key(spec)}': "
                        f"canonicalization_ref '{cref}' not found"
                    )
                else:
                    _check_canon_def(cdef, f"canonicalizations['{cref}']")

            # Inline and override canonicalizations may each carry a transform ref
            _check_canon_def(
                getattr(spec, "canonicalization", None),
                f"{where_prefix} '{_where_key(spec)}'.canonicalization",
            )
            _check_canon_def(
                getattr(spec, "canonicalization_override", None),
                f"{where_prefix} '{_where_key(spec)}'.canonicalization_override",
            )

        # ---------- scene checks ----------
        catalog_keys = asset_keys | target_keys
        dups = asset_keys & target_keys
        if dups:
            err(f"Catalog has duplicate keys: {dups}")
        for n in self.scene.nodes:
            if n.asset not in catalog_keys:
                err(f"scene.nodes['{n.key}']: asset '{n.asset}' not found in catalog")
            _check_transform_ref(
                getattr(n, "transform", None), f"scene.nodes['{n.key}'].transform"
            )
            if (
                getattr(n, "pose_source_probe", None)
                and n.pose_source_probe not in probe_names
            ):
                err(
                    f"scene.nodes['{n.key}'].pose_source_probe "
                    f"'{n.pose_source_probe}' not in plan.probes"
                )

        # ---------- targets ----------
        for t in self.targets:
            # Derived targets must reference existing assets
            if getattr(t, "source_key", None) and t.source_key not in asset_keys:
                err(
                    f"target '{t.key}': source_key '{t.source_key}' not found in assets"
                )
            _check_canon_fields(t, "target")

        # ---------- assets / resources ----------
        for a in self.assets:
            _check_canon_fields(a, "asset")
        for r in self.resources:
            _check_canon_fields(r, "resource")

        # ---------- plan (arcs, probes, calibrations) ----------
        seen_catalog_target_names = set()
        seen_node_target_names = set()
        for pname, p in self.plan.probes.items():
            if p.arc is not None and p.arc not in arc_keys:
                err(f"plan.probes['{pname}']: arc '{p.arc}' not found in plan.arcs")
            if p.bind_ap_to_arc and p.arc is None:
                err(f"plan.probes['{pname}']: bind_ap_to_arc=True but arc is not set")
            if p.target.kind == "inline":
                pass  # self-contained RAS point, no xref needed
            elif p.target.kind == "catalog":
                target_key = p.target.key
                if target_key not in target_keys:
                    err(
                        f"plan.probes['{pname}']: catalog target "
                        f"'{target_key}' not found"
                    )
                seen_catalog_target_names.add(target_key)
            else:  # node
                target_key = p.target.key
                if target_key not in node_keys:
                    err(
                        f"plan.probes['{pname}']: node target "
                        f"'{target_key}' not in scene.nodes"
                    )
                node_idx = node_idx_by_key.get(target_key)
                target_ref = self.scene.nodes[node_idx].asset
                if target_ref in asset_keys:
                    err(
                        f"plan.probes['{pname}']: node target '{target_key}' "
                        f"references asset '{target_ref}' instead of target"
                    )
                seen_node_target_names.add(target_key)
        conflicting_names = seen_catalog_target_names.intersection(
            seen_node_target_names
        )
        if conflicting_names:
            err(
                f"plan.probes: targets '{conflicting_names}' ambiguous "
                "(in both catalog and node targets)"
            )
        # A parallax directory needs its reticle declared. A single calibration
        # file carries its own, and CalibrationSourceModel forbids naming one —
        # so requiring a reticle of every source made `file:` unusable. Both
        # wiring modes are checked; only `sources` went unchecked before.
        calibration_sources = [
            (f"files['{cal_id}']", cal) for cal_id, cal in cal_files.items()
        ] + [
            (f"sources[{i}]", src)
            for i, src in enumerate(self.plan.calibrations.sources)
        ]
        for label, source in calibration_sources:
            if source.directory is None:
                continue
            if source.reticle not in reticle_names:
                err(
                    f"plan.calibrations.{label}: "
                    f"reticle '{source.reticle}' not in plan.reticles"
                )

        # probe_to_ref must point to a valid probe and cal file id
        for probe_name, ref in self.plan.calibrations.probe_to_ref.items():
            if probe_name not in probe_names:
                err(
                    f"plan.calibrations.probe_to_ref: "
                    f"probe '{probe_name}' not in plan.probes"
                )
            if ref.cal_id not in cal_files:
                err(
                    f"plan.calibrations.probe_to_ref['{probe_name}']: "
                    f"cal_id '{ref.cal_id}' not in files"
                )

        # ---------- canonicalization defs themselves ----------
        for cname, cdef in self.canonicalizations.items():
            _check_transform_ref(
                cdef.transform, f"canonicalizations['{cname}'].transform"
            )
            # Require a transform (key OR inline) when source_space is FILE_NATIVE
            if cdef.source_space == "FILE_NATIVE" and cdef.transform is None:
                errors.append(
                    f"canonicalizations['{cname}']: FILE_NATIVE requires "
                    "a transform (key or inline)"
                )

        for seq, kind in (
            (self.assets, "asset"),
            (self.targets, "target"),
            (self.resources, "resource"),
        ):
            for item in seq:
                eff = effective_canon_for_spec(item, self.canonicalizations)
                if eff is None:
                    continue

                # ----- XOR rule -----
                named_space = eff.source_space != "FILE_NATIVE"
                has_tx = has_transform(eff.transform)

                # Exactly one must be true:
                if named_space == has_tx:  # both True or both False
                    # Build a concise, actionable message:
                    key = getattr(item, "key", "?")
                    if named_space and has_tx:
                        err(
                            f"{kind} '{key}': use source_space "
                            f"({eff.source_space}) OR transform, not both"
                        )
                    else:
                        err(
                            f"{kind} '{key}': provide source_space "
                            "(RAS/LPS/…) OR transform (for FILE_NATIVE)"
                        )

        # ---------- final ----------
        if errors:
            raise ValueError(
                "Config cross-reference errors:\n  - " + "\n  - ".join(errors)
            )
        return self

    def to_explicit_dict(self) -> dict[str, Any]:
        """Export expanded config as a dict suitable for YAML serialization.

        The output contains no bulk specs (they've been expanded), templates
        have been applied, and auto-generated scene nodes are included.
        The result can be re-loaded as a valid ConfigModel.

        Returns
        -------
        dict
            JSON-serializable dict (Path→str, Enum→value).
        """
        return self.model_dump(mode="json", exclude_defaults=False)


def expand_config(config_data: dict[str, Any]) -> dict[str, Any]:
    """Parse config, expand all bulk specs and auto-generation, return explicit dict.

    This is a convenience function for the common pattern of loading a concise
    config and exporting it to an explicit form for inspection or archival.

    Parameters
    ----------
    config_data : dict
        Raw config data (e.g., from YAML).

    Returns
    -------
    dict
        Fully expanded, explicit config that can be re-loaded.
    """
    config = ConfigModel.model_validate(config_data)
    return config.to_explicit_dict()
