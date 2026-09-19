"""Per-node appearance: the warning overlays and the default opacities.

Overlays are layered by priority in ``OverlayResolver``, so a probe that
is both colliding and over-inserted shows the collision colour.
"""

from __future__ import annotations

from aind_mri_utils.arc_angles import arc_angles_to_affine

from aind_rutter.ccf.ontology import CCFOntology
from aind_rutter.domain.catalog import AssetCatalog
from aind_rutter.domain.plan import kinematic_violations
from aind_rutter.domain.pose import ProbePose
from aind_rutter.render.overlays import OverlaySpec

# Overlay colour + priority for over-insertion warnings. Collisions are
# at priority 30, so they still win when both overlays apply to the same
# probe (a colliding probe will appear collision-red even if
# additionally over-inserted).
OVERINSERTION_OVERLAY = OverlaySpec(
    color=0xFF8800,  # orange
    alpha=0.7,
    source="overinsertion",
    priority=25,
)

# Kinematic-violation overlay: pairs of probes whose AP arcs are <16°
# apart, or pairs of probes on the same arc with ML <16° apart, can't
# physically be set up on the rig. Lower priority than over-insertion
# (the geometry is recoverable by retargeting; this isn't).
KINEMATIC_OVERLAY = OverlaySpec(
    color=0xFFD400,  # amber yellow
    alpha=0.7,
    source="kinematic",
    priority=20,
)

VISIBILITY_GROUPS: list[tuple[str, str, set[str], set[str]]] = [
    # (state-key, display label, tags-to-include, tags-to-exclude).
    # Groups are independent — a node ends up in every group whose
    # include set intersects its tags AND whose exclude set doesn't.
    # The exclude column is what keeps the implant slider distinct
    # from the rest of the fixtures (the implant has both "implant"
    # and "fixture" tags, so excluding "implant" from the broader
    # group is how we split it out without renaming any tags).
    #
    # Probes go first since they're what the user is positioning;
    # opacity edits persist via the node's material_override (the
    # renderer reads override before the asset's base material), so
    # subsequent probe pose updates don't reset the slider.
    ("probes", "Probes", {"probe"}, set()),
    ("brain", "Brain outline", {"brain"}, set()),
    ("structures", "CCF regions", {"structure"}, set()),
    ("implant", "Implant", {"implant"}, set()),
    ("fixtures", "Other fixtures", {"fixture", "headframe"}, {"implant"}),
]


def ccf_color_for_target(catalog: AssetCatalog, target_key: str | None) -> str | None:
    """Return the CCF color_hex for *target_key* if its source asset
    carries CCF metadata (set by AtlasMeshPackSpecModel.expand). Returns
    None for targets that aren't derived from a CCF region."""
    if target_key is None:
        return None
    target_spec = catalog.targets.get(target_key)
    if target_spec is None or target_spec.source_key is None:
        return None
    source_spec = catalog.assets.get(target_spec.source_key)
    if source_spec is None:
        return None
    acronym = source_spec.metadata.get("ccf_acronym")
    if not acronym:
        return None
    structure = CCFOntology.from_bundled().find_by_acronym(acronym)
    return structure.color_hex if structure else None


# Default opacity overrides per scene-tag. Implant is mostly transparent so
# probes threading through holes are visible; other fixtures (headframe, well,
# probe guard) sit at 40% transparency so they read as a frame of reference
# without obscuring the implant and brain.
_DEFAULT_OPACITY_BY_TAG: tuple[tuple[str, float, frozenset[str]], ...] = (
    ("implant", 0.2, frozenset()),
    ("fixture", 0.6, frozenset({"implant"})),
    ("headframe", 0.6, frozenset({"implant"})),
)


class MaterialsMixin:
    """See :class:`~aind_rutter.web.controller.TrameController`."""

    def _nodes_with_any_tag(
        self,
        include: set[str],
        exclude: set[str] = frozenset(),
    ):
        """Scene nodes whose tag set intersects *include* and is disjoint
        from *exclude* (used by visibility groups)."""
        return [
            n
            for n in self.render_adapter.scene.nodes.values()
            if (n.tags & include) and not (n.tags & exclude)
        ]

    def _apply_material_to_nodes(
        self,
        node_ids,
        *,
        opacity: float | None = None,
        visible: bool | None = None,
    ) -> None:
        """Mutate ``material_override`` on a set of nodes, preserving every
        field except the ones explicitly overridden. Issues a single
        ``repaint_materials`` call for the affected nodes."""
        affected: list[str] = []
        for nid in node_ids:
            node = self.render_adapter.scene.nodes.get(nid)
            if node is None:
                continue
            base = node.material_override
            if base is None:
                base = self.assets.get_spec(node.asset_key).default_material
            overrides = {}
            if opacity is not None:
                overrides["opacity"] = float(opacity)
            if visible is not None:
                overrides["visible"] = bool(visible)
            node.material_override = base.replace(**overrides)
            affected.append(nid)
        if affected:
            self.render_adapter.repaint_materials(affected)

    def _on_collision_state_changed(
        self,
        coll_state,
        flips,
        plan,
        chained_callback=None,
    ) -> None:
        """Wrapper installed over the collision handler's
        ``on_state_changed``. Runs the original overlay-repaint callback
        first, then refreshes our collision readouts. Kept distinct from
        ``_refresh_readouts`` because collision changes fire from a
        worker thread; we forward to the main loop via the readout
        state (which is the trame server state)."""
        if chained_callback is not None:
            chained_callback(coll_state, flips, plan)
        state = self._readout_state
        if state is None:
            return
        probe_name = state.probe
        coll_local, coll_scene = self._compute_collision_strs(probe_name or "")
        with state:
            state.probe_collision_str = coll_local
            state.scene_collision_str = coll_scene

    def _refresh_kinematic_overlay(self) -> None:
        """Recompute kinematic violations and update the shared overlay
        state. Repaints only probes whose membership flipped."""
        viols = kinematic_violations(self.store.state)
        affected: set[str] = set()
        for arc_a, arc_b in viols["arc_ap"]:
            for name, plan in self.store.state.probes.items():
                if plan.arc_id in (arc_a, arc_b):
                    affected.add(f"probe:{name}")
        for a, b in viols["within_arc_ml"]:
            affected.add(f"probe:{a}")
            affected.add(f"probe:{b}")
        flips = list(self._prev_kinematic ^ affected)
        self._prev_kinematic = affected
        overlays_state = self.overlays_resolver.overlays
        overlays_state.clear_source("kinematic")
        if affected:
            overlays_state.set_for_source(list(affected), KINEMATIC_OVERLAY)
        if flips:
            self.render_adapter.repaint_materials(flips)

    def _refresh_overinsertion_overlay(self, changed_ids) -> None:
        """For every probe in *changed_ids*, recompute over-insertion and
        update the shared overlay state. Repaints only the probes whose
        over-insertion status flipped."""
        brain_mesh = self._get_brain_world_mesh()
        if brain_mesh is None:
            return
        flips: list[str] = []
        for probe_name in changed_ids:
            if probe_name not in self.store.state.probes:
                continue
            pose = ProbePose.from_planning_state(
                self.store.state, probe_name, catalog=self.assets
            )
            R = arc_angles_to_affine(pose.ap, pose.ml, pose.spin)
            n_over, n_total = self._count_overinserted_shanks(
                probe_name, pose, R, brain_mesh
            )
            nid = f"probe:{probe_name}"
            was = nid in self._prev_overinserted
            is_now = n_over > 0 and n_total > 0
            if is_now and not was:
                self._prev_overinserted.add(nid)
                flips.append(nid)
            elif was and not is_now:
                self._prev_overinserted.discard(nid)
                flips.append(nid)
        if not flips:
            return
        overlays_state = self.overlays_resolver.overlays
        overlays_state.clear_source("overinsertion")
        if self._prev_overinserted:
            overlays_state.set_for_source(
                list(self._prev_overinserted), OVERINSERTION_OVERLAY
            )
        self.render_adapter.repaint_materials(flips)

    def _apply_target_based_color(self, probe_name: str) -> None:
        """Set ``probe:<name>`` node's material_override to the CCF color of
        its current target (no-op if the target isn't CCF-derived)."""
        plan = self.store.state.probes.get(probe_name)
        if plan is None or plan.target_key is None:
            return
        color = ccf_color_for_target(self.assets, plan.target_key)
        if color is None:
            return
        nid = f"probe:{probe_name}"
        node = self.render_adapter.scene.nodes.get(nid)
        if node is None:
            return
        # Inherit other material fields (opacity, point_size, …) from the
        # current override or the asset's default material.
        base = node.material_override
        if base is None:
            base = self.assets.get_spec(node.asset_key).default_material
        node.material_override = base.replace(color_hex_str=color)
        self.render_adapter.repaint_materials([nid])

    def apply_default_opacities(self) -> None:
        """Set the startup opacity of tagged fixtures so the implant is mostly
        transparent and the rest sit at 40%.

        Written to each node's ``material_override``, which is what the
        renderer resolves from, rather than to the actor: an actor property is
        restored by the next repaint, and a collision flip repaints.

        A node takes the first matching row of the table, so a node tagged both
        ``implant`` and ``fixture`` gets the implant value.
        """
        by_opacity: dict[float, list[str]] = {}
        for nid, node in self.render_adapter.scene.nodes.items():
            for tag, opacity, excluded in _DEFAULT_OPACITY_BY_TAG:
                if tag in node.tags and not (node.tags & excluded):
                    by_opacity.setdefault(float(opacity), []).append(nid)
                    break
        for opacity, node_ids in by_opacity.items():
            self._apply_material_to_nodes(node_ids, opacity=opacity)
