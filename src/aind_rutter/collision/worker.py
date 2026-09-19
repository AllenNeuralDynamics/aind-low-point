"""Running collision checks off the UI thread, and reporting what changed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set, Tuple

import fcl

from aind_rutter.collision.adapter import (
    CollisionAdapter,
    CollisionPair,
    CollisionState,
)
from aind_rutter.collision.geometry import rt_to_fcl_transform
from aind_rutter.domain.plan import PlanningState
from aind_rutter.domain.scene import Scene


def objects_in_collision(collision_pairs: List[CollisionPair]) -> List[Tuple[str, str]]:
    objects_in_collision = set()
    for coll_pair in collision_pairs:
        objects_in_collision.add((coll_pair.id1, coll_pair.id2))
    return list(objects_in_collision)


def _diff_hot(curr: CollisionState, prev: CollisionState | None = None) -> set[str]:
    # nodes that flipped collision state
    if prev is None:
        return set(curr.hot)
    return set((prev.hot - curr.hot) | (curr.hot - prev.hot))


@dataclass
class CollisionHandler:
    scene: Scene
    adapter: CollisionAdapter
    state: CollisionState = field(default_factory=CollisionState)
    on_state_changed: Optional[
        Callable[[CollisionState, Set[str], PlanningState], None]
    ] = None
    _prev_state: CollisionState | None = None

    def __call__(self, plan: PlanningState, changed_ids: List[str]) -> None:
        # keep backend up-to-date for moved probes (transform-only, no BVH rebuild)
        moved = [pid for pid in changed_ids if f"probe:{pid}" in self.scene.nodes]
        if moved:
            self.adapter.update_probe_transforms(plan, moved)

        # recompute collisions
        pairs = self.adapter.collide_internal(enable_contacts=False)
        new_pairs = {(p.id1, p.id2) for p in pairs}
        new_state = self.state.replace(new_pairs)

        # notify only if something flipped
        flips = _diff_hot(new_state, self._prev_state)
        self._prev_state = new_state
        self.state = new_state
        if flips and self.on_state_changed:
            self.on_state_changed(new_state, flips, plan)

    # --- factored methods for async worker ---

    def prepare(
        self, plan: PlanningState, changed_ids: List[str]
    ) -> List[Tuple[str, fcl.Transform]]:
        """Main thread: compute (node_id, fcl.Transform) pairs."""
        resolver = self.adapter._make_resolver(plan)
        transforms: List[Tuple[str, fcl.Transform]] = []
        for pid in changed_ids:
            nid = f"probe:{pid}"
            node = self.scene.nodes.get(nid)
            if node and self.adapter.include(node, self.adapter.assets):
                R, t = resolver.world_rt_for_node(node)
                tf = rt_to_fcl_transform(R, t, name=f"pose:{nid}")
                transforms.append((nid, tf))
        return transforms

    def work(
        self, transforms: List[Tuple[str, fcl.Transform]]
    ) -> Tuple[CollisionState, Set[str]]:
        """Worker thread: update transforms + run collision detection."""
        if transforms:
            self.adapter.backend.update_transforms(transforms)
        pairs = self.adapter.collide_internal(
            enable_contacts=False,
        )
        new_pairs = {(p.id1, p.id2) for p in pairs}
        new_state = self.state.replace(new_pairs)
        flips = _diff_hot(new_state, self._prev_state)
        self._prev_state = new_state
        self.state = new_state
        return (new_state, flips)

    def deliver(
        self,
        result: Tuple[CollisionState, Set[str]],
        plan: PlanningState,
    ) -> None:
        """Main thread: update overlays if collision state changed."""
        new_state, flips = result
        if flips and self.on_state_changed:
            self.on_state_changed(new_state, flips, plan)
