"""The Plans tab: step through a solved plan set within one session.

Every plan in a set shares the config the planner was opened with, so moving
between them is a plan apply, not a scene rebuild. The table sorts and filters
in the browser; the server mirrors that order so the next/previous keys walk
the rows as they are shown.
"""

from __future__ import annotations

from typing import Any

from trame.widgets import vuetify3

from aind_rutter.plan_io.plan_set import PlanSet

HEADERS: list[dict[str, Any]] = [
    {"title": "Plan", "key": "rank"},
    {"title": "Cov", "key": "coverage"},
    {"title": "FCL mm", "key": "fcl_mm"},
    {"title": "Arcs", "key": "n_arcs"},
    {"title": "Bores", "key": "holes", "sortable": False},
]


def plan_rows(plan_set: PlanSet) -> list[dict]:
    return [
        {
            "file": e.file,
            "rank": e.rank,
            "coverage": None if e.coverage is None else round(e.coverage, 3),
            "fcl_mm": None if e.fcl_mm is None else round(e.fcl_mm, 3),
            "n_arcs": e.n_arcs,
            "holes": e.holes_str(plan_set.probes),
        }
        for e in plan_set.entries
    ]


def ordered_files(rows: list[dict], sort_by: list[dict], search: str) -> list[str]:
    """The files in the order the table shows them.

    Mirrors the table's default behaviour: a row survives the search when any
    shown column contains it, case-insensitively, and sort keys apply in
    priority order. Blank values sort last either way.
    """
    keys = [h["key"] for h in HEADERS]
    needle = (search or "").strip().lower()
    shown = [
        r
        for r in rows
        if not needle
        or any(needle in str(r[k]).lower() for k in keys if r[k] is not None)
    ]
    for spec in reversed(sort_by or []):
        key, desc = spec["key"], spec.get("order") == "desc"
        present = [r for r in shown if r[key] is not None]
        blank = [r for r in shown if r[key] is None]
        shown = sorted(present, key=lambda r: r[key], reverse=desc) + blank
    return [r["file"] for r in shown]


class PlanSetMixin:
    """See :class:`~aind_rutter.web.controller.TrameController`."""

    def _init_plan_set_state(self, state) -> None:
        ps = self.plan_set
        state.plan_set_rows = plan_rows(ps) if ps else []
        state.plan_set_headers = HEADERS
        state.plan_set_sort = [{"key": "coverage", "order": "desc"}]
        state.plan_set_search = ""
        state.plan_set_pick = None
        state.plan_set_current = None
        state.plan_set_status = ""
        if ps is None:
            return
        state.plan_set_status = f"{len(ps.entries)} plans in {ps.plans_dir}" + (
            "" if ps.indexed else " (no manifest.json: rank and coverage only)"
        )

    def _wire_plan_set_handlers(self, state) -> None:
        if self.plan_set is None:
            return

        @state.change("plan_set_pick")
        def _on_pick(**_):
            if state.plan_set_pick:
                self._apply_plan_set_file(state, state.plan_set_pick)
                state.plan_set_pick = None

    def _apply_plan_set_file(self, state, file: str) -> None:
        ps = self.plan_set
        entry = next(e for e in ps.entries if e.file == file)
        try:
            self._apply_plan_yaml_bytes(ps.path(entry).read_bytes())
        except Exception as exc:
            state.plan_set_status = f"⚠ {file}: {exc}"
            return
        self._sync_arc_list(state)
        self._load_probe_state(state)
        order = ordered_files(
            state.plan_set_rows, state.plan_set_sort, state.plan_set_search
        )
        where = f"{order.index(file) + 1} of {len(order)}" if file in order else ""
        parts = [f"plan-{entry.rank:02d}"]
        if entry.coverage is not None:
            parts.append(f"cov {entry.coverage:.3f}")
        if entry.fcl_mm is not None:
            parts.append(f"FCL {entry.fcl_mm:+.3f} mm")
        if entry.n_arcs is not None:
            parts.append(f"{entry.n_arcs} arcs")
        state.plan_set_current = file
        state.plan_set_status = " · ".join(parts) + (f"  ({where})" if where else "")

    def _step_plan_set(self, state, step: int) -> None:
        if self.plan_set is None:
            return
        order = ordered_files(
            state.plan_set_rows, state.plan_set_sort, state.plan_set_search
        )
        if not order:
            return
        cur = state.plan_set_current
        i = order.index(cur) + step if cur in order else (0 if step > 0 else -1)
        self._apply_plan_set_file(state, order[i % len(order)])

    def _build_plan_set_tab(self) -> None:
        vuetify3.VLabel("{{ plan_set_status }}", classes="text-caption mb-1")
        with vuetify3.VRow(dense=True, no_gutters=True, classes="mb-1"):
            with vuetify3.VCol():
                vuetify3.VBtn(
                    "Previous (p)",
                    block=True,
                    density="compact",
                    variant="outlined",
                    click=lambda: self._step_plan_set(self._readout_state, -1),
                )
            with vuetify3.VCol():
                vuetify3.VBtn(
                    "Next (n)",
                    block=True,
                    density="compact",
                    variant="outlined",
                    click=lambda: self._step_plan_set(self._readout_state, 1),
                )
        vuetify3.VTextField(
            v_model=("plan_set_search",),
            label="Filter (e.g. a bore: MD 12)",
            density="compact",
            hide_details=True,
            clearable=True,
            classes="mb-1",
        )
        vuetify3.VDataTable(
            headers=("plan_set_headers",),
            items=("plan_set_rows",),
            search=("plan_set_search",),
            sort_by=("plan_set_sort",),
            item_value="file",
            density="compact",
            fixed_header=True,
            height="60vh",
            items_per_page=-1,
            hide_default_footer=True,
            hover=True,
            update_sortBy="plan_set_sort = $event",
            # trame's VDataTable does not declare click:row; add it here.
            **{"__events": [("click_row", "click:row")]},
            click_row="(e, row) => { plan_set_pick = row.item.file }",
            row_props=(
                "(row) => row.item.file === plan_set_current"
                " ? { class: 'bg-primary' } : {}",
            ),
        )
