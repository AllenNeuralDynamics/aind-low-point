"""Vuetify layout for the planner: tabs, sliders, dialogs.

Split out of the controller because it is the half that answers "what
does the page look like"; the controller answers "what happens when it
is used". Nothing here reads geometry — every value it shows is already
in the trame state, put there by the readouts.
"""

from __future__ import annotations

from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import client, vuetify3
from trame_pyvista.ui import plotter_ui

from aind_rutter.domain.commands import (
    SetProbeOffsetsRA,
    SetProbePastTarget,
    SetProbeTarget,
)
from aind_rutter.web.keybindings import KB_ACTIONS, KB_KEYS_TO_INTERCEPT
from aind_rutter.web.materials import VISIBILITY_GROUPS


class LayoutMixin:
    """See :class:`~aind_rutter.web.controller.TrameController`."""

    def _build_layout(self, server, ctrl) -> None:
        """Build the Vuetify3 + PyVista layout."""

        def on_set_target():
            state = server.state
            if not state.probe or not state.target:
                return
            self.store.dispatch(
                SetProbeTarget(name=state.probe, target_key=state.target)
            )
            # Reset depth/offsets to defaults so the kinematic chain's
            # auto-centering takes over (recording-array center lands
            # on the target). The user can dial deviations afterwards.
            self.store.dispatch(
                SetProbePastTarget(name=state.probe, past_target_mm=0.0)
            )
            self.store.dispatch(SetProbeOffsetsRA(name=state.probe, R_mm=0.0, A_mm=0.0))
            state.depth = 0.0
            state.offset_r = 0.0
            state.offset_a = 0.0
            self._apply_target_based_color(state.probe)

        with SinglePageLayout(server) as layout:
            layout.title.set_text("Probe Planner")

            # Install a window-level keydown listener once on mount.
            # We can't bind to SinglePageLayout's root with v-on:keydown
            # (focus has to be on a focusable element for that to fire),
            # so we attach to the document. The listener skips events
            # whose target is an INPUT/TEXTAREA so the user can still
            # type into search fields. Each captured keypress writes a
            # fresh dict to the ``kb_event`` server state via
            # ``trame.state.set(...)``; the ``state.change("kb_event")``
            # handler then dispatches the action. NB: this string lands
            # inside a double-quoted HTML attribute, so every literal
            # inside the JS must use single quotes — embedded ``"``
            # would terminate the attribute and break Vue templating.
            # NB: Vue 3's template compiler does not expose ``document``
            # as a global to v-on inline handlers (it resolves to
            # ``undefined``), but ``window`` is reachable via its proxy
            # context. Route DOM access through ``window.document``.
            #
            # Sentinel name is bumped (``_aind_kb_listener_v3``) so that
            # any stale listener from a previous attempt in the same
            # browser session does not block re-installation. Wrapped
            # in try/catch so the next failure is loud in the browser
            # console rather than silently leaving the keyboard inert.
            #
            # ``stopImmediatePropagation`` (in addition to
            # ``preventDefault``) is required because VTK.js's render
            # window interactor registers its own keydown handler that
            # binds e.g. 'r' to ResetCamera. Without stopping
            # propagation, our handler absorbs the application
            # action and VTK still fires its own — so 'r' both deepens
            # the probe (intended) and resets the camera (a bug). Our
            # listener runs in the capture phase on ``document``, so
            # stopping propagation here halts the event before it
            # reaches descendant elements like the render canvas.
            kb_keys_js = ",".join(f"'{k}'" for k in KB_KEYS_TO_INTERCEPT)
            client.ClientTriggers(
                mounted=(
                    "(function(){"
                    "if (window._aind_kb_listener_v3) return;"
                    "try {"
                    "  const trapped = new Set([" + kb_keys_js + "]);"
                    "  window.document.addEventListener('keydown', function(e){"
                    "    const tag = e.target && e.target.tagName;"
                    "    if (tag === 'INPUT' || tag === 'TEXTAREA' || "
                    "(e.target && e.target.isContentEditable)) return;"
                    "    if (!trapped.has(e.key) && "
                    "!trapped.has(e.key.toLowerCase())) return;"
                    "    trame.state.set('kb_event', {"
                    "      key: e.key, shift: e.shiftKey, "
                    "ctrl: e.ctrlKey, t: Date.now()"
                    "    });"
                    "    e.preventDefault();"
                    "    e.stopImmediatePropagation();"
                    "  }, true);"
                    "  window._aind_kb_listener_v3 = true;"
                    "} catch (err) {"
                    "  window.console && window.console.error("
                    "'aind keyboard listener install failed:', err);"
                    "}"
                    "})();"
                )
            )
            # Silence the benign ``ResizeObserver loop completed with
            # undelivered notifications`` / ``ResizeObserver loop limit
            # exceeded`` error that Vuetify + trame's render-window
            # sizing fires whenever an observed element schedules a
            # second resize in the same frame. It's a W3C-spec
            # informational notice (not a real failure) but bubbles to
            # ``window.error`` in Chrome / Firefox and clutters the
            # console. Suppress that *one* message; let everything else
            # through.
            client.ClientTriggers(
                mounted=(
                    "(function(){"
                    "if (window._aind_resize_obs_filter) return;"
                    "const SILENCED = ["
                    "  'ResizeObserver loop completed with undelivered notifications.',"
                    "  'ResizeObserver loop limit exceeded'"
                    "];"
                    "window.addEventListener('error', function(e){"
                    "  if (e && e.message && SILENCED.some(function(m){ "
                    "return e.message.indexOf(m) !== -1; })) {"
                    "    e.stopImmediatePropagation();"
                    "    e.preventDefault();"
                    "  }"
                    "}, true);"
                    "window._aind_resize_obs_filter = true;"
                    "})();"
                )
            )

            with layout.content:
                with vuetify3.VContainer(fluid=True, classes="fill-height"):
                    with vuetify3.VRow(classes="fill-height"):
                        self._build_controls(on_set_target)
                        with vuetify3.VCol(cols=9, classes="fill-height"):
                            view = plotter_ui(
                                self.plotter,
                                mode="client",
                                picking_modes=("['click']",),
                                click=(self._on_view_click, "[$event]"),
                                after_scene_loaded=self._on_scene_loaded,
                            )
                            ctrl.view_update = view.update

            # Help dialog — bound to state.kb_help_open (toggled by '?').
            self._build_kb_help_dialog()

    def _build_controls(self, on_set_target) -> None:
        """Build the left-column control widgets.

        Organized into four tabs so the column doesn't outgrow the
        viewport:

        - **Pose** — probe/arc/type selection, R/A/depth + AP/ML/spin
          sliders, target picker, *and* the per-probe readouts +
          keyboard-speed mode that you read off while editing.
        - **Display** — scene visibility toggles + CCF overlay.
        - **Files** — save / export / plan-IO buttons.
        """
        with vuetify3.VCol(cols=3, classes="fill-height overflow-y-auto"):
            with vuetify3.VTabs(
                v_model=("ctrl_tab",),
                density="compact",
                grow=True,
            ):
                vuetify3.VTab(text="Pose", value="pose")
                vuetify3.VTab(text="Display", value="display")
                vuetify3.VTab(text="Files", value="files")
            with vuetify3.VTabsWindow(v_model=("ctrl_tab",), classes="mt-2"):
                with vuetify3.VTabsWindowItem(value="pose"):
                    self._build_pose_tab(on_set_target)
                    vuetify3.VDivider(classes="my-2")
                    self._build_readouts()
                    vuetify3.VDivider(classes="my-2")
                    self._build_kb_speed_controls()
                with vuetify3.VTabsWindowItem(value="display"):
                    self._build_display_controls()
                    if self.ccf_overlay is not None:
                        vuetify3.VDivider(classes="my-2")
                        self._build_ccf_controls()
                with vuetify3.VTabsWindowItem(value="files"):
                    self._build_files_tab()

    def _build_pose_tab(self, on_set_target) -> None:
        """Probe/arc/type + sliders + target — the pose-editing flow."""
        vuetify3.VSelect(
            v_model=("probe",),
            items=("probes",),
            label="Probe",
            hide_details=True,
            density="compact",
        )
        vuetify3.VSelect(
            v_model=("arc",),
            items=("arcs",),
            label="Arc",
            hide_details=True,
            density="compact",
        )
        vuetify3.VSelect(
            v_model=("probe_kind",),
            items=("probe_kinds",),
            label="Probe type",
            hide_details=True,
            density="compact",
        )
        # Hidden when the probe has only one shank — the dropdown would
        # just show "1" and it'd waste a row.
        vuetify3.VSelect(
            v_model=("probe_position_bearing_shank",),
            items=("probe_shank_options",),
            label="Reported shank",
            hide_details=True,
            density="compact",
            v_show=("probe_shank_options.length > 1",),
            classes="mt-1",
        )
        vuetify3.VDivider(classes="my-2")
        self._slider_row("offset_r", "R (mm)", -7.5, 7.5, 0.05)
        self._slider_row("offset_a", "A (mm)", -7.5, 7.5, 0.05)
        self._slider_row("depth", "Depth (mm)", -10, 10, 0.1)
        vuetify3.VDivider(classes="my-2")
        self._slider_row(
            "ap_tilt",
            "AP tilt (°)",
            -60,
            60,
            0.5,
            disabled="probe_locked_axes.includes('ap_tilt')",
        )
        self._slider_row(
            "ml_tilt",
            "ML tilt (°)",
            -60,
            60,
            0.5,
            disabled="probe_locked_axes.includes('ml_tilt')",
        )
        self._slider_row("spin", "Spin (°)", -180, 180, 1)
        vuetify3.VDivider(classes="my-2")
        vuetify3.VSelect(
            v_model=("target",),
            items=("targets",),
            label="Target",
            hide_details=True,
            density="compact",
        )
        vuetify3.VBtn(
            "Set target",
            color="primary",
            click=on_set_target,
            classes="mt-2",
        )
        vuetify3.VDivider(classes="my-2")
        self._build_newscale_section()

    def _build_newscale_section(self) -> None:
        """NewScale machine-coords ↔ probe-pose tools.

        The whole section is shown only when a calibration is loaded for
        the current probe. The Apply inputs + button are disabled when
        the probe isn't currently using calibration (so AP/ML aren't
        locked to the cal). Readouts show the inverse conversion of the
        current pose's position-bearing shank tip.
        """
        with vuetify3.VCard(classes="mt-2", flat=True):
            vuetify3.VCardSubtitle("NewScale")
            with vuetify3.VCardText():
                vuetify3.VSwitch(
                    v_model=("probe_calibrated",),
                    label="Use calibration (lock AP/ML)",
                    hide_details=True,
                    density="compact",
                    disabled=("!probe_has_calibration",),
                )
                vuetify3.VLabel(
                    "Calibrated shank = 'Reported shank' above.",
                    classes="text-caption text-medium-emphasis mb-2",
                )
                vuetify3.VDivider(classes="my-2")
                vuetify3.VLabel(
                    "Apply reading (probe-frame mm) → tip",
                    classes="text-caption mb-1",
                )
                with vuetify3.VRow(dense=True, classes="mb-1"):
                    with vuetify3.VCol(cols=4):
                        vuetify3.VTextField(
                            v_model_number=("probe_newscale_apply_x",),
                            label="x",
                            type="number",
                            step=0.01,
                            density="compact",
                            hide_details=True,
                            disabled=("!probe_calibrated",),
                        )
                    with vuetify3.VCol(cols=4):
                        vuetify3.VTextField(
                            v_model_number=("probe_newscale_apply_y",),
                            label="y",
                            type="number",
                            step=0.01,
                            density="compact",
                            hide_details=True,
                            disabled=("!probe_calibrated",),
                        )
                    with vuetify3.VCol(cols=4):
                        vuetify3.VTextField(
                            v_model_number=("probe_newscale_apply_z",),
                            label="z",
                            type="number",
                            step=0.01,
                            density="compact",
                            hide_details=True,
                            disabled=("!probe_calibrated",),
                        )
                vuetify3.VBtn(
                    "Apply NewScale → tip",
                    color="primary",
                    density="compact",
                    disabled=("!probe_calibrated",),
                    click=self._on_apply_newscale_click,
                )
                vuetify3.VDivider(classes="my-2")
                vuetify3.VLabel(
                    "Current pose NewScale readout",
                    classes="text-caption mb-1",
                )
                vuetify3.VLabel(
                    "x: {{ probe_newscale_readout_x }}   "
                    "y: {{ probe_newscale_readout_y }}   "
                    "z: {{ probe_newscale_readout_z }}",
                    classes="text-body-2",
                )

    def _build_files_tab(self) -> None:
        """Save/export/plan-IO actions — grouped on their own tab so the
        Pose tab stays focused on edits and the row of save buttons
        doesn't crowd the slider stack.

        Verb convention: ``Save`` = re-importable file (this tool can
        reopen it). ``Export`` = derived hand-off (rig technician,
        downstream pipeline; not re-importable). ``Save plan`` and
        ``Load plan`` are deliberately adjacent — they're the
        plan-slice round-trip pair.
        """
        # Save config: server-side write of the full ConfigModel YAML
        # (paths, transforms, asset catalog, scene, options, plan).
        # Re-importable.
        if self.on_save is not None:
            vuetify3.VBtn(
                "Save config",
                color="success",
                click=self.on_save,
                classes="mt-2",
                block=True,
            )
        # Save plan + Load plan: the plan-only YAML round-trip pair.
        # Save: server-side write to the configured plan path. Load: a
        # browser file picker (VFileInput) — uploads the picked YAML to
        # the server, our state.change handler parses + applies. The
        # plan-only YAML has no asset list, so it ports across configs
        # that share probe rosters.
        if self.on_save_plan is not None:
            vuetify3.VBtn(
                "Save plan",
                color="info",
                click=self.on_save_plan,
                classes="mt-2",
                block=True,
            )
        vuetify3.VFileInput(
            v_model=("plan_file",),
            label="Load plan",
            accept=".yml,.yaml",
            show_size=True,
            truncate_length=20,
            density="compact",
            hide_details=True,
            classes="mt-2",
            prepend_icon="mdi-folder-open",
        )
        # Status line for the load (✓ Loaded foo.yml / ⚠ failure msg).
        vuetify3.VLabel(
            "{{ plan_load_status }}",
            v_show=("plan_load_status",),
            classes="mt-1 text-caption",
        )
        # Export poses: derived per-probe geometry hand-off (resolved
        # tip positions, depths, AP/ML/spin numbers). Not re-importable.
        if self.on_export_plan is not None:
            vuetify3.VBtn(
                "Export poses",
                color="primary",
                click=self.on_export_plan,
                classes="mt-2",
                block=True,
            )

    def _slider_row(
        self,
        model: str,
        label: str,
        min_val: float,
        max_val: float,
        step: float,
        disabled: str | None = None,
    ) -> None:
        """Slider + editable number field bound to the same state variable.

        ``disabled`` is an optional Vue expression (e.g. ``"probe_calibrated"``)
        used to gray out both controls when truthy.
        """
        kwargs = {"disabled": (disabled,)} if disabled else {}
        with vuetify3.VRow(align="center", no_gutters=True, dense=True):
            with vuetify3.VCol():
                vuetify3.VSlider(
                    v_model=(model, 0),
                    min=min_val,
                    max=max_val,
                    step=step,
                    label=label,
                    hide_details=True,
                    **kwargs,
                )
            with vuetify3.VCol(cols="auto"):
                vuetify3.VTextField(
                    v_model=(model, 0),
                    type="number",
                    step=step,
                    density="compact",
                    hide_details=True,
                    style="width:80px",
                    **kwargs,
                )

    def _build_readouts(self) -> None:
        """Read-only geometric summary for the currently selected probe."""
        with vuetify3.VRow(no_gutters=True, dense=True):
            with vuetify3.VCol(cols=4):
                vuetify3.VLabel("Tip (RAS)")
            with vuetify3.VCol():
                vuetify3.VLabel("{{ probe_tip_str }}")
        with vuetify3.VRow(no_gutters=True, dense=True):
            with vuetify3.VCol(cols=4):
                vuetify3.VLabel("Depth (brain)")
            with vuetify3.VCol():
                vuetify3.VLabel("{{ probe_depth_str }}")
        with vuetify3.VRow(no_gutters=True, dense=True):
            with vuetify3.VCol(cols=4):
                vuetify3.VLabel("Over-inserted")
            with vuetify3.VCol():
                vuetify3.VLabel("{{ probe_overinsertion_str }}")
        with vuetify3.VRow(no_gutters=True, dense=True):
            with vuetify3.VCol(cols=4):
                vuetify3.VLabel("Kinematic")
            with vuetify3.VCol():
                vuetify3.VLabel("{{ probe_kinematic_str }}")
        # Collisions split into local (this probe) and scene-wide.
        # Local lists the other side of each pair the current probe
        # touches; scene shows a total count so problems elsewhere
        # still surface without you having to switch probes to check.
        with vuetify3.VRow(no_gutters=True, dense=True):
            with vuetify3.VCol(cols=4):
                vuetify3.VLabel("Collide (this)")
            with vuetify3.VCol():
                vuetify3.VLabel("{{ probe_collision_str }}")
        with vuetify3.VRow(no_gutters=True, dense=True):
            with vuetify3.VCol(cols=4):
                vuetify3.VLabel("Collide (all)")
            with vuetify3.VCol():
                vuetify3.VLabel("{{ scene_collision_str }}")

    def _build_display_controls(self) -> None:
        """Per-group visibility toggles + opacity sliders for the major
        asset categories. Probes are always visible."""
        vuetify3.VLabel("Display")
        vuetify3.VBtn(
            "Recenter on brain",
            click=self.recenter_view,
            classes="mt-1 mb-2",
            block=True,
            variant="outlined",
            density="compact",
        )
        for skey, label, _include, _exclude in VISIBILITY_GROUPS:
            with vuetify3.VRow(align="center", no_gutters=True, dense=True):
                with vuetify3.VCol(cols="auto"):
                    vuetify3.VSwitch(
                        v_model=(f"{skey}_visible",),
                        label=label,
                        hide_details=True,
                        density="compact",
                        inset=True,
                    )
                with vuetify3.VCol():
                    vuetify3.VSlider(
                        v_model=(f"{skey}_opacity",),
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        hide_details=True,
                        density="compact",
                    )

    def _build_kb_speed_controls(self) -> None:
        """Speed-mode toggle for keyboard increments (1/2/3 hotkeys also
        switch this). Multiplies the base step; stacks with Shift/Ctrl
        modifiers."""
        with vuetify3.VRow(no_gutters=True, dense=True, align="center"):
            with vuetify3.VCol(cols=4):
                vuetify3.VLabel("Speed")
            with vuetify3.VCol():
                vuetify3.VBtnToggle(
                    v_model=("kb_speed",),
                    mandatory=True,
                    density="compact",
                    divided=True,
                    children=None,
                    hide_details=True,
                ).add_children(
                    [
                        vuetify3.VBtn("Slow", value="slow", size="small"),
                        vuetify3.VBtn("Norm", value="normal", size="small"),
                        vuetify3.VBtn("Fast", value="fast", size="small"),
                    ]
                )

    def _build_kb_help_dialog(self) -> None:
        """Modal that lists the keyboard shortcuts. Toggle with '?'."""
        # Drop duplicate action ids (e.g. arrow-key aliases of WASD).
        seen_actions: set[str] = set()
        rows: list[tuple[str, str]] = []
        for key, action_id, label in KB_ACTIONS:
            if action_id in seen_actions:
                continue
            seen_actions.add(action_id)
            rows.append((key, label))
        rows.append(("Shift + key", "10× step (coarse, stacks with mode)"))
        rows.append(("Ctrl + key", "0.2× step (fine, stacks with mode)"))

        with vuetify3.VDialog(v_model=("kb_help_open",), max_width="480"):
            with vuetify3.VCard():
                vuetify3.VCardTitle("Keyboard shortcuts")
                with vuetify3.VCardText():
                    with vuetify3.VList(density="compact"):
                        for key, label in rows:
                            with vuetify3.VListItem():
                                with vuetify3.VRow(no_gutters=True, align="center"):
                                    with vuetify3.VCol(cols="3"):
                                        vuetify3.VKbd(key)
                                    with vuetify3.VCol():
                                        vuetify3.VListItemTitle(label)
                with vuetify3.VCardActions():
                    vuetify3.VBtn(
                        "Close",
                        color="primary",
                        click="kb_help_open = false",
                    )

    def _build_ccf_controls(self) -> None:
        """Build the CCF region overlay UI widgets."""
        vuetify3.VDivider(classes="my-2")
        vuetify3.VLabel("CCF Regions")
        vuetify3.VAutocomplete(
            v_model=("ccf_selected_region",),
            items=("ccf_search_results", []),
            label="Search brain region",
            hide_details=True,
            density="compact",
            clearable=True,
            classes="mt-1",
            update_search=("ccf_search_query = $event"),
        )
        vuetify3.VSlider(
            v_model=("ccf_global_opacity", 0.3),
            min=0,
            max=1,
            step=0.05,
            label="CCF opacity",
            hide_details=True,
            classes="mt-2",
        )
