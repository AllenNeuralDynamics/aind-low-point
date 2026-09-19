"""The keyboard map: which key means which planning action, and how fast.

A leaf, because the controller dispatches from it and the layout draws the
help dialog and the JS interception list from the same tables.
"""

from __future__ import annotations

# Gaming-style keyboard shortcuts: WASD for R/A offsets, IJKL for tilts, UO
# for spin, EQ for depth, Tab to cycle probes, ? for help. Shift is a coarse
# 10× one-shot, Ctrl a fine 0.2×.
KB_ACTIONS: list[tuple[str, str, str]] = [
    # (key, action_id, display label)
    ("w", "a_inc", "+A offset"),
    ("s", "a_dec", "−A offset"),
    ("a", "r_dec", "−R offset"),
    ("d", "r_inc", "+R offset"),
    ("ArrowUp", "a_inc", "+A offset"),
    ("ArrowDown", "a_dec", "−A offset"),
    ("ArrowLeft", "r_dec", "−R offset"),
    ("ArrowRight", "r_inc", "+R offset"),
    # ``r`` and ``f`` are left for VTK.js's defaults (reset camera /
    # fly to point); depth uses the "elevator" convention with
    # ``e`` = extend deeper, ``q`` = retract shallower.
    ("e", "depth_inc", "Deeper (+depth)"),
    ("q", "depth_dec", "Shallower (−depth)"),
    ("i", "ap_inc", "AP tilt up"),
    ("k", "ap_dec", "AP tilt down"),
    ("j", "ml_dec", "ML tilt left"),
    ("l", "ml_inc", "ML tilt right"),
    ("u", "spin_dec", "Spin −"),
    ("o", "spin_inc", "Spin +"),
    ("Tab", "next_probe", "Next probe"),
    ("1", "speed_slow", "Slow speed"),
    ("2", "speed_normal", "Normal speed"),
    ("3", "speed_fast", "Fast speed"),
    ("c", "recenter", "Recenter on brain"),
    ("t", "focus_target", "Focus on target"),
    ("?", "help", "Toggle help"),
]

# Persistent speed-mode multipliers. Stack with Shift/Ctrl one-shot
# multipliers (Shift = ×10 coarse, Ctrl = ×0.2 fine), so the effective
# step is base × mode × modifier.
KB_SPEED_MULTIPLIER: dict[str, float] = {
    "slow": 0.5,
    "normal": 1.0,
    "fast": 5.0,
}

KB_SPEED_LABEL: dict[str, str] = {
    "slow": "Slow",
    "normal": "Normal",
    "fast": "Fast",
}

# Deduplicate keys for the JS-side fan-out (Tab and ArrowKeys need
# preventDefault; we send the lowercase key + modifier flags, the
# server picks the action.)
KB_KEYS_TO_INTERCEPT = sorted({k for k, *_ in KB_ACTIONS})
