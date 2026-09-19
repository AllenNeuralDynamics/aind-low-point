"""Plan models: arcs, probes, calibrations and the head mount."""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    model_validator,
)


# -----------------------------------------------------------------------------
# Domain (mechanics: arcs, probes, calibrations, target declarations)
# -----------------------------------------------------------------------------
class CatalogTargetRefModel(BaseModel):
    model_config = {"extra": "forbid"}

    kind: Literal["catalog"] = "catalog"
    key: str  # TargetSpecModel.key


class NodeTargetRefModel(BaseModel):
    model_config = {"extra": "forbid"}

    kind: Literal["node"] = "node"
    key: str  # SceneNodeModel.id / NodeInstance.id


class InlineTargetRefModel(BaseModel):
    """Ad-hoc target specified as a single RAS coordinate."""

    model_config = {"extra": "forbid"}

    kind: Literal["inline"] = "inline"
    point_RAS: list[float] = Field(..., min_length=3, max_length=3)


TargetRef = Annotated[
    Union[CatalogTargetRefModel, NodeTargetRefModel, InlineTargetRefModel],
    Field(discriminator="kind"),
]


class ProbeDeclModel(BaseModel):
    model_config = {"extra": "forbid"}

    kind: str
    arc: Optional[str] = None
    slider_ml: float = 0.0
    spin: float = 0.0

    # Per-probe AP override. If None, AP comes from arc angle at load time.
    ap_local: Optional[float] = None
    bind_ap_to_arc: bool = True

    target: TargetRef
    past_target_mm: float = 0.0
    offsets_RA: list[float] = Field(
        default_factory=lambda: [0.0, 0.0], min_length=2, max_length=2
    )

    # Which shank's tip is the kinematic pivot (1-indexed). For
    # multi-shank probes the user may want shank 1, 2, 3, or 4 to be
    # "the named shank" — the one whose tip lands at the inline target
    # when ``past_target_mm = 0``, the one whose RAS position is shown
    # in the readout, and the one along which the brain-surface depth
    # is measured. For single-shank probes there's only one option;
    # default 1 covers both cases. ``past_target_mm`` is interpreted
    # as the named shank's tip distance past target, measured along
    # the shaft direction.
    position_bearing_shank: int = 1

    # initial lock state; actual calibration affine comes from 'calibrations' map
    calibrated: bool = False

    # Auto scene node generation (creates "probe:{name}" node)
    auto_scene: bool = True
    scene_tags: list[str] = Field(default_factory=lambda: ["probe", "dynamic"])

    @model_validator(mode="before")
    @classmethod
    def _coerce_target(cls, v):
        if isinstance(v, dict) and "target" in v:
            t = v["target"]
            if isinstance(t, str):
                v["target"] = {"kind": "catalog", "key": t}
            elif isinstance(t, (list, tuple)) and len(t) == 3:
                v["target"] = {"kind": "inline", "point_RAS": list(t)}
        return v


class CalibrationRefModel(BaseModel):
    """Reference a specific calibration entry inside a calibration file."""

    model_config = {"extra": "forbid"}

    cal_id: str  # key into CalibrationsModel.files
    probe_code: str  # 5-digit code in the file (keep as str; accept ints)

    # allow shorthand "cal_id:probe_code"
    @classmethod
    def from_string(cls, s: str) -> "CalibrationRefModel":
        if ":" not in s:
            raise ValueError("Expected '<cal_id>:<probe_code>'")
        cal_id, probe_code = s.split(":", 1)
        return cls(cal_id=cal_id.strip(), probe_code=str(probe_code).strip())


class CalibrationReticleModel(BaseModel):
    """Model for calibration reticle used in calibrations"""

    model_config = {"extra": "forbid"}

    offset_RAS: list[float] = Field(default_factory=list, min_length=3, max_length=3)
    rotation_z: float = 0.0


class CalibrationSourceModel(BaseModel):
    """
    One calibration 'bank' source:
      - EITHER a single file (e.g., .xlsx). In this case NO reticle is allowed.
      - OR a directory for parallax. In this case a reticle IS REQUIRED.
    """

    model_config = {"extra": "forbid"}

    file: Optional[FilePath] = Field(
        default=None, description="Path to a single calibration file (e.g., .xlsx)"
    )
    directory: Optional[DirectoryPath] = Field(
        default=None, description="Path to a parallax calibration directory"
    )
    reticle: Optional[str] = Field(
        default=None, description="Name of reticle (required when 'directory' is set)"
    )

    @model_validator(mode="after")
    def _xor_and_require_reticle(self):
        has_file = self.file is not None
        has_dir = self.directory is not None
        if has_file == has_dir:
            # both set or both None → invalid
            raise ValueError(
                "Specify exactly one of 'file' or 'directory' in a calibration source"
            )

        if has_file and self.reticle is not None:
            # forbid reticle with file
            raise ValueError("'reticle' must not be provided when 'file' is used")

        if has_dir and not self.reticle:
            # require reticle with directory
            raise ValueError("'reticle' is required when 'directory' is used")

        return self


class CalibrationsModel(BaseModel):
    """Two equivalent ways to wire calibrations:

    **Legacy mode** — ``files`` is a named dict of sources and
    ``probe_to_ref`` picks one ``(cal_id, probe_code)`` per probe.

    **Stacked mode** — ``sources`` is an ordered list (mix of manual
    files and parallax dirs); ``probe_to_code`` maps each probe name
    to a probe code. All sources are merged into a single
    ``code → (R, t)`` bank with **last source wins** semantics, then
    each probe's code is looked up in the merged bank. Use this when
    you have multiple calibration runs and want to layer them.

    Exactly one mode at a time: stacked-mode fields and legacy-mode
    fields cannot both be non-empty.
    """

    model_config = {"extra": "forbid"}

    files: dict[str, CalibrationSourceModel] = Field(default_factory=dict)
    # probe_name → {"cal_id": "...", "probe_code": "..."} OR "cal_id:probe_code"
    probe_to_ref: dict[str, Union[CalibrationRefModel, str]] = Field(
        default_factory=dict
    )
    sources: list[CalibrationSourceModel] = Field(default_factory=list)
    probe_to_code: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_refs_and_check_modes(self):
        # convert any string refs to CalibrationRefModel
        normalized: dict[str, CalibrationRefModel] = {}
        for probe_name, ref in self.probe_to_ref.items():
            if isinstance(ref, str):
                normalized[probe_name] = CalibrationRefModel.from_string(ref)
            else:
                normalized[probe_name] = ref
        object.__setattr__(self, "probe_to_ref", normalized)

        legacy_set = bool(self.files) or bool(self.probe_to_ref)
        stacked_set = bool(self.sources) or bool(self.probe_to_code)
        if legacy_set and stacked_set:
            raise ValueError(
                "calibrations: cannot mix legacy (files/probe_to_ref) and "
                "stacked (sources/probe_to_code) modes — pick one."
            )
        # In stacked mode the two fields must agree on presence: an
        # empty probe_to_code with non-empty sources is just dead config,
        # and vice versa. Fail loudly so the user notices.
        if bool(self.sources) != bool(self.probe_to_code):
            raise ValueError(
                "calibrations: in stacked mode, 'sources' and "
                "'probe_to_code' must both be non-empty (got "
                f"sources={len(self.sources)}, "
                f"probe_to_code={len(self.probe_to_code)})."
            )
        return self


class HeadMountModel(BaseModel):
    """Rotation from the rig's mechanical frame to subject anatomical LPS.

    Encodes the head-tilt of the mounted mouse. When the mouse is bolted
    to the headframe, the rig's mechanical axes typically don't align
    with the subject's anatomical axes — there's a fixed pitch about
    the right-left (R) axis from how the headframe holds the head.

    The rotation is applied as
    ``R_probe_in_subject = R_subject_from_rig @ arc_angles_to_affine(ap, ml, spin)``
    so the optimizer's ``(ap, ml, spin)`` variables are interpreted in
    the rig's frame (and so are the rig's angular limits / separation
    thresholds), while all geometry stays in subject anatomical LPS.

    Specified as axis-angle in subject-LPS basis (so ``[1, 0, 0]`` is
    L, ``[0, 1, 0]`` is P, ``[0, 0, 1]`` is S). Right-hand rule about
    the axis. For the AIND headframe (mouse mounted nose-down 14°),
    use ``axis_LPS=(1, 0, 0), angle_deg=14`` — equivalent to a -14°
    rotation about the R-axis (since R = -L).

    Default ``(axis_LPS=(1, 0, 0), angle_deg=0)`` = identity, which
    matches the legacy behaviour where rig and subject were silently
    assumed to coincide.
    """

    model_config = {"extra": "forbid"}

    axis_LPS: tuple[float, float, float] = (1.0, 0.0, 0.0)
    angle_deg: float = 0.0


class PlanningModel(BaseModel):
    model_config = {"extra": "forbid"}

    arcs: dict[str, float] = Field(
        default_factory=dict, description="arc_id → AP angle (deg)"
    )
    subject_from_rig: HeadMountModel = Field(
        default_factory=HeadMountModel,
        description=(
            "Rotation from rig mechanical frame to subject anatomical LPS. "
            "Encodes the mounted mouse's head tilt. See HeadMountModel."
        ),
    )
    probes: dict[str, ProbeDeclModel] = Field(
        default_factory=dict, description="probe_name → probe declaration"
    )
    reticles: dict[str, CalibrationReticleModel] = Field(default_factory=dict)
    calibrations: CalibrationsModel = Field(default_factory=CalibrationsModel)
