"""Tests for ProbeDeclModel field alignment and round-trip serialization."""

import pytest
from pydantic import ValidationError

from aind_low_point.build_runtime import (
    apply_plan_model_to_state,
    planning_state_to_plan_model,
    save_plan_to_config,
)
from aind_low_point.state_change import PlanStore
from aind_low_point.config import (
    CatalogTargetRefModel,
    ConfigModel,
    InlineTargetRefModel,
    NodeTargetRefModel,
    PlanningModel,
    ProbeDeclModel,
)
from aind_low_point.planning import Kinematics, PlanningState, ProbePlan
from tests.config_factories import (
    AssetFactory,
    ConfigFactory,
    TargetFactory,
)


class TestProbeDeclModelFields:
    """Test ProbeDeclModel with new optional fields."""

    def test_arc_optional_accepts_none(self):
        """ProbeDeclModel accepts arc=None (off-arc probes)."""
        decl = ProbeDeclModel(
            kind="neuropixels",
            arc=None,
            target={"kind": "catalog", "key": "t1"},
            bind_ap_to_arc=False,
        )
        assert decl.arc is None
        assert decl.bind_ap_to_arc is False

    def test_arc_optional_accepts_string(self):
        """ProbeDeclModel still accepts arc as a string."""
        decl = ProbeDeclModel(
            kind="neuropixels",
            arc="arc1",
            target={"kind": "catalog", "key": "t1"},
        )
        assert decl.arc == "arc1"

    def test_ap_local_default_none(self):
        """ap_local defaults to None."""
        decl = ProbeDeclModel(
            kind="neuropixels",
            arc="a",
            target={"kind": "catalog", "key": "t1"},
        )
        assert decl.ap_local is None

    def test_ap_local_accepts_value(self):
        """ap_local accepts a float value."""
        decl = ProbeDeclModel(
            kind="neuropixels",
            arc=None,
            target={"kind": "catalog", "key": "t1"},
            ap_local=17.5,
            bind_ap_to_arc=False,
        )
        assert decl.ap_local == 17.5

    def test_bind_ap_to_arc_defaults_true(self):
        """bind_ap_to_arc defaults to True."""
        decl = ProbeDeclModel(
            kind="neuropixels",
            arc="a",
            target={"kind": "catalog", "key": "t1"},
        )
        assert decl.bind_ap_to_arc is True

    def test_target_coercion_still_works(self):
        """String target is still coerced to CatalogTargetRefModel."""
        decl = ProbeDeclModel(
            kind="neuropixels",
            arc="a",
            target="my_target",
        )
        assert decl.target.kind == "catalog"
        assert decl.target.key == "my_target"


class TestProbeDeclValidation:
    """Test cross-reference validation for new ProbeDeclModel fields."""

    def test_bind_ap_to_arc_without_arc_raises(self):
        """bind_ap_to_arc=True with arc=None should raise validation error."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "targets": [TargetFactory.explicit_target(key="target1")],
                "plan": {
                    "arcs": {"arc1": 15.0},
                    "probes": {
                        "probe1": {
                            "kind": "neuropixels",
                            "arc": None,
                            "target": "target1",
                            "bind_ap_to_arc": True,
                        }
                    },
                    "reticles": {},
                    "calibrations": {"files": {}, "probe_to_ref": {}},
                },
            }
        )
        with pytest.raises(
            ValidationError,
            match="bind_ap_to_arc=True but arc is not set",
        ):
            ConfigModel.model_validate(config_data)

    def test_off_arc_probe_valid(self):
        """Probe with arc=None and bind_ap_to_arc=False is valid."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "targets": [TargetFactory.explicit_target(key="target1")],
                "plan": {
                    "arcs": {},
                    "probes": {
                        "probe1": {
                            "kind": "neuropixels",
                            "arc": None,
                            "target": "target1",
                            "bind_ap_to_arc": False,
                            "ap_local": 12.0,
                            "auto_scene": False,
                        }
                    },
                    "reticles": {},
                    "calibrations": {"files": {}, "probe_to_ref": {}},
                },
            }
        )
        config = ConfigModel.model_validate(config_data)
        assert config.plan.probes["probe1"].arc is None
        assert config.plan.probes["probe1"].ap_local == 12.0
        assert config.plan.probes["probe1"].bind_ap_to_arc is False

    def test_arc_with_ap_local_valid(self):
        """Probe can have both arc and ap_local (ap_local is override hint)."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "targets": [TargetFactory.explicit_target(key="target1")],
                "plan": {
                    "arcs": {"arc1": 15.0},
                    "probes": {
                        "probe1": {
                            "kind": "neuropixels",
                            "arc": "arc1",
                            "target": "target1",
                            "ap_local": 20.0,
                            "auto_scene": False,
                        }
                    },
                    "reticles": {},
                    "calibrations": {"files": {}, "probe_to_ref": {}},
                },
            }
        )
        config = ConfigModel.model_validate(config_data)
        assert config.plan.probes["probe1"].ap_local == 20.0


class TestPlanningStateToModel:
    """Test planning_state_to_plan_model round-trip."""

    @staticmethod
    def _make_state(
        arc_angles=None,
        probes=None,
    ) -> PlanningState:
        """Helper to build a PlanningState."""
        return PlanningState(
            kinematics=Kinematics(arc_angles=arc_angles or {"a": 15.0}),
            probes=probes
            or {
                "p1": ProbePlan(
                    kind="neuropixels",
                    arc_id="a",
                    bind_ap_to_arc=True,
                    ap_local=15.0,
                    ml_local=3.0,
                    spin=1.0,
                    past_target_mm=2.0,
                    offsets_RA=(0.5, 0.3),
                    target_key="t1",
                    calibrated=False,
                ),
            },
        )

    @staticmethod
    def _make_original_plan() -> PlanningModel:
        """Helper to build a PlanningModel."""
        return PlanningModel(
            arcs={"a": 15.0},
            probes={
                "p1": ProbeDeclModel(
                    kind="neuropixels",
                    arc="a",
                    target=CatalogTargetRefModel(key="t1"),
                    slider_ml=3.0,
                    spin=1.0,
                    past_target_mm=2.0,
                    offsets_RA=[0.5, 0.3],
                ),
            },
        )

    def test_round_trip_basic(self):
        """Basic round-trip preserves all probe fields."""
        state = self._make_state()
        original = self._make_original_plan()

        result = planning_state_to_plan_model(state, original)

        assert result.arcs == {"a": 15.0}
        p = result.probes["p1"]
        assert p.kind == "neuropixels"
        assert p.arc == "a"
        assert p.slider_ml == 3.0
        assert p.spin == 1.0
        assert p.ap_local == 15.0
        assert p.bind_ap_to_arc is True
        assert p.past_target_mm == 2.0
        assert p.offsets_RA == [0.5, 0.3]
        assert p.calibrated is False
        assert p.target.kind == "catalog"
        assert p.target.key == "t1"

    def test_mutated_arc_angle(self):
        """Changed arc angle is reflected in output."""
        state = self._make_state()
        state.kinematics.arc_angles["a"] = 22.0

        original = self._make_original_plan()
        result = planning_state_to_plan_model(state, original)

        assert result.arcs["a"] == 22.0

    def test_mutated_probe_fields(self):
        """Changed probe fields are reflected in output."""
        state = self._make_state()
        p = state.probes["p1"]
        p.spin = 45.0
        p.past_target_mm = 5.0
        p.offsets_RA = (1.0, 2.0)

        original = self._make_original_plan()
        result = planning_state_to_plan_model(state, original)

        rp = result.probes["p1"]
        assert rp.spin == 45.0
        assert rp.past_target_mm == 5.0
        assert rp.offsets_RA == [1.0, 2.0]

    def test_target_key_unchanged_preserves_kind(self):
        """When target_key matches original, the TargetRef kind is preserved."""
        state = self._make_state()
        original = PlanningModel(
            arcs={"a": 15.0},
            probes={
                "p1": ProbeDeclModel(
                    kind="neuropixels",
                    arc="a",
                    target=NodeTargetRefModel(key="t1"),
                ),
            },
        )
        result = planning_state_to_plan_model(state, original)
        assert result.probes["p1"].target.kind == "node"

    def test_target_key_changed_defaults_catalog(self):
        """When target_key differs from original, default to catalog kind."""
        state = self._make_state()
        state.probes["p1"].target_key = "new_target"

        original = self._make_original_plan()
        result = planning_state_to_plan_model(state, original)

        assert result.probes["p1"].target.kind == "catalog"
        assert result.probes["p1"].target.key == "new_target"

    def test_new_probe_not_in_original(self):
        """A probe added at runtime (not in original) is serialized."""
        state = self._make_state()
        state.probes["p2"] = ProbePlan(
            kind="tetrode",
            arc_id=None,
            bind_ap_to_arc=False,
            ap_local=10.0,
            ml_local=0.0,
            target_key="t1",
        )
        original = self._make_original_plan()
        result = planning_state_to_plan_model(state, original)

        assert "p2" in result.probes
        p2 = result.probes["p2"]
        assert p2.kind == "tetrode"
        assert p2.arc is None
        assert p2.bind_ap_to_arc is False
        assert p2.ap_local == 10.0

    def test_preserves_reticles_and_calibrations(self):
        """Reticles and calibrations are preserved from original."""
        state = self._make_state()
        original = self._make_original_plan()

        result = planning_state_to_plan_model(state, original)

        assert result.reticles == original.reticles
        assert result.calibrations == original.calibrations


class TestSavePlanToConfig:
    """Test save_plan_to_config preserves non-plan sections."""

    def test_preserves_non_plan_sections(self):
        """Non-plan sections of config are preserved."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "assets": [
                    AssetFactory.mesh_asset(key="brain_mesh"),
                ],
                "targets": [
                    TargetFactory.explicit_target(key="target1"),
                ],
                "plan": {
                    "arcs": {"arc1": 15.0},
                    "probes": {
                        "probe1": {
                            "kind": "neuropixels",
                            "arc": "arc1",
                            "target": "target1",
                            "auto_scene": False,
                        }
                    },
                    "reticles": {},
                    "calibrations": {"files": {}, "probe_to_ref": {}},
                },
            }
        )
        original_config = ConfigModel.model_validate(config_data)

        # Build a PlanningState from the original
        state = PlanningState(
            kinematics=Kinematics(arc_angles={"arc1": 15.0}),
            probes={
                "probe1": ProbePlan(
                    kind="neuropixels",
                    arc_id="arc1",
                    bind_ap_to_arc=True,
                    ap_local=15.0,
                    ml_local=0.0,
                    target_key="target1",
                ),
            },
        )

        # Mutate state
        state.kinematics.arc_angles["arc1"] = 25.0
        state.probes["probe1"].spin = 10.0

        result = save_plan_to_config(state, original_config)

        # Plan section is updated
        assert result.plan.arcs["arc1"] == 25.0
        assert result.plan.probes["probe1"].spin == 10.0

        # Non-plan sections are preserved
        assert len(result.assets) == len(original_config.assets)
        assert len(result.targets) == len(original_config.targets)
        assert result.version == original_config.version

    def test_serialization_produces_valid_json(self):
        """The output can be serialized to a JSON-compatible dict."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "targets": [
                    TargetFactory.explicit_target(key="target1"),
                ],
                "plan": {
                    "arcs": {"arc1": 15.0},
                    "probes": {
                        "probe1": {
                            "kind": "neuropixels",
                            "arc": "arc1",
                            "target": "target1",
                            "auto_scene": False,
                        }
                    },
                    "reticles": {},
                    "calibrations": {"files": {}, "probe_to_ref": {}},
                },
            }
        )
        original_config = ConfigModel.model_validate(config_data)

        state = PlanningState(
            kinematics=Kinematics(arc_angles={"arc1": 15.0}),
            probes={
                "probe1": ProbePlan(
                    kind="neuropixels",
                    arc_id="arc1",
                    bind_ap_to_arc=True,
                    ap_local=15.0,
                    ml_local=0.0,
                    target_key="target1",
                ),
            },
        )

        result = save_plan_to_config(state, original_config)
        dumped = result.model_dump(mode="json")

        # Should be a plain dict with no special types
        assert isinstance(dumped, dict)
        assert isinstance(dumped["plan"]["arcs"]["arc1"], float)
        assert isinstance(dumped["plan"]["probes"]["probe1"]["kind"], str)


class TestInlineTargetRefModel:
    """Test InlineTargetRefModel and its integration."""

    def test_construction(self):
        """InlineTargetRefModel can be constructed directly."""
        ref = InlineTargetRefModel(point_RAS=[1.0, 2.0, 3.0])
        assert ref.kind == "inline"
        assert ref.point_RAS == [1.0, 2.0, 3.0]

    def test_probe_decl_with_inline_target(self):
        """ProbeDeclModel accepts an inline target dict."""
        decl = ProbeDeclModel(
            kind="neuropixels",
            arc=None,
            bind_ap_to_arc=False,
            target={"kind": "inline", "point_RAS": [10.0, 20.0, 30.0]},
        )
        assert decl.target.kind == "inline"
        assert decl.target.point_RAS == [10.0, 20.0, 30.0]

    def test_coerce_list_to_inline(self):
        """A bare [x, y, z] list is coerced to InlineTargetRefModel."""
        decl = ProbeDeclModel(
            kind="neuropixels",
            arc=None,
            bind_ap_to_arc=False,
            target=[1.5, 2.5, 3.5],
        )
        assert decl.target.kind == "inline"
        assert decl.target.point_RAS == [1.5, 2.5, 3.5]

    def test_coerce_tuple_to_inline(self):
        """A bare (x, y, z) tuple is coerced to InlineTargetRefModel."""
        decl = ProbeDeclModel(
            kind="neuropixels",
            arc=None,
            bind_ap_to_arc=False,
            target=(4.0, 5.0, 6.0),
        )
        assert decl.target.kind == "inline"
        assert decl.target.point_RAS == [4.0, 5.0, 6.0]

    def test_config_validation_inline_target(self):
        """Config with inline target probe passes validation."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "plan": {
                    "arcs": {},
                    "probes": {
                        "probe1": {
                            "kind": "neuropixels",
                            "arc": None,
                            "bind_ap_to_arc": False,
                            "target": {
                                "kind": "inline",
                                "point_RAS": [10.0, 20.0, 30.0],
                            },
                            "auto_scene": False,
                        }
                    },
                    "reticles": {},
                    "calibrations": {"files": {}, "probe_to_ref": {}},
                },
            }
        )
        config = ConfigModel.model_validate(config_data)
        p = config.plan.probes["probe1"]
        assert p.target.kind == "inline"
        assert p.target.point_RAS == [10.0, 20.0, 30.0]

    def test_round_trip_target_point_RAS(self):
        """target_point_RAS set at runtime round-trips as InlineTargetRefModel."""
        state = PlanningState(
            kinematics=Kinematics(arc_angles={"a": 15.0}),
            probes={
                "p1": ProbePlan(
                    kind="neuropixels",
                    arc_id="a",
                    bind_ap_to_arc=True,
                    ap_local=15.0,
                    ml_local=0.0,
                    target_key=None,
                    target_point_RAS=(10.0, 20.0, 30.0),
                ),
            },
        )
        original = PlanningModel(
            arcs={"a": 15.0},
            probes={
                "p1": ProbeDeclModel(
                    kind="neuropixels",
                    arc="a",
                    target=CatalogTargetRefModel(key="t1"),
                ),
            },
        )
        result = planning_state_to_plan_model(state, original)
        p = result.probes["p1"]
        assert p.target.kind == "inline"
        assert p.target.point_RAS == [10.0, 20.0, 30.0]

    def test_round_trip_inline_target_in_config(self):
        """Inline target in config survives save_plan_to_config round-trip."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "plan": {
                    "arcs": {},
                    "probes": {
                        "probe1": {
                            "kind": "neuropixels",
                            "arc": None,
                            "bind_ap_to_arc": False,
                            "target": {
                                "kind": "inline",
                                "point_RAS": [10.0, 20.0, 30.0],
                            },
                            "auto_scene": False,
                        }
                    },
                    "reticles": {},
                    "calibrations": {"files": {}, "probe_to_ref": {}},
                },
            }
        )
        original_config = ConfigModel.model_validate(config_data)

        state = PlanningState(
            kinematics=Kinematics(arc_angles={}),
            probes={
                "probe1": ProbePlan(
                    kind="neuropixels",
                    arc_id=None,
                    bind_ap_to_arc=False,
                    ap_local=0.0,
                    ml_local=0.0,
                    target_key=None,
                    target_point_RAS=(10.0, 20.0, 30.0),
                ),
            },
        )

        result = save_plan_to_config(state, original_config)
        p = result.plan.probes["probe1"]
        assert p.target.kind == "inline"
        assert p.target.point_RAS == [10.0, 20.0, 30.0]

    def test_inline_target_serializes_to_json(self):
        """Inline target serializes cleanly to JSON dict."""
        decl = ProbeDeclModel(
            kind="neuropixels",
            arc=None,
            bind_ap_to_arc=False,
            target=InlineTargetRefModel(point_RAS=[1.0, 2.0, 3.0]),
        )
        dumped = decl.model_dump(mode="json")
        assert dumped["target"] == {
            "kind": "inline",
            "point_RAS": [1.0, 2.0, 3.0],
        }


class TestExplicitNodeTracking:
    """Test auto-generated vs explicit scene node tracking."""

    def test_explicit_node_keys_recorded(self):
        """SceneModel._explicit_node_keys is populated during validation."""
        config_data = ConfigFactory.minimal_config()
        config_data["scene"] = {"nodes": [{"key": "my_node", "asset": "x"}]}
        config_data["assets"] = [AssetFactory.mesh_asset(key="x")]
        config = ConfigModel.model_validate(config_data)
        assert "my_node" in config.scene._explicit_node_keys

    def test_auto_generated_nodes_not_in_explicit(self):
        """Auto-generated probe nodes are not in _explicit_node_keys."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "assets": [
                    AssetFactory.mesh_asset(key="probe:neuropixels"),
                ],
                "targets": [TargetFactory.explicit_target(key="t1")],
                "plan": {
                    "arcs": {"arc1": 15.0},
                    "probes": {
                        "probe1": {
                            "kind": "neuropixels",
                            "arc": "arc1",
                            "target": "t1",
                        }
                    },
                    "reticles": {},
                    "calibrations": {"files": {}, "probe_to_ref": {}},
                },
            }
        )
        config = ConfigModel.model_validate(config_data)
        # Auto-generated node exists in scene
        node_keys = {n.key for n in config.scene.nodes}
        assert "probe:probe1" in node_keys
        # But not in explicit set
        assert "probe:probe1" not in config.scene._explicit_node_keys

    def test_save_plan_add_probe_generates_node(self):
        """Adding a probe at runtime produces its auto-generated scene node."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "assets": [
                    AssetFactory.mesh_asset(key="probe:neuropixels"),
                ],
                "targets": [TargetFactory.explicit_target(key="t1")],
                "plan": {
                    "arcs": {"arc1": 15.0},
                    "probes": {
                        "probe1": {
                            "kind": "neuropixels",
                            "arc": "arc1",
                            "target": "t1",
                        }
                    },
                    "reticles": {},
                    "calibrations": {"files": {}, "probe_to_ref": {}},
                },
            }
        )
        original_config = ConfigModel.model_validate(config_data)

        # Runtime state adds a second probe
        state = PlanningState(
            kinematics=Kinematics(arc_angles={"arc1": 15.0}),
            probes={
                "probe1": ProbePlan(
                    kind="neuropixels",
                    arc_id="arc1",
                    bind_ap_to_arc=True,
                    ap_local=15.0,
                    ml_local=0.0,
                    target_key="t1",
                ),
                "probe2": ProbePlan(
                    kind="neuropixels",
                    arc_id="arc1",
                    bind_ap_to_arc=True,
                    ap_local=15.0,
                    ml_local=0.0,
                    target_key="t1",
                ),
            },
        )

        result = save_plan_to_config(state, original_config)
        node_keys = {n.key for n in result.scene.nodes}
        assert "probe:probe1" in node_keys
        assert "probe:probe2" in node_keys

    def test_save_plan_remove_probe_removes_node(self):
        """Removing a probe at runtime removes its auto-generated scene node."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "assets": [
                    AssetFactory.mesh_asset(key="probe:neuropixels"),
                ],
                "targets": [TargetFactory.explicit_target(key="t1")],
                "plan": {
                    "arcs": {"arc1": 15.0},
                    "probes": {
                        "probe1": {
                            "kind": "neuropixels",
                            "arc": "arc1",
                            "target": "t1",
                        },
                        "probe2": {
                            "kind": "neuropixels",
                            "arc": "arc1",
                            "target": "t1",
                        },
                    },
                    "reticles": {},
                    "calibrations": {"files": {}, "probe_to_ref": {}},
                },
            }
        )
        original_config = ConfigModel.model_validate(config_data)

        # Runtime state only has probe1 (probe2 was removed)
        state = PlanningState(
            kinematics=Kinematics(arc_angles={"arc1": 15.0}),
            probes={
                "probe1": ProbePlan(
                    kind="neuropixels",
                    arc_id="arc1",
                    bind_ap_to_arc=True,
                    ap_local=15.0,
                    ml_local=0.0,
                    target_key="t1",
                ),
            },
        )

        result = save_plan_to_config(state, original_config)
        node_keys = {n.key for n in result.scene.nodes}
        assert "probe:probe1" in node_keys
        assert "probe:probe2" not in node_keys

    def test_explicit_nodes_preserved_after_save(self):
        """Explicit scene nodes survive round-trip through save_plan_to_config."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "assets": [
                    AssetFactory.mesh_asset(key="brain"),
                    AssetFactory.mesh_asset(key="probe:neuropixels"),
                ],
                "targets": [TargetFactory.explicit_target(key="t1")],
                "scene": {"nodes": [{"key": "my_brain_node", "asset": "brain"}]},
                "plan": {
                    "arcs": {"arc1": 15.0},
                    "probes": {
                        "probe1": {
                            "kind": "neuropixels",
                            "arc": "arc1",
                            "target": "t1",
                        }
                    },
                    "reticles": {},
                    "calibrations": {"files": {}, "probe_to_ref": {}},
                },
            }
        )
        original_config = ConfigModel.model_validate(config_data)

        state = PlanningState(
            kinematics=Kinematics(arc_angles={"arc1": 15.0}),
            probes={
                "probe1": ProbePlan(
                    kind="neuropixels",
                    arc_id="arc1",
                    bind_ap_to_arc=True,
                    ap_local=15.0,
                    ml_local=0.0,
                    target_key="t1",
                ),
            },
        )

        result = save_plan_to_config(state, original_config)
        node_keys = {n.key for n in result.scene.nodes}
        # Explicit node preserved
        assert "my_brain_node" in node_keys
        # Auto-generated probe node present
        assert "probe:probe1" in node_keys


class TestApplyPlanModelToState:
    """Test ``apply_plan_model_to_state`` — the load side of the
    Save plan / Load plan UI buttons."""

    @staticmethod
    def _make_initial_state() -> PlanningState:
        return PlanningState(
            kinematics=Kinematics(arc_angles={"a": 0.0, "b": 0.0}),
            probes={
                "P1": ProbePlan(
                    kind="2.1", arc_id="a", bind_ap_to_arc=True,
                    ml_local=0.0, spin=0.0,
                    past_target_mm=0.0, offsets_RA=(0.0, 0.0),
                    target_key="t1", calibrated=False,
                    position_bearing_shank=1,
                ),
                "P2": ProbePlan(
                    kind="2.1", arc_id="a", bind_ap_to_arc=True,
                    ml_local=0.0, spin=0.0,
                    past_target_mm=0.0, offsets_RA=(0.0, 0.0),
                    target_key="t2", calibrated=False,
                    position_bearing_shank=1,
                ),
            },
        )

    def test_apply_overwrites_arcs_and_probes(self):
        """Loaded plan overrides arc angles and per-probe fields."""
        state = self._make_initial_state()
        store = PlanStore(state)
        loaded = PlanningModel(
            arcs={"a": 12.5, "b": -8.0},
            probes={
                "P1": ProbeDeclModel(
                    kind="quadbase", arc="b",
                    target=CatalogTargetRefModel(key="t1"),
                    slider_ml=4.5, spin=141.0,
                    past_target_mm=0.5, offsets_RA=[0.1, -0.2],
                    position_bearing_shank=4,
                ),
                "P2": ProbeDeclModel(
                    kind="2.1", arc="a",
                    target=CatalogTargetRefModel(key="t2"),
                    slider_ml=-3.0, spin=20.0,
                ),
            },
        )
        touched = apply_plan_model_to_state(loaded, store)
        assert set(touched) == {"P1", "P2"}
        assert store.state.kinematics.arc_angles["a"] == 12.5
        assert store.state.kinematics.arc_angles["b"] == -8.0
        p1 = store.state.probes["P1"]
        assert p1.kind == "quadbase"
        assert p1.arc_id == "b"
        assert p1.ml_local == 4.5
        assert p1.spin == 141.0
        assert p1.past_target_mm == 0.5
        assert p1.offsets_RA == (0.1, -0.2)
        assert p1.position_bearing_shank == 4
        p2 = store.state.probes["P2"]
        assert p2.spin == 20.0
        assert p2.ml_local == -3.0

    def test_apply_skips_unknown_probes(self, capsys):
        """Probes not in the current state are skipped (with a stdout note)."""
        state = self._make_initial_state()
        store = PlanStore(state)
        loaded = PlanningModel(
            arcs={"a": 5.0},
            probes={
                "GHOST": ProbeDeclModel(
                    kind="2.1", arc="a",
                    target=CatalogTargetRefModel(key="t1"),
                ),
            },
        )
        touched = apply_plan_model_to_state(loaded, store)
        assert touched == []
        captured = capsys.readouterr()
        assert "GHOST" in captured.out

    def test_save_then_load_round_trip(self):
        """``planning_state_to_plan_model`` followed by
        ``apply_plan_model_to_state`` reproduces the source state."""
        # Source state with non-trivial values.
        src = PlanningState(
            kinematics=Kinematics(arc_angles={"a": 13.0, "b": -10.0}),
            probes={
                "MD": ProbePlan(
                    kind="quadbase", arc_id="a", bind_ap_to_arc=True,
                    ml_local=-12.0, spin=141.0,
                    past_target_mm=0.0675, offsets_RA=(0.0, 0.0),
                    target_key="MD_target", calibrated=False,
                    position_bearing_shank=1,
                ),
            },
        )
        original = PlanningModel(
            arcs={"a": 13.0, "b": -10.0},
            probes={
                "MD": ProbeDeclModel(
                    kind="quadbase", arc="a",
                    target=CatalogTargetRefModel(key="MD_target"),
                    slider_ml=-12.0, spin=141.0,
                    past_target_mm=0.0675, offsets_RA=[0.0, 0.0],
                    position_bearing_shank=1,
                ),
            },
        )
        saved_plan = planning_state_to_plan_model(src, original)

        # Fresh state with default values.
        dst = PlanningState(
            kinematics=Kinematics(arc_angles={"a": 0.0, "b": 0.0}),
            probes={
                "MD": ProbePlan(
                    kind="2.1", arc_id="a", bind_ap_to_arc=True,
                    ml_local=0.0, spin=0.0,
                    past_target_mm=0.0, offsets_RA=(0.0, 0.0),
                    target_key="MD_target", calibrated=False,
                    position_bearing_shank=1,
                ),
            },
        )
        store = PlanStore(dst)
        apply_plan_model_to_state(saved_plan, store)

        assert store.state.kinematics.arc_angles == src.kinematics.arc_angles
        loaded = store.state.probes["MD"]
        for field in (
            "kind", "arc_id", "bind_ap_to_arc",
            "ml_local", "spin", "past_target_mm",
            "target_key", "position_bearing_shank",
        ):
            assert getattr(loaded, field) == getattr(src.probes["MD"], field), field
        assert loaded.offsets_RA == src.probes["MD"].offsets_RA
