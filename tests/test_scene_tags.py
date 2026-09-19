"""Asset tags reach the node, and a near-miss tag is called out.

`AssetSpec.tags` used to be filled from config on every asset and queried by
nothing, so a label like `structure` on an asset was invisible to anything that
filters scene nodes. Tags themselves stay open — a subject groups its nodes
however it likes — but a tag one letter from one the code acts on is a typo, and
a typo is silent: a misspelled `fixture` drops the well out of collision
checking.
"""

from __future__ import annotations

import warnings

import pytest

from aind_rutter.build.queries import FIXTURE_EXCLUDED_TAGS, FIXTURE_TAGS
from aind_rutter.config import ConfigModel
from aind_rutter.domain.enums import KNOWN_SCENE_TAGS


def _config(**asset_over):
    asset = {"key": "brain", "kind": "mesh", "src": "b.obj", "loader": "trimesh"}
    asset.update(asset_over)
    return {"version": 1, "assets": [asset]}


def test_asset_tags_land_on_the_generated_node() -> None:
    """The defect: `tags` was written to the spec and read by nobody."""
    cfg = ConfigModel.model_validate(_config(tags=["structure"], scene_tags=["static"]))
    (node,) = cfg.scene.nodes
    assert sorted(node.tags) == ["static", "structure"]


def test_tags_alone_are_enough_to_place_a_node() -> None:
    cfg = ConfigModel.model_validate(_config(tags=["structure"]))
    assert [n.key for n in cfg.scene.nodes] == ["brain"]


def test_the_two_lists_are_unioned_not_overwritten() -> None:
    cfg = ConfigModel.model_validate(
        _config(tags=["structure", "static"], scene_tags=["static", "brain"])
    )
    (node,) = cfg.scene.nodes
    assert sorted(node.tags) == ["brain", "static", "structure"]


def test_a_near_miss_tag_is_warned_about() -> None:
    with pytest.warns(UserWarning, match="did you mean 'fixture'"):
        ConfigModel.model_validate(_config(scene_tags=["fixtrue"]))


def test_an_unfamiliar_tag_is_silent() -> None:
    """A subject's own grouping is not a mistake."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ConfigModel.model_validate(_config(scene_tags=["probe-guard"]))


@pytest.mark.parametrize("tag", sorted(KNOWN_SCENE_TAGS))
def test_a_known_tag_is_silent(tag: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ConfigModel.model_validate(_config(scene_tags=[tag]))


def test_every_tag_the_fixture_rule_uses_is_known() -> None:
    """Otherwise the warning would point people away from a live tag."""
    assert FIXTURE_TAGS <= KNOWN_SCENE_TAGS
    assert FIXTURE_EXCLUDED_TAGS <= KNOWN_SCENE_TAGS


def test_the_retired_options_block_is_refused() -> None:
    with pytest.raises(Exception, match="options"):
        ConfigModel.model_validate({**_config(), "options": {"color_map": "rainbow"}})
