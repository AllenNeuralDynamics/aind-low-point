"""Scene models: which nodes exist and how each is placed."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, PrivateAttr

from aind_rutter.config.models_common import TransformRefModel

# -----------------------------------------------------------------------------
# Scene (WHERE: instances and bindings)
# -----------------------------------------------------------------------------


class SceneNodeModel(BaseModel):
    model_config = {"extra": "forbid"}

    key: str
    asset: str = Field(description="Key of an AssetSpec in catalog")
    tags: list[str] = Field(default_factory=list)

    # Reference a named transform (from ConfigModel.transforms) or None for identity
    transform: Optional[TransformRefModel] = None

    # Optional domain binding for pose (probes): ties node to domain.probes[name]
    pose_source_probe: Optional[str] = Field(
        default=None,
        description="If set, renderer takes pose from plan.probes[pose_source_probe].",
    )


class SceneModel(BaseModel):
    model_config = {"extra": "forbid"}

    nodes: list[SceneNodeModel] = Field(default_factory=list)
    _explicit_node_keys: set[str] = PrivateAttr(default_factory=set)
