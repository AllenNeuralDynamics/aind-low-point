# %%
import logging
from pathlib import Path

from omegaconf import OmegaConf

from aind_low_point.config import ConfigModel

# %%
logging.basicConfig(format="%(message)s", level=logging.DEBUG)

# %%
example_config_path = Path("/home/galen.lynch/786864-config.yml")
app_cfg = OmegaConf.load(example_config_path)
app_cfg_resolved = OmegaConf.to_container(app_cfg, resolve=True)
config = ConfigModel(**app_cfg_resolved)
