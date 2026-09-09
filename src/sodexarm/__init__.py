"""SO-DexARM extensions for Hugging Face LeRobot.

Importing this package registers the SO-DexARM robot and Quest teleoperator
configuration classes with LeRobot's ``draccus`` choice registries.  The
implementations remain in this package; no ``lerobot`` modules are shadowed.
"""

from importlib.metadata import PackageNotFoundError, version

from sodexarm.robots.bi_so101_amazinghand.config_bi_so101_amazinghand import (
    BiSO101AmazingHandFollowerConfig,
)
from sodexarm.robots.so101_amazinghand.config_so101_amazinghand import (
    SO101AmazingHandFollowerConfig,
)
from sodexarm.robots.so101_amazinghand_left.config_so101_amazinghand_left import (
    SO101AmazingHandLeftConfig,
)
from sodexarm.robots.so101_amazinghand_right.config_so101_amazinghand_right import (
    SO101AmazingHandRightConfig,
)
from sodexarm.teleoperators.quest_hts.config_quest_hts import (
    QuestHTSRightTeleoperatorConfig,
    QuestHTSTeleoperatorConfig,
)
from sodexarm.teleoperators.quest_hts.config_quest_hts_bimanual import (
    QuestHTSBimanualTeleoperatorConfig,
)

try:
    __version__ = version("sodexarm")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "BiSO101AmazingHandFollowerConfig",
    "QuestHTSBimanualTeleoperatorConfig",
    "QuestHTSRightTeleoperatorConfig",
    "QuestHTSTeleoperatorConfig",
    "SO101AmazingHandFollowerConfig",
    "SO101AmazingHandLeftConfig",
    "SO101AmazingHandRightConfig",
]
