from .config_quest_hts import QuestHTSRightTeleoperatorConfig, QuestHTSTeleoperatorConfig
from .config_quest_hts_bimanual import QuestHTSBimanualTeleoperatorConfig

__all__ = [
    "QuestHTSTeleoperator",
    "QuestHTSTeleoperatorConfig",
    "QuestHTSRightTeleoperator",
    "QuestHTSRightTeleoperatorConfig",
    "QuestHTSBimanualTeleoperator",
    "QuestHTSBimanualTeleoperatorConfig",
]


def __getattr__(name: str):
    if name == "QuestHTSTeleoperator":
        from .quest_hts import QuestHTSTeleoperator

        return QuestHTSTeleoperator
    if name == "QuestHTSRightTeleoperator":
        from .quest_hts import QuestHTSRightTeleoperator

        return QuestHTSRightTeleoperator
    if name == "QuestHTSBimanualTeleoperator":
        from .quest_hts_bimanual import QuestHTSBimanualTeleoperator

        return QuestHTSBimanualTeleoperator
    raise AttributeError(name)
