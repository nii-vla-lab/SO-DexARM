from .config_quest_hts import QuestHTSRightTeleoperatorConfig, QuestHTSTeleoperatorConfig
from .config_quest_hts_dual_arm import QuestHTSDualArmTeleoperatorConfig

__all__ = [
    "QuestHTSTeleoperator",
    "QuestHTSTeleoperatorConfig",
    "QuestHTSRightTeleoperator",
    "QuestHTSRightTeleoperatorConfig",
    "QuestHTSDualArmTeleoperator",
    "QuestHTSDualArmTeleoperatorConfig",
]


def __getattr__(name: str):
    if name == "QuestHTSTeleoperator":
        from .quest_hts import QuestHTSTeleoperator

        return QuestHTSTeleoperator
    if name == "QuestHTSRightTeleoperator":
        from .quest_hts import QuestHTSRightTeleoperator

        return QuestHTSRightTeleoperator
    if name == "QuestHTSDualArmTeleoperator":
        from .quest_hts_dual_arm import QuestHTSDualArmTeleoperator

        return QuestHTSDualArmTeleoperator
    raise AttributeError(name)
