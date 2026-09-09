from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig

import sodexarm  # noqa: F401


def test_robot_configs_are_registered() -> None:
    choices = RobotConfig.get_known_choices()
    assert "so101_amazinghand_follower" in choices
    assert "so101_amazinghand_right" in choices
    assert "so101_amazinghand_left" in choices
    assert "bi_so101_amazinghand_follower" in choices


def test_teleoperator_configs_are_registered() -> None:
    choices = TeleoperatorConfig.get_known_choices()
    assert "quest_hts" in choices
    assert "quest_hts_right" in choices
    assert "quest_hts_bimanual" in choices
