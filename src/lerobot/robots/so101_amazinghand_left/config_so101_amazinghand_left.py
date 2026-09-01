#!/usr/bin/env python

from dataclasses import dataclass
from pathlib import Path

from lerobot.robots.config import RobotConfig
from lerobot.robots.so101_amazinghand_right.config_so101_amazinghand_right import (
    SO101AmazingHandRightConfig,
)


@RobotConfig.register_subclass("so101_amazinghand_left")
@dataclass
class SO101AmazingHandLeftConfig(SO101AmazingHandRightConfig):
    """Left-side SO-101 + AmazingHand LeRobot robot config.

    Mirror of so101_amazinghand_right but uses the physical left-side ports:
      arm:  /dev/ttyso101_amazinghand_l_arm
      hand: /dev/ttyso101_amazinghand_l_hand
    """

    port: str = "/dev/ttyso101_amazinghand_l_arm"
    hand_port: str | None = "/dev/ttyso101_amazinghand_l_hand"
    mode: str = "arm-and-hand"
    require_calibration: bool = True
    hand_calibration_source: str = "quest_yaml"
    hand_calibration_file: Path = Path(".cache/so_dexarm/quest_hts_left_hand_calibration.yaml")
    hand_error_policy: str = "disable-hand"
    arm_error_policy: str = "abort"
    side: str = "left"
