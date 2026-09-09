#!/usr/bin/env python

from dataclasses import dataclass
from pathlib import Path

from lerobot.robots.config import RobotConfig

from sodexarm.robots.so101_amazinghand.config_so101_amazinghand import SO101AmazingHandFollowerConfig


@RobotConfig.register_subclass("so101_amazinghand_left")
@dataclass
class SO101AmazingHandLeftConfig(SO101AmazingHandFollowerConfig):
    """Left-side SO-101 + AmazingHand LeRobot robot config.

    Mirror of so101_amazinghand_right but uses the physical left-side ports:
      arm:  /dev/ttyso101_amazinghand_r_arm  (r_arm symlink = physical left arm)
      hand: /dev/ttyso101_amazinghand_l_hand
    """

    port: str = "/dev/ttyso101_amazinghand_r_arm"   # r_arm symlink = physical left arm
    hand_port: str | None = "/dev/ttyso101_amazinghand_l_hand"
    mode: str = "arm-and-hand"
    require_calibration: bool = True
    hand_calibration_source: str = "quest_yaml"
    hand_calibration_file: Path = Path("configs/quest_hts_left_hand_calibration.yaml")
    hand_error_policy: str = "disable-hand"
    arm_error_policy: str = "abort"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode not in {"arm-only", "hand-only", "arm-and-hand"}:
            raise ValueError("--robot.mode must be arm-only, hand-only, or arm-and-hand.")
        if self.hand_calibration_source not in {"quest_yaml", "lerobot"}:
            raise ValueError("--robot.hand-calibration-source must be quest_yaml or lerobot.")
        if self.hand_error_policy not in {"abort", "continue", "disable-hand"}:
            raise ValueError("--robot.hand-error-policy must be abort, continue, or disable-hand.")
        if self.arm_error_policy not in {"abort", "continue", "disable-arm"}:
            raise ValueError("--robot.arm-error-policy must be abort, continue, or disable-arm.")
