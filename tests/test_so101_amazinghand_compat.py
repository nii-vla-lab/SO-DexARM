from types import SimpleNamespace

import pytest

from sodexarm.robots.so101_amazinghand.so101_amazinghand_follower import (
    _PLACEFRUIT_RIGHT_ID3_ACTION_SCALE,
    SO101AmazingHandFollower,
)


def make_robot(**overrides) -> SO101AmazingHandFollower:
    robot = object.__new__(SO101AmazingHandFollower)
    config = {
        "invert_hand_id8": False,
        "placefruit_right_compat": False,
        "placefruit_left_grip_gain": 1.0,
    }
    config.update(overrides)
    robot.config = SimpleNamespace(**config)
    return robot


def test_placefruit_right_id3_round_trip_restores_recording_range() -> None:
    robot = make_robot(placefruit_right_compat=True)
    physical = robot._physical_hand_goal("finger2_motor1", 75.0)
    assert physical == pytest.approx(75.0 * _PLACEFRUIT_RIGHT_ID3_ACTION_SCALE)
    assert robot._logical_hand_value("finger2_motor1", physical) == pytest.approx(75.0)


def test_id8_inversion_composes_with_placefruit_compatibility() -> None:
    robot = make_robot(invert_hand_id8=True, placefruit_right_compat=True)
    assert robot._physical_hand_goal("finger4_motor2", 20.0) == pytest.approx(80.0)
    assert robot._logical_hand_value("finger4_motor2", 80.0) == pytest.approx(20.0)
