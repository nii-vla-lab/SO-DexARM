#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("bi_so101_amazinghand_follower")
@dataclass
class BiSO101AmazingHandFollowerConfig(RobotConfig):
    """26-DOF bilateral: right arm + left arm, each with SO-101 (5 DOF) + AmazingHand (8 DOF).

    Each arm has ALL motors (STS3215 IDs 1-5 + SCS0009 IDs 1-8) on ONE RS485 port.
    Motor keys are prefixed: r_ (right) and l_ (left).
    """

    # Stable udev names. Each public name refers to the corresponding physical side.
    port_r: str = "/dev/ttyso101_amazinghand_r_arm"
    port_l: str = "/dev/ttyso101_amazinghand_l_arm"

    # Separate port for each AmazingHand (SCS0009 IDs 1-8) — the hands are on their OWN USB
    # adapters, NOT shared with the arm bus. Use the stable udev by-serial symlinks
    # (scripts/udev/99-tty-custom.rules); never raw /dev/ttyACMx (those shuffle on reboot and
    # can land on the arm's device → port collision → juddery motion).
    hand_port_r: str | None = "/dev/ttyso101_amazinghand_r_hand"
    hand_port_l: str | None = "/dev/ttyso101_amazinghand_l_hand"

    # Separate adapters allow both arms to use the standard SO-101 IDs. When
    # both arms share a physical bus, configure non-overlapping ID tuples.
    arm_ids_r: tuple[int, ...] = (1, 2, 3, 4, 5)
    arm_ids_l: tuple[int, ...] = (1, 2, 3, 4, 5)

    # Which side(s) to operate.  "both" (default) = full 26-DOF bilateral.
    # "left" / "right" = SINGLE-ARM mode: only that side's arm + hand are built, connected, and
    # exposed (13 DOF, still l_/r_ prefixed).  Use it when only one side is physically connected
    # (e.g. recording a left-only dataset) — the other side's ports/IDs are ignored, and the merged
    # calibration file is NOT overwritten (so the unused side's calibration is preserved).
    # Pair this with the QuestHTS teleoperator's active_sides=… .
    sides: str = "both"

    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    use_degrees: bool = False

    # STS3215 arm position-loop gains (passed to both sub-arms). Keep near factory 32 — high P on
    # the gravity-loaded joints trips the over-torque protection (continuous Overload on
    # shoulder_lift). See SO101AmazingHandFollowerConfig.
    arm_p_coefficient: int = 32
    arm_p_coefficient_lift: int = 32
    arm_d_coefficient: int = 32

    hand_range_min: int = 200
    hand_range_max: int = 800
    auto_calibrate_hand: bool = True

    # AmazingHand (SCS0009) grip / torque — passed to both sub-arms. The raw torque is already
    # maxed on this rig (P=32, Max_Torque=1000), so grip is raised via the SCS0009 overload
    # protection: hold harder (hand_protective_torque) and longer (hand_protection_time) before
    # the firmware derates a stalled (gripping) finger. See SO101AmazingHandFollowerConfig.
    hand_protective_torque: int = 100
    hand_protection_time: int = 200
    hand_p_coefficient: int = 32
    hand_max_torque_limit: int = 1000

    # Per-side compatibility shim for a replaced AmazingHand ID8 servo. Enable only for policies
    # trained before that motor was replaced if the new motor moves finger4_motor2 in the opposite
    # physical direction. The sub-arm maps normalized values as value -> 100 - value.
    right_hand_id8_inverted: bool = False
    left_hand_id8_inverted: bool = False

    # Enabled only by the legacy PlaceFruit inference launcher. These do not alter calibration
    # files and remain disabled for teleop, recording, and policies trained on the current hand.
    right_hand_placefruit_compat: bool = False
    left_hand_placefruit_grip_gain: float = 1.0

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
