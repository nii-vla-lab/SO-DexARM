#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("dual_arm")
@dataclass
class DualArmConfig(RobotConfig):
    """26-DOF dual-arm: right arm + left arm, each with SO-101 (5 DOF) + AmazingHand (8 DOF).

    Each side has five SO-101 arm motors and eight AmazingHand motors. The default hardware
    uses separate arm and hand USB adapters.
    Motor keys are prefixed: r_ (right) and l_ (left).
    """

    # Canonical side convention used throughout SO-DexARM:
    #   r_* / port_r = physical RIGHT, l_* / port_l = physical LEFT.
    port_r: str = "/dev/ttyso101_amazinghand_r_arm"
    port_l: str = "/dev/ttyso101_amazinghand_l_arm"

    # Separate port for each AmazingHand (SCS0009 IDs 1-8) — the hands are on their OWN USB
    # adapters, NOT shared with the arm bus. Use the stable udev by-serial symlinks
    # (scripts/udev/99-so-dexarm.rules.example); never raw /dev/ttyACMx (those shuffle on reboot and
    # can land on the arm's device → port collision → juddery motion).
    hand_port_r: str | None = "/dev/ttyso101_amazinghand_r_hand"
    hand_port_l: str | None = "/dev/ttyso101_amazinghand_l_hand"

    # Standard SO-101 IDs. Separate arm adapters allow both sides to use the official 1-5 IDs.
    # If both arms share one serial bus, override one side with a non-overlapping five-ID tuple.
    arm_ids_r: tuple[int, ...] = (1, 2, 3, 4, 5)
    arm_ids_l: tuple[int, ...] = (1, 2, 3, 4, 5)

    # Which side(s) to operate.  "both" (default) = full 26-DOF dual-arm.
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

    # AmazingHand (SCS0009) grip / torque settings passed to both sub-arms.
    hand_protective_torque: int = 100
    hand_protection_time: int = 200
    hand_p_coefficient: int = 32
    hand_max_torque_limit: int = 1000

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
