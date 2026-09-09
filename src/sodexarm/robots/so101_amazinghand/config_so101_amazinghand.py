#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("so101_amazinghand_follower")
@dataclass
class SO101AmazingHandFollowerConfig(RobotConfig):
    """13-DOF: SO-101 arm (5x STS3215, IDs 1-5, protocol 0) +
    AmazingHand (8x SCS0009, IDs 1-8, protocol 1) on the SAME RS485 bus.

    Both motor types share one physical serial port (one USB-RS485 adapter).
    """

    # RS485 port for the five SO-101 arm motors.
    port: str = "/dev/ttyso101_amazinghand_r_arm"

    # IDs in joint order (shoulder_pan → wrist_roll). For a shared two-arm bus,
    # override one side with a non-overlapping five-ID tuple.
    arm_motor_ids: tuple[int, ...] = (1, 2, 3, 4, 5)

    # Optional separate RS485 port for hand motors (SCS0009, IDs 1-8).
    # When None (default), hand and arm share the same port (single USB adapter).
    # Set this when the hand is on its own USB adapter. Use a stable udev by-serial symlink
    # (e.g. "/dev/ttyso101_amazinghand_r_hand"), never raw /dev/ttyACMx — ACM numbers shuffle
    # on reboot and a wrong value can land on the arm's device → port collision → judder.
    hand_port: str | None = None

    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    use_degrees: bool = False

    # STS3215 arm position-loop gains (written in configure()).
    # NOTE: raising P does NOT cleanly fix gravity sag — a high P makes the gravity-loaded joints
    # (shoulder_lift ID12 / elbow_flex ID13) hold at HIGH CURRENT, which trips the STS3215
    # over-torque protection → continuous "Overload error" and the joint goes limp (seen at
    # P_lift=64, 2026-06-09). Keep these at/near the factory 32. For sag, re-capture the start
    # pose slightly higher (capture-startup with the arm raised a bit) rather than cranking P.
    arm_p_coefficient: int = 32          # shoulder_pan, wrist_flex, wrist_roll
    arm_p_coefficient_lift: int = 32     # shoulder_lift, elbow_flex — >~40 risks over-torque trip
    arm_d_coefficient: int = 32          # damping

    # Safe position limits for SCS0009 (used as fallback if auto_calibrate_hand=False
    # or if a motor barely moves during the auto-sweep).
    # Note: min==max==0 triggers wheel mode on SCS0009!
    hand_range_min: int = 200
    hand_range_max: int = 800

    # Automatically discover each finger motor's physical range during calibration
    # by sweeping to raw 0 then raw 1023 and recording the stopping positions.
    # Set False to use the fixed hand_range_min / hand_range_max values instead.
    auto_calibrate_hand: bool = True

    # ── AmazingHand (SCS0009) grip / torque (written to the hand motors in configure()) ──
    # The motors' raw torque is ALREADY maxed out on this rig (measured 2026-06-16:
    # P_Coefficient=32, Max_Torque_Limit=1000 on all 8 motors of both hands), so raising those
    # does nothing. The lever that's left for "grips then drops the object partway" is the SCS0009
    # OVERLOAD PROTECTION: once a finger stalls (gripping) past `hand_protection_time`, the servo
    # derates its output to `hand_protective_torque`. Raising both makes the grip hold harder and
    # for longer before the firmware backs off. Both are monotonically grip-positive and still
    # bounded by the motor's separate thermal cutoff (Max_Temperature_Limit), so this can't
    # over-drive the servo — but a finger left stalled for a long time will run hot.
    #   hand_protective_torque: post-protection torque, %% (was 80). 100 = hold full torque.
    #   hand_protection_time:   time tolerated before derating, raw units (was 100). Higher = grips
    #                           longer before backing off; max 254.
    # hand_p_coefficient / hand_max_torque_limit are exposed too, defaulted to the measured
    # (already-max) values so the knobs exist; lower them if the grip is too stiff / buzzes.
    hand_protective_torque: int = 100
    hand_protection_time: int = 200
    hand_p_coefficient: int = 32
    hand_max_torque_limit: int = 1000

    # Hardware compatibility shim for a replaced AmazingHand ID8 servo whose physical direction is
    # reversed relative to policies/datasets collected with the previous motor. When enabled, the
    # logical normalized coordinate for finger4_motor2 stays old-policy-compatible by mapping
    # observation and action values as value -> 100 - value at the robot boundary.
    invert_hand_id8: bool = False

    # Task-local policy compatibility. The PlaceFruit dataset predates the 2026-07-30 expansion
    # of right finger2_motor1 from raw [200, 800] to [200, 1013]. This restores the old policy
    # coordinate at the robot boundary without changing the shared calibration file.
    placefruit_right_compat: bool = False

    # PlaceFruit demonstrations generally commanded the left hand only lightly closed. Values
    # above 1 strengthen closure around the current open/fist anchors while observations are
    # mapped back into the policy coordinate. Keep 1.0 outside that task.
    placefruit_left_grip_gain: float = 1.0

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
