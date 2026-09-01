#!/usr/bin/env python

import warnings
from dataclasses import dataclass
from pathlib import Path

from lerobot.teleoperators.config import TeleoperatorConfig

DEFAULT_ROBOT_HOME_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0)
DEFAULT_HAND_OPEN_TARGET = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
DEFAULT_HAND_CLOSED_TARGET = (100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
DEFAULT_JOINT_LIMITS = (
    (-30.0, 30.0),
    (-30.0, 30.0),
    (-30.0, 30.0),
    (-30.0, 30.0),
    (-30.0, 30.0),
)
DEFAULT_MOTOR_LIMITS = tuple((0.0, 100.0) for _ in range(8))


@TeleoperatorConfig.register_subclass("quest_hts")
@dataclass
class QuestHTSTeleoperatorConfig(TeleoperatorConfig):
    """Quest Hand Tracking Streamer teleoperator config.

    Dry-run/sensing only: this teleoperator returns action dictionaries and never
    connects to robot serial ports or motors.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    transport: str = "tcp_server"

    scale: float = 50.0
    arm_max_step: float = 2.0
    hand_max_step: float = 5.0
    max_abs_input_m: float = 2.0

    right_axis_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)
    left_axis_signs: tuple[float, float, float] = (-1.0, 1.0, 1.0)

    right_robot_home_joints: tuple[float, float, float, float, float] = DEFAULT_ROBOT_HOME_JOINTS
    left_robot_home_joints: tuple[float, float, float, float, float] = DEFAULT_ROBOT_HOME_JOINTS

    right_hand_open_target: tuple[float, float, float, float, float, float, float, float] = (
        DEFAULT_HAND_OPEN_TARGET
    )
    left_hand_open_target: tuple[float, float, float, float, float, float, float, float] = (
        DEFAULT_HAND_OPEN_TARGET
    )
    right_hand_closed_target: tuple[float, float, float, float, float, float, float, float] = (
        DEFAULT_HAND_CLOSED_TARGET
    )
    left_hand_closed_target: tuple[float, float, float, float, float, float, float, float] = (
        DEFAULT_HAND_CLOSED_TARGET
    )

    joint_limits: tuple[tuple[float, float], ...] = DEFAULT_JOINT_LIMITS
    motor_limits: tuple[tuple[float, float], ...] = DEFAULT_MOTOR_LIMITS

    print_debug: bool = False
    start_receiver: bool = True

    def __post_init__(self) -> None:
        if self.transport != "tcp_server":
            raise ValueError("QuestHTS currently supports only transport='tcp_server'.")
        if len(self.right_axis_signs) != 3 or len(self.left_axis_signs) != 3:
            raise ValueError("axis_signs must contain exactly 3 values.")
        if len(self.right_robot_home_joints) != 5 or len(self.left_robot_home_joints) != 5:
            raise ValueError("robot_home_joints must contain exactly 5 values.")
        for target_name in (
            "right_hand_open_target",
            "left_hand_open_target",
            "right_hand_closed_target",
            "left_hand_closed_target",
        ):
            if len(getattr(self, target_name)) != 8:
                raise ValueError(f"{target_name} must contain exactly 8 values.")
        if len(self.joint_limits) != 5:
            raise ValueError("joint_limits must contain exactly 5 (min, max) pairs.")
        if len(self.motor_limits) != 8:
            raise ValueError("motor_limits must contain exactly 8 (min, max) pairs.")


@TeleoperatorConfig.register_subclass("quest_hts_right")
@dataclass
class QuestHTSRightTeleoperatorConfig(TeleoperatorConfig):
    """Right-side QuestHTS teleoperator for calibrated SO-101 + AmazingHand.

    Output actions are LeRobot calibrated joint actions, not raw ticks. The
    production interface uses the native QuestHTS ``Right`` stream for the
    physical right hand and emits canonical ``r_*.pos`` action keys.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    transport: str = "tcp_server"
    physical_hand: str = "right"
    input_assignment_mode: str = "native_right"
    binding_mode: str = "saved_map"
    side_mapping_file: Path = Path("scripts/configs/quest_hts_physical_side_mapping.yaml")
    bound_input_label: str | None = None
    print_hand_binding: bool = False
    print_debug_transport_labels: bool = False
    hand_binding_timeout_s: float = 5.0
    min_binding_frames: int = 3
    min_binding_movement_m: float = 0.005
    require_live_side_confirmation: bool = True
    side_confirmation_timeout_s: float = 5.0
    side_confirmation_min_frames: int = 3
    input_hand: str | None = None
    stale_policy: str = "wait-reconnect"
    stale_timeout_s: float = 1.0
    rebaseline_after_reconnect: bool = True
    mode: str = "arm-and-hand"
    start_receiver: bool = True

    arm_control_mode: str = "ik_ee"
    manual_baseline: bool = True
    start_pose_file: Path | None = None
    constraints_file: Path | None = None
    reacquire_human_baseline: bool = True
    allow_start_pose_warning: bool = True
    start_pose_warning_threshold: float = 8.0
    start_pose_policy: str = "abort"
    start_pose_max_error: float = 5.0
    max_hand_joint_step: float = 2.0
    max_hand_joint_delta_from_start: float = 10.0
    tcp_config_file: Path = Path(".cache/so_dexarm/quest_hts_right_amazinghand_tcp.yaml")
    tcp_offset_x: float | None = None
    tcp_offset_y: float | None = None
    tcp_offset_z: float | None = None
    tcp_offset_roll: float | None = None
    tcp_offset_pitch: float | None = None
    tcp_offset_yaw: float | None = None
    print_current_tcp_pose: bool = False
    save_current_tcp_as_home: bool = False
    home_tcp_pose_file: Path = Path(".cache/so_dexarm/quest_hts_right_so101_amazinghand_home_tcp.yaml")
    ik_failure_policy: str = "hold-last"
    ik_solution_policy: str = "closest-to-last-action"
    max_tcp_translation_m: float = 0.20
    max_tcp_rotation_rad: float = 1.2
    workspace_limits_file: Path | None = None
    planar_workspace_policy: str = "project"
    max_controlled_joint_step: float = 2.0
    max_controlled_joint_delta_from_start: float = 15.0
    print_ik_debug: bool = False
    tcp_translation_scale: float = 1.0
    tcp_rotation_scale: float = 1.0
    tcp_axis_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)
    robot_calibration_file: Path = Path(".cache/calibration/robots/so101_amazinghand_right/right.json")
    arm_action_units: str = "range_m100_100"

    baseline_samples: int = 20
    hold_last_target_s: float = 0.25
    arm_controller_preset: str = "right_teleop_v1"
    arm_pos_gain_x: float = 600.0
    arm_pos_gain_y: float = 450.0
    arm_pos_gain_z: float = 600.0
    arm_rot_gain_roll: float = 45.0
    arm_rot_gain_pitch: float = 45.0
    arm_rot_gain_yaw: float = 35.0
    arm_deadzone_pos: float = 0.004
    arm_deadzone_rot: float = 0.025
    arm_smoothing_alpha: float = 0.45
    max_arm_step: float = 5.0
    max_arm_cumulative: float = 50.0
    arm_joint_limits: tuple[tuple[float, float], ...] = tuple((-100.0, 100.0) for _ in range(5))

    use_hand_calibration: bool = True
    right_hand_calibration_file: Path = Path(".cache/so_dexarm/quest_hts_right_hand_calibration.yaml")
    hand_flex_gain: float = 1.0
    hand_flex_saturation_threshold: float = 1.0
    require_calibration: bool = True

    print_debug: bool = False

    def __post_init__(self) -> None:
        if self.transport != "tcp_server":
            raise ValueError("QuestHTSRight currently supports only transport='tcp_server'.")
        if self.physical_hand not in {"right", "left", "auto"}:
            raise ValueError("--teleop.physical-hand must be right, left, or auto.")
        if self.input_assignment_mode not in {
            "native_right",
            "physical_side_mapping",
            "single_visible_right",
        }:
            raise ValueError(
                "--teleop.input-assignment-mode must be native_right, physical_side_mapping, or single_visible_right."
            )
        if self.input_assignment_mode == "native_right" and self.physical_hand != "right":
            raise ValueError(
                "--teleop.input-assignment-mode=native_right requires --teleop.physical-hand=right."
            )
        if self.input_assignment_mode == "single_visible_right" and self.physical_hand != "right":
            raise ValueError(
                "--teleop.input-assignment-mode=single_visible_right requires --teleop.physical-hand=right."
            )
        if self.input_assignment_mode == "native_right":
            self.binding_mode = "explicit"
            self.require_live_side_confirmation = False
        if self.binding_mode not in {"saved_map", "auto", "explicit"}:
            raise ValueError("--teleop.binding-mode must be saved_map, auto, or explicit.")
        if self.bound_input_label is not None and self.bound_input_label not in {"left", "right"}:
            raise ValueError("--teleop.bound-input-label must be left or right.")
        if self.binding_mode != "explicit" and self.bound_input_label is not None:
            raise ValueError("--teleop.bound-input-label is only valid with --teleop.binding-mode=explicit.")
        if self.input_hand is not None and self.input_hand not in {"left", "right"}:
            raise ValueError("--teleop.input-hand must be left or right.")
        if self.input_hand is not None:
            warnings.warn(
                "--teleop.input_hand is a deprecated raw QuestHTS label override. "
                "Right-side production commands should use --teleop.input_assignment_mode=native_right; "
                "for fixed raw-label debugging use --teleop.binding_mode=explicit --teleop.bound_input_label=<left|right>.",
                UserWarning,
                stacklevel=2,
            )
        if self.input_assignment_mode == "native_right" and self.input_hand not in {None, "right"}:
            raise ValueError(
                "--teleop.input-assignment-mode=native_right only supports --teleop.input-hand=right."
            )
        if (
            self.input_assignment_mode not in {"native_right", "single_visible_right"}
            and self.binding_mode == "explicit"
            and self.bound_input_label is None
            and self.input_hand is None
        ):
            raise ValueError("--teleop.binding-mode=explicit requires --teleop.bound-input-label.")
        if self.stale_policy not in {"wait-reconnect", "abort-run", "abort-episode"}:
            raise ValueError("--teleop.stale-policy must be wait-reconnect, abort-run, or abort-episode.")
        if self.stale_timeout_s <= 0:
            raise ValueError("--teleop.stale-timeout-s must be > 0.")
        if self.hand_binding_timeout_s <= 0:
            raise ValueError("--teleop.hand-binding-timeout-s must be > 0.")
        if self.min_binding_frames <= 0:
            raise ValueError("--teleop.min-binding-frames must be > 0.")
        if self.min_binding_movement_m < 0:
            raise ValueError("--teleop.min-binding-movement-m must be >= 0.")
        if self.side_confirmation_timeout_s <= 0:
            raise ValueError("--teleop.side-confirmation-timeout-s must be > 0.")
        if self.side_confirmation_min_frames <= 0:
            raise ValueError("--teleop.side-confirmation-min-frames must be > 0.")
        if self.mode not in {"arm-only", "hand-only", "arm-and-hand"}:
            raise ValueError("--teleop.mode must be arm-only, hand-only, or arm-and-hand.")
        if self.arm_control_mode not in {"ik_ee", "joint_delta", "constrained_planar_ik"}:
            raise ValueError(
                "--teleop.arm-control-mode must be ik_ee, joint_delta, or constrained_planar_ik."
            )
        if self.start_pose_warning_threshold < 0:
            raise ValueError("--teleop.start-pose-warning-threshold must be >= 0.")
        if self.start_pose_policy not in {"abort", "warn", "ignore"}:
            raise ValueError("--teleop.start-pose-policy must be abort, warn, or ignore.")
        if self.start_pose_max_error <= 0:
            raise ValueError("--teleop.start-pose-max-error must be > 0.")
        if self.max_hand_joint_step <= 0:
            raise ValueError("--teleop.max-hand-joint-step must be > 0.")
        if self.max_hand_joint_delta_from_start <= 0:
            raise ValueError("--teleop.max-hand-joint-delta-from-start must be > 0.")
        if self.ik_failure_policy not in {"hold-last", "skip", "abort"}:
            raise ValueError("--teleop.ik-failure-policy must be hold-last, skip, or abort.")
        if self.ik_solution_policy not in {"closest-to-last-action"}:
            raise ValueError("--teleop.ik-solution-policy must be closest-to-last-action.")
        if self.max_controlled_joint_step <= 0:
            raise ValueError("--teleop.max-controlled-joint-step must be > 0.")
        if self.max_controlled_joint_delta_from_start <= 0:
            raise ValueError("--teleop.max-controlled-joint-delta-from-start must be > 0.")
        if self.planar_workspace_policy not in {"project", "fail"}:
            raise ValueError("--teleop.planar-workspace-policy must be project or fail.")
        if len(self.tcp_axis_signs) != 3:
            raise ValueError("--teleop.tcp-axis-signs must contain exactly 3 values.")
        if self.arm_action_units not in {"range_m100_100", "degrees"}:
            raise ValueError("--teleop.arm-action-units must be range_m100_100 or degrees.")
        if self.max_tcp_translation_m <= 0:
            raise ValueError("--teleop.max-tcp-translation-m must be > 0.")
        if self.max_tcp_rotation_rad <= 0:
            raise ValueError("--teleop.max-tcp-rotation-rad must be > 0.")
        if self.tcp_translation_scale <= 0:
            raise ValueError("--teleop.tcp-translation-scale must be > 0.")
        if self.tcp_rotation_scale <= 0:
            raise ValueError("--teleop.tcp-rotation-scale must be > 0.")
        if self.baseline_samples <= 0:
            raise ValueError("--teleop.baseline-samples must be > 0.")
        if self.hold_last_target_s < 0:
            raise ValueError("--teleop.hold-last-target-s must be >= 0.")
        if len(self.arm_joint_limits) != 5:
            raise ValueError("--teleop.arm-joint-limits must contain exactly five (min, max) pairs.")
        if not 0.0 <= self.arm_smoothing_alpha <= 1.0:
            raise ValueError("--teleop.arm-smoothing-alpha must be between 0 and 1.")
        if self.max_arm_step <= 0 or self.max_arm_cumulative <= 0:
            raise ValueError("--teleop max arm limits must be > 0.")
        if self.hand_flex_gain <= 0:
            raise ValueError("--teleop.hand-flex-gain must be > 0.")
        if not 0.0 <= self.hand_flex_saturation_threshold <= 1.0:
            raise ValueError("--teleop.hand-flex-saturation-threshold must be in [0, 1].")
