#!/usr/bin/env python

"""Phase 4 dual_arm dry-run mapper for SO-101 + AmazingHand.

Consumes HTS Right/Left wrist and landmarks CSV lines and produces one fixed
26D dry-run action. This module never opens serial ports, never sends motor
commands, and never imports LeRobot Robot classes.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import socket
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from lerobot.teleoperators.quest_hts.hts_protocol import parse_hts_line
from lerobot.teleoperators.quest_hts.right_arm_mapping import (
    RIGHT_ARM_ACTION_KEYS,
    RightArmDryRunMapper,
    RightArmMappingConfig,
    RightArmMappingResult,
    _parse_float_tuple,
)
from lerobot.teleoperators.quest_hts.right_hand_mapping import (
    RIGHT_HAND_ACTION_KEYS,
    RightHandDryRunMapper,
    RightHandMappingConfig,
    RightHandMappingResult,
)

RIGHT_SIDE_ACTION_KEYS = RIGHT_ARM_ACTION_KEYS + RIGHT_HAND_ACTION_KEYS
LEFT_ARM_ACTION_KEYS = tuple(key.replace("right_arm.", "left_arm.") for key in RIGHT_ARM_ACTION_KEYS)
LEFT_HAND_ACTION_KEYS = tuple(key.replace("right_hand.", "left_hand.") for key in RIGHT_HAND_ACTION_KEYS)
LEFT_SIDE_ACTION_KEYS = LEFT_ARM_ACTION_KEYS + LEFT_HAND_ACTION_KEYS
DUAL_ARM_ACTION_KEYS = RIGHT_SIDE_ACTION_KEYS + LEFT_SIDE_ACTION_KEYS

LEROBOT_DUAL_ARM_ACTION_KEYS = (
    "r_shoulder_pan.pos",
    "r_shoulder_lift.pos",
    "r_elbow_flex.pos",
    "r_wrist_flex.pos",
    "r_wrist_roll.pos",
    "r_finger1_motor1.pos",
    "r_finger1_motor2.pos",
    "r_finger2_motor1.pos",
    "r_finger2_motor2.pos",
    "r_finger3_motor1.pos",
    "r_finger3_motor2.pos",
    "r_finger4_motor1.pos",
    "r_finger4_motor2.pos",
    "l_shoulder_pan.pos",
    "l_shoulder_lift.pos",
    "l_elbow_flex.pos",
    "l_wrist_flex.pos",
    "l_wrist_roll.pos",
    "l_finger1_motor1.pos",
    "l_finger1_motor2.pos",
    "l_finger2_motor1.pos",
    "l_finger2_motor2.pos",
    "l_finger3_motor1.pos",
    "l_finger3_motor2.pos",
    "l_finger4_motor1.pos",
    "l_finger4_motor2.pos",
)


@dataclass(frozen=True)
class SideConfig:
    arm: RightArmMappingConfig
    hand: RightHandMappingConfig
    axis_signs: tuple[float, float, float]


@dataclass(frozen=True)
class DualArmDryRunConfig:
    right: SideConfig
    left: SideConfig


@dataclass(frozen=True)
class SideUpdateResult:
    arm_result: RightArmMappingResult | None
    hand_result: RightHandMappingResult | None
    wrist_received: bool
    landmarks_received: bool
    arm_initialized: bool
    hand_initialized: bool

    @property
    def clipped(self) -> bool:
        arm_clipped = self.arm_result.clipped if self.arm_result is not None else False
        hand_clipped = self.hand_result.clipped if self.hand_result is not None else False
        return arm_clipped or hand_clipped


@dataclass(frozen=True)
class DualArmUpdateResult:
    action: OrderedDict[str, float]
    right: SideUpdateResult | None
    left: SideUpdateResult | None

    @property
    def clipped(self) -> bool:
        right_clipped = self.right.clipped if self.right is not None else False
        left_clipped = self.left.clipped if self.left is not None else False
        return right_clipped or left_clipped


class SideDryRunMapper:
    """One side of arm+hand mapping, using Phase 1/2 mappers internally."""

    def __init__(
        self,
        side: str,
        config: SideConfig,
        arm_keys: Sequence[str],
        hand_keys: Sequence[str],
    ) -> None:
        self.side = side
        self.config = config
        self.arm_mapper = RightArmDryRunMapper(config.arm)
        self.hand_mapper = RightHandDryRunMapper(config.hand)
        self.arm_keys = tuple(arm_keys)
        self.hand_keys = tuple(hand_keys)
        self.last_arm_action: OrderedDict[str, float] = OrderedDict(
            zip(self.arm_keys, config.arm.robot_home_joints, strict=True)
        )
        self.last_hand_action: OrderedDict[str, float] = OrderedDict(
            zip(self.hand_keys, config.hand.open_targets, strict=True)
        )
        self.arm_initialized = False
        self.hand_initialized = False

    def combined_action(self) -> OrderedDict[str, float]:
        action: OrderedDict[str, float] = OrderedDict()
        action.update(self.last_arm_action)
        action.update(self.last_hand_action)
        return action

    def update(self, kind: str, values: Sequence[float]) -> SideUpdateResult | None:
        arm_result = None
        hand_result = None
        wrist_received = False
        landmarks_received = False

        if kind == "wrist":
            wrist_received = True
            signed_values = self._apply_axis_signs_to_wrist(values)
            arm_result = self.arm_mapper.map_wrist(signed_values)
            if arm_result is None:
                logging.warning(
                    "Rejected %s wrist: invalid shape, non-finite value, or too-large input", self.side
                )
                return None
            self.last_arm_action = OrderedDict(zip(self.arm_keys, arm_result.action.values(), strict=True))
            self.arm_initialized = True
        elif kind == "landmarks":
            landmarks_received = True
            hand_result = self.hand_mapper.map_landmarks(values)
            if hand_result is None:
                logging.warning(
                    "Rejected %s landmarks: invalid shape, non-finite value, or too-large input", self.side
                )
                return None
            self.last_hand_action = OrderedDict(zip(self.hand_keys, hand_result.action.values(), strict=True))
            self.hand_initialized = True
        else:
            return None

        return SideUpdateResult(
            arm_result=arm_result,
            hand_result=hand_result,
            wrist_received=wrist_received,
            landmarks_received=landmarks_received,
            arm_initialized=self.arm_initialized,
            hand_initialized=self.hand_initialized,
        )

    def _apply_axis_signs_to_wrist(self, values: Sequence[float]) -> tuple[float, ...]:
        if len(values) != 7:
            return tuple(values)
        sx, sy, sz = self.config.axis_signs
        return (
            values[0] * sx,
            values[1] * sy,
            values[2] * sz,
            values[3],
            values[4],
            values[5],
            values[6],
        )


class DualArmDryRunMapper:
    """Right+left dry-run mapper with fixed-order 26D output."""

    def __init__(self, config: DualArmDryRunConfig | None = None) -> None:
        self.config = config or default_dual_arm_config()
        self.right = SideDryRunMapper(
            "right", self.config.right, RIGHT_ARM_ACTION_KEYS, RIGHT_HAND_ACTION_KEYS
        )
        self.left = SideDryRunMapper("left", self.config.left, LEFT_ARM_ACTION_KEYS, LEFT_HAND_ACTION_KEYS)

    def combined_action(self) -> OrderedDict[str, float]:
        action: OrderedDict[str, float] = OrderedDict()
        action.update(self.right.combined_action())
        action.update(self.left.combined_action())
        return action

    def action_values(self) -> tuple[float, ...]:
        return tuple(self.combined_action().values())

    def update_from_parsed(self, side: str, kind: str, values: Sequence[float]) -> DualArmUpdateResult | None:
        if side == "right":
            right_result = self.right.update(kind, values)
            if right_result is None:
                return None
            return DualArmUpdateResult(action=self.combined_action(), right=right_result, left=None)
        if side == "left":
            left_result = self.left.update(kind, values)
            if left_result is None:
                return None
            return DualArmUpdateResult(action=self.combined_action(), right=None, left=left_result)
        return None

    def update_from_line(self, line: str) -> DualArmUpdateResult | None:
        parsed = parse_hts_line(line)
        if parsed is None:
            logging.debug("Ignored malformed or unsupported HTS line")
            return None
        return self.update_from_parsed(*parsed)


def default_dual_arm_config() -> DualArmDryRunConfig:
    return DualArmDryRunConfig(
        right=SideConfig(
            arm=RightArmMappingConfig(),
            hand=RightHandMappingConfig(),
            axis_signs=(1.0, 1.0, 1.0),
        ),
        left=SideConfig(
            arm=RightArmMappingConfig(),
            hand=RightHandMappingConfig(),
            axis_signs=(-1.0, 1.0, 1.0),
        ),
    )


def provisional_to_lerobot_action(
    action: OrderedDict[str, float] | dict[str, float],
) -> OrderedDict[str, float]:
    values = [action[key] for key in DUAL_ARM_ACTION_KEYS]
    return OrderedDict(zip(LEROBOT_DUAL_ARM_ACTION_KEYS, values, strict=True))


def format_result(result: DualArmUpdateResult) -> str:
    parts = []
    if result.right is not None:
        parts.append(_format_side("right", result.right))
    if result.left is not None:
        parts.append(_format_side("left", result.left))
    parts.append(f"combined26={_format_action(result.action)} clipped={result.clipped}")
    return " | ".join(parts)


def _format_side(side: str, result: SideUpdateResult) -> str:
    parts = [
        f"{side} received wrist={result.wrist_received} landmarks={result.landmarks_received}",
        f"initialized arm={result.arm_initialized} hand={result.hand_initialized}",
    ]
    if result.arm_result is not None:
        arm_action = _display_action_for_side(side, result.arm_result.action)
        parts.append(
            f"wrist_xyz={_format_tuple(result.arm_result.current_xyz)} "
            f"delta_xyz={_format_tuple(result.arm_result.delta_xyz)} "
            f"arm5={_format_action(arm_action)} clipped={result.arm_result.clipped}"
        )
    if result.hand_result is not None:
        curls = result.hand_result.curls
        hand_action = _display_action_for_side(side, result.hand_result.action)
        feature_text = _format_finger_features(result.hand_result.finger_features)
        parts.append(
            f"curls=thumb:{curls.thumb:.3f} index:{curls.index:.3f} middle:{curls.middle:.3f} "
            f"ring:{curls.ring:.3f} little:{curls.little:.3f} "
            f"finger_features={feature_text} "
            f"hand8={_format_action(hand_action)} clipped={result.hand_result.clipped}"
        )
    return " ".join(parts)


def _display_action_for_side(side: str, action: dict[str, float]) -> OrderedDict[str, float]:
    if side == "right":
        return OrderedDict(action.items())
    if side == "left":
        return OrderedDict((key.replace("right_", "left_", 1), value) for key, value in action.items())
    return OrderedDict(action.items())


def _format_tuple(values: Sequence[float]) -> str:
    return "(" + ", ".join(f"{v:+.4f}" for v in values) + ")"


def _format_action(action: dict[str, float] | OrderedDict[str, float]) -> str:
    return "{" + ", ".join(f"{key}: {value:+.3f}" for key, value in action.items()) + "}"


def _format_finger_features(features: dict) -> str:
    parts = []
    for name, feature in features.items():
        joints = ",".join(f"{value:.4f}" for value in feature.joint_distances)
        parts.append(
            f"{name}(tip={feature.tip_distance:.4f},joints=({joints}),score={feature.open_close_score:.3f})"
        )
    return "[" + " ".join(parts) + "]"


def _parse_axis_signs(text: str) -> tuple[float, float, float]:
    values = _parse_float_tuple(text, 3, "axis-signs")
    return (values[0], values[1], values[2])


def _build_config_from_args(args: argparse.Namespace) -> DualArmDryRunConfig:
    max_abs_input_m = args.max_abs_input_m
    if not math.isfinite(max_abs_input_m) or max_abs_input_m <= 0.0:
        raise ValueError("--max-abs-input-m must be a positive finite float.")
    return DualArmDryRunConfig(
        right=SideConfig(
            arm=RightArmMappingConfig(
                robot_home_joints=args.right_robot_home_joints,
                scale=args.scale,
                joint_limits=args.right_joint_limits,
                max_step=args.arm_max_step,
                max_abs_input_m=max_abs_input_m,
            ),
            hand=RightHandMappingConfig(
                open_targets=args.right_hand_open_target,
                closed_targets=args.right_hand_closed_target,
                motor_limits=args.right_motor_limits,
                max_step=args.hand_max_step,
                max_abs_input_m=max_abs_input_m,
            ),
            axis_signs=args.right_axis_signs,
        ),
        left=SideConfig(
            arm=RightArmMappingConfig(
                robot_home_joints=args.left_robot_home_joints,
                scale=args.scale,
                joint_limits=args.left_joint_limits,
                max_step=args.arm_max_step,
                max_abs_input_m=max_abs_input_m,
            ),
            hand=RightHandMappingConfig(
                open_targets=args.left_hand_open_target,
                closed_targets=args.left_hand_closed_target,
                motor_limits=args.left_motor_limits,
                max_step=args.hand_max_step,
                max_abs_input_m=max_abs_input_m,
            ),
            axis_signs=args.left_axis_signs,
        ),
    )


def serve_tcp(host: str, port: int, mapper: DualArmDryRunMapper, print_raw: bool) -> None:
    running = True

    def _stop(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    server.settimeout(0.5)

    logging.info("Phase 4 dry-run only: no serial ports, motors, or LeRobot Robot will be used.")
    logging.info("TCP server listening on %s:%d", host, port)
    logging.info("For wired Quest TCP, run: adb reverse tcp:%d tcp:%d", port, port)

    try:
        while running:
            try:
                conn, addr = server.accept()
            except TimeoutError:
                continue

            logging.info("Accepted HTS connection from %s", addr)
            with conn:
                conn.settimeout(0.5)
                buffer = ""
                while running:
                    data: bytes | None
                    try:
                        data = conn.recv(8192)
                    except TimeoutError:
                        data = None
                    except OSError as exc:
                        logging.warning("Connection error: %s", exc)
                        break
                    if data is None:
                        continue
                    if not data:
                        break
                    try:
                        buffer += data.decode("utf-8")
                    except UnicodeDecodeError:
                        logging.warning("Rejected non-UTF-8 packet")
                        continue
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        if print_raw:
                            logging.info("raw: %s", line)
                        result = mapper.update_from_line(line)
                        if result is not None:
                            logging.info(format_result(result))
            logging.info("HTS connection closed")
    finally:
        server.close()
        logging.info("TCP server stopped")
