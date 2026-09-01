#!/usr/bin/env python

"""Phase 1 dry-run mapper from HTS right wrist xyz to right SO-101 arm targets.

This script never opens serial ports and never sends motor commands. It only
listens to Hand Tracking Streamer TCP lines, extracts the right wrist pose, and
prints a clipped 5-DoF target for later SO-101 integration.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import socket
from collections.abc import Sequence
from dataclasses import dataclass

from lerobot.teleoperators.quest_hts.hts_protocol import parse_hts_line

RIGHT_ARM_ACTION_KEYS = (
    "right_arm.joint_1.pos",
    "right_arm.joint_2.pos",
    "right_arm.joint_3.pos",
    "right_arm.joint_4.pos",
    "right_arm.joint_5.pos",
)

DEFAULT_ROBOT_HOME_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0)
DEFAULT_JOINT_LIMITS = (
    (-30.0, 30.0),
    (-30.0, 30.0),
    (-30.0, 30.0),
    (-30.0, 30.0),
    (-30.0, 30.0),
)


@dataclass(frozen=True)
class RightWristPose:
    xyz: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float]

    @classmethod
    def from_hts_wrist(cls, wrist: Sequence[float]) -> RightWristPose | None:
        if len(wrist) != 7:
            return None
        values = tuple(float(v) for v in wrist)
        if not all(math.isfinite(v) for v in values):
            return None
        return cls(xyz=values[:3], quat_xyzw=values[3:7])


@dataclass(frozen=True)
class RightArmMappingConfig:
    robot_home_joints: tuple[float, float, float, float, float] = DEFAULT_ROBOT_HOME_JOINTS
    scale: float = 50.0
    joint_limits: tuple[tuple[float, float], ...] = DEFAULT_JOINT_LIMITS
    max_step: float = 2.0
    max_abs_input_m: float = 2.0

    def __post_init__(self) -> None:
        if len(self.robot_home_joints) != 5:
            raise ValueError("robot_home_joints must contain exactly 5 values.")
        if len(self.joint_limits) != 5:
            raise ValueError("joint_limits must contain exactly 5 (min, max) pairs.")
        if self.scale < 0.0:
            raise ValueError("scale must be non-negative.")
        if self.max_step < 0.0:
            raise ValueError("max_step must be non-negative.")
        if self.max_abs_input_m <= 0.0:
            raise ValueError("max_abs_input_m must be positive.")
        for min_value, max_value in self.joint_limits:
            if min_value > max_value:
                raise ValueError("Each joint limit minimum must be <= maximum.")


@dataclass(frozen=True)
class RightArmMappingResult:
    operator_home_xyz: tuple[float, float, float]
    current_xyz: tuple[float, float, float]
    delta_xyz: tuple[float, float, float]
    unclipped_action: tuple[float, float, float, float, float]
    limited_action: tuple[float, float, float, float, float]
    action: dict[str, float]
    joint_limit_clipped: bool
    max_step_clipped: bool

    @property
    def clipped(self) -> bool:
        return self.joint_limit_clipped or self.max_step_clipped


def _is_valid_xyz(xyz: Sequence[float], max_abs_input_m: float) -> bool:
    if len(xyz) != 3:
        return False
    return all(math.isfinite(v) and abs(v) <= max_abs_input_m for v in xyz)


def _clip(value: float, min_value: float, max_value: float) -> float:
    return min(max(value, min_value), max_value)


def _apply_joint_limits(
    values: Sequence[float], joint_limits: Sequence[tuple[float, float]]
) -> tuple[tuple[float, float, float, float, float], bool]:
    clipped_values = []
    clipped = False
    for value, (min_value, max_value) in zip(values, joint_limits, strict=True):
        clipped_value = _clip(value, min_value, max_value)
        clipped_values.append(clipped_value)
        clipped = clipped or clipped_value != value
    return (tuple(clipped_values), clipped)  # type: ignore[return-value]


def _apply_max_step(
    values: Sequence[float], previous: Sequence[float], max_step: float
) -> tuple[tuple[float, float, float, float, float], bool]:
    clipped_values = []
    clipped = False
    for value, previous_value in zip(values, previous, strict=True):
        delta = value - previous_value
        clipped_delta = _clip(delta, -max_step, max_step)
        clipped_values.append(previous_value + clipped_delta)
        clipped = clipped or clipped_delta != delta
    return (tuple(clipped_values), clipped)  # type: ignore[return-value]


def delta_xyz_to_joint_offsets(
    delta_xyz: Sequence[float], scale: float
) -> tuple[float, float, float, float, float]:
    """Temporary Phase 1 mapping. Replace this with IK in a later phase.

    The mapping is deliberately simple and low authority:
      x -> joint 1, y -> joint 2 and joint 4, z -> joint 3, small x/y mix -> joint 5.
    """
    dx, dy, dz = delta_xyz
    return (
        scale * dx,
        scale * dy,
        scale * dz,
        0.5 * scale * dy,
        0.25 * scale * (dx - dy),
    )


class RightArmDryRunMapper:
    """Stateful right-wrist-to-right-arm dry-run mapper with safety clipping."""

    def __init__(self, config: RightArmMappingConfig | None = None) -> None:
        self.config = config or RightArmMappingConfig()
        self.operator_home_xyz: tuple[float, float, float] | None = None
        self.previous_action: tuple[float, float, float, float, float] = self.config.robot_home_joints

    def map_wrist(self, wrist: Sequence[float] | None) -> RightArmMappingResult | None:
        if wrist is None:
            return None

        pose = RightWristPose.from_hts_wrist(wrist)
        if pose is None or not _is_valid_xyz(pose.xyz, self.config.max_abs_input_m):
            return None

        if self.operator_home_xyz is None:
            self.operator_home_xyz = pose.xyz

        delta_xyz = tuple(
            current - home for current, home in zip(pose.xyz, self.operator_home_xyz, strict=True)
        )
        offsets = delta_xyz_to_joint_offsets(delta_xyz, self.config.scale)
        unclipped_action = tuple(
            home + offset for home, offset in zip(self.config.robot_home_joints, offsets, strict=True)
        )
        joint_limited_action, joint_limit_clipped = _apply_joint_limits(
            unclipped_action, self.config.joint_limits
        )
        step_limited_action, max_step_clipped = _apply_max_step(
            joint_limited_action, self.previous_action, self.config.max_step
        )
        self.previous_action = step_limited_action

        return RightArmMappingResult(
            operator_home_xyz=self.operator_home_xyz,
            current_xyz=pose.xyz,
            delta_xyz=delta_xyz,
            unclipped_action=unclipped_action,
            limited_action=step_limited_action,
            action=dict(zip(RIGHT_ARM_ACTION_KEYS, step_limited_action, strict=True)),
            joint_limit_clipped=joint_limit_clipped,
            max_step_clipped=max_step_clipped,
        )


def format_result(result: RightArmMappingResult) -> str:
    return (
        f"operator_home={_format_tuple(result.operator_home_xyz)} "
        f"current_xyz={_format_tuple(result.current_xyz)} "
        f"delta_xyz={_format_tuple(result.delta_xyz)} "
        f"action={{{', '.join(f'{k}: {v:+.3f}' for k, v in result.action.items())}}} "
        f"joint_limit_clipped={result.joint_limit_clipped} "
        f"max_step_clipped={result.max_step_clipped}"
    )


def _format_tuple(values: Sequence[float]) -> str:
    return "(" + ", ".join(f"{v:+.4f}" for v in values) + ")"


def _parse_float_tuple(text: str, expected_len: int, name: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(values) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} must contain {expected_len} comma-separated floats.")
    if not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError(f"{name} must contain finite floats.")
    return values


def _parse_joint_limits(text: str) -> tuple[tuple[float, float], ...]:
    values = _parse_float_tuple(text, 10, "joint-limits")
    limits = tuple((values[idx], values[idx + 1]) for idx in range(0, 10, 2))
    for min_value, max_value in limits:
        if min_value > max_value:
            raise argparse.ArgumentTypeError("joint-limits min values must be <= max values.")
    return limits


def handle_hts_line(line: str, mapper: RightArmDryRunMapper) -> RightArmMappingResult | None:
    parsed = parse_hts_line(line)
    if parsed is None:
        return None

    side, kind, values = parsed
    if side != "right" or kind != "wrist":
        return None
    return mapper.map_wrist(values)


def serve_tcp(host: str, port: int, mapper: RightArmDryRunMapper, print_raw: bool) -> None:
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

    logging.info("Phase 1 dry-run only: no serial ports, motors, or LeRobot Robot will be used.")
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
                        result = handle_hts_line(line, mapper)
                        if result is not None:
                            logging.info(format_result(result))

            logging.info("HTS connection closed")
    finally:
        server.close()
        logging.info("TCP server stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 right SO-101 arm dry-run mapper for HTS.")
    parser.add_argument("--host", default="0.0.0.0", help="TCP bind host.")
    parser.add_argument("--port", type=int, default=8000, help="TCP bind port.")
    parser.add_argument("--scale", type=float, default=50.0, help="Quest xyz to joint target scale.")
    parser.add_argument("--max-step", type=float, default=2.0, help="Max target change per frame per joint.")
    parser.add_argument(
        "--max-abs-input-m",
        type=float,
        default=2.0,
        help="Reject wrist xyz values whose absolute value exceeds this many meters.",
    )
    parser.add_argument(
        "--robot-home-joints",
        type=lambda text: _parse_float_tuple(text, 5, "robot-home-joints"),
        default=DEFAULT_ROBOT_HOME_JOINTS,
        help="Comma-separated 5D robot home target.",
    )
    parser.add_argument(
        "--joint-limits",
        type=_parse_joint_limits,
        default=DEFAULT_JOINT_LIMITS,
        help="Comma-separated min,max pairs for five joints.",
    )
    parser.add_argument("--print-raw", action="store_true", help="Print every raw HTS CSV line.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    mapper = RightArmDryRunMapper(
        RightArmMappingConfig(
            robot_home_joints=args.robot_home_joints,
            scale=args.scale,
            joint_limits=args.joint_limits,
            max_step=args.max_step,
            max_abs_input_m=args.max_abs_input_m,
        )
    )
    serve_tcp(host=args.host, port=args.port, mapper=mapper, print_raw=args.print_raw)


if __name__ == "__main__":
    main()
