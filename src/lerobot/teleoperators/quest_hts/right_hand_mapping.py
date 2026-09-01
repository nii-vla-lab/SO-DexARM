#!/usr/bin/env python

"""Phase 2 dry-run mapper from HTS right hand landmarks to AmazingHand targets.

This script never opens serial ports and never sends motor commands. It only
listens to Hand Tracking Streamer TCP lines, extracts right-hand landmarks, and
prints clipped 8-motor targets for later AmazingHand integration.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import socket
from collections.abc import Sequence
from dataclasses import dataclass

from lerobot.teleoperators.quest_hts.hts_protocol import HandSnapshot, parse_hts_line

RIGHT_HAND_ACTION_KEYS = (
    "right_hand.motor_1.pos",
    "right_hand.motor_2.pos",
    "right_hand.motor_3.pos",
    "right_hand.motor_4.pos",
    "right_hand.motor_5.pos",
    "right_hand.motor_6.pos",
    "right_hand.motor_7.pos",
    "right_hand.motor_8.pos",
)

FINGER_INDICES = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}

DEFAULT_OPEN_TARGETS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
DEFAULT_CLOSED_TARGETS = (100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
DEFAULT_MOTOR_LIMITS = tuple((0.0, 100.0) for _ in range(8))


@dataclass(frozen=True)
class FingerCurls:
    thumb: float
    index: float
    middle: float
    ring: float
    little: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.thumb, self.index, self.middle, self.ring, self.little)


@dataclass(frozen=True)
class FingerFeature:
    tip_distance: float
    joint_distances: tuple[float, float, float]
    open_close_score: float


@dataclass(frozen=True)
class RightHandMappingConfig:
    open_targets: tuple[float, float, float, float, float, float, float, float] = DEFAULT_OPEN_TARGETS
    closed_targets: tuple[float, float, float, float, float, float, float, float] = DEFAULT_CLOSED_TARGETS
    motor_limits: tuple[tuple[float, float], ...] = DEFAULT_MOTOR_LIMITS
    max_step: float = 5.0
    max_abs_input_m: float = 2.0
    open_ratio: float = 0.95
    closed_ratio: float = 0.45

    def __post_init__(self) -> None:
        if len(self.open_targets) != 8:
            raise ValueError("open_targets must contain exactly 8 values.")
        if len(self.closed_targets) != 8:
            raise ValueError("closed_targets must contain exactly 8 values.")
        if len(self.motor_limits) != 8:
            raise ValueError("motor_limits must contain exactly 8 (min, max) pairs.")
        if self.max_step < 0.0:
            raise ValueError("max_step must be non-negative.")
        if self.max_abs_input_m <= 0.0:
            raise ValueError("max_abs_input_m must be positive.")
        if self.open_ratio <= self.closed_ratio:
            raise ValueError("open_ratio must be greater than closed_ratio.")
        for min_value, max_value in self.motor_limits:
            if min_value > max_value:
                raise ValueError("Each motor limit minimum must be <= maximum.")


@dataclass(frozen=True)
class RightHandMappingResult:
    curls: FingerCurls
    finger_features: dict[str, FingerFeature]
    unclipped_targets: tuple[float, float, float, float, float, float, float, float]
    limited_targets: tuple[float, float, float, float, float, float, float, float]
    action: dict[str, float]
    motor_limit_clipped: bool
    max_step_clipped: bool

    @property
    def clipped(self) -> bool:
        return self.motor_limit_clipped or self.max_step_clipped


def _clip(value: float, min_value: float, max_value: float) -> float:
    return min(max(value, min_value), max_value)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b, strict=True)))


def landmarks_from_flat(
    values: Sequence[float] | None, max_abs_input_m: float = 2.0
) -> tuple[tuple[float, float, float], ...] | None:
    if values is None or len(values) != 63:
        return None
    float_values = tuple(float(v) for v in values)
    if not all(math.isfinite(v) and abs(v) <= max_abs_input_m for v in float_values):
        return None

    hand = HandSnapshot("right")
    hand.update_landmarks(float_values)
    return hand.landmarks


def _finger_curl_from_points(
    points: Sequence[tuple[float, float, float]],
    indices: tuple[int, int, int, int],
    open_ratio: float,
    closed_ratio: float,
) -> float:
    mcp_idx, proximal_idx, distal_idx, tip_idx = indices
    mcp = points[mcp_idx]
    proximal = points[proximal_idx]
    distal = points[distal_idx]
    tip = points[tip_idx]

    path_length = _distance(mcp, proximal) + _distance(proximal, distal) + _distance(distal, tip)
    if path_length <= 1e-9:
        return 1.0

    straight_distance = _distance(mcp, tip)
    extension_ratio = straight_distance / path_length
    curl = (open_ratio - extension_ratio) / (open_ratio - closed_ratio)
    return _clip(curl, 0.0, 1.0)


def _finger_feature_from_points(
    points: Sequence[tuple[float, float, float]],
    indices: tuple[int, int, int, int],
    open_ratio: float,
    closed_ratio: float,
) -> FingerFeature:
    mcp_idx, proximal_idx, distal_idx, tip_idx = indices
    mcp = points[mcp_idx]
    proximal = points[proximal_idx]
    distal = points[distal_idx]
    tip = points[tip_idx]
    joint_distances = (
        _distance(mcp, proximal),
        _distance(proximal, distal),
        _distance(distal, tip),
    )
    path_length = sum(joint_distances)
    tip_distance = _distance(mcp, tip)
    if path_length <= 1e-9:
        score = 1.0
    else:
        extension_ratio = tip_distance / path_length
        score = _clip((open_ratio - extension_ratio) / (open_ratio - closed_ratio), 0.0, 1.0)
    return FingerFeature(tip_distance=tip_distance, joint_distances=joint_distances, open_close_score=score)


def compute_finger_curls(
    landmarks: Sequence[tuple[float, float, float]],
    open_ratio: float = 0.95,
    closed_ratio: float = 0.45,
) -> FingerCurls:
    if len(landmarks) != 21:
        raise ValueError("landmarks must contain exactly 21 xyz points.")
    return FingerCurls(
        thumb=_finger_curl_from_points(landmarks, FINGER_INDICES["thumb"], open_ratio, closed_ratio),
        index=_finger_curl_from_points(landmarks, FINGER_INDICES["index"], open_ratio, closed_ratio),
        middle=_finger_curl_from_points(landmarks, FINGER_INDICES["middle"], open_ratio, closed_ratio),
        ring=_finger_curl_from_points(landmarks, FINGER_INDICES["ring"], open_ratio, closed_ratio),
        little=_finger_curl_from_points(landmarks, FINGER_INDICES["little"], open_ratio, closed_ratio),
    )


def compute_finger_features(
    landmarks: Sequence[tuple[float, float, float]],
    open_ratio: float = 0.95,
    closed_ratio: float = 0.45,
) -> dict[str, FingerFeature]:
    if len(landmarks) != 21:
        raise ValueError("landmarks must contain exactly 21 xyz points.")
    return {
        name: _finger_feature_from_points(landmarks, indices, open_ratio, closed_ratio)
        for name, indices in FINGER_INDICES.items()
    }


def curls_to_motor_targets(
    curls: FingerCurls,
    open_targets: Sequence[float],
    closed_targets: Sequence[float],
) -> tuple[float, float, float, float, float, float, float, float]:
    """Map five curls to eight provisional AmazingHand motor targets.

    AmazingHand mapping:
      motor 1,2 = finger1 = index
      motor 3,4 = finger2 = middle
      motor 5,6 = finger3 = ring/little coupled
      motor 7,8 = finger4 = thumb
    """
    thumb = _clip(curls.thumb, 0.0, 1.0)
    index = _clip(curls.index, 0.0, 1.0)
    middle = _clip(curls.middle, 0.0, 1.0)
    ring = _clip(curls.ring, 0.0, 1.0)
    little = _clip(curls.little, 0.0, 1.0)
    curl_values = (
        index,
        index,
        middle,
        middle,
        0.75 * ring + 0.25 * little,
        0.75 * ring + 0.25 * little,
        thumb,
        thumb,
    )
    targets = []
    for curl, open_target, closed_target in zip(curl_values, open_targets, closed_targets, strict=True):
        targets.append(open_target + curl * (closed_target - open_target))
    return tuple(targets)  # type: ignore[return-value]


def _apply_motor_limits(
    values: Sequence[float], motor_limits: Sequence[tuple[float, float]]
) -> tuple[tuple[float, float, float, float, float, float, float, float], bool]:
    clipped_values = []
    clipped = False
    for value, (min_value, max_value) in zip(values, motor_limits, strict=True):
        clipped_value = _clip(value, min_value, max_value)
        clipped_values.append(clipped_value)
        clipped = clipped or clipped_value != value
    return (tuple(clipped_values), clipped)  # type: ignore[return-value]


def _apply_max_step(
    values: Sequence[float], previous: Sequence[float], max_step: float
) -> tuple[tuple[float, float, float, float, float, float, float, float], bool]:
    clipped_values = []
    clipped = False
    for value, previous_value in zip(values, previous, strict=True):
        delta = value - previous_value
        clipped_delta = _clip(delta, -max_step, max_step)
        clipped_values.append(previous_value + clipped_delta)
        clipped = clipped or clipped_delta != delta
    return (tuple(clipped_values), clipped)  # type: ignore[return-value]


class RightHandDryRunMapper:
    """Stateful right-landmarks-to-AmazingHand dry-run mapper with safety clipping."""

    def __init__(self, config: RightHandMappingConfig | None = None) -> None:
        self.config = config or RightHandMappingConfig()
        self.previous_targets: tuple[float, float, float, float, float, float, float, float] = (
            self.config.open_targets
        )

    def map_landmarks(self, values: Sequence[float] | None) -> RightHandMappingResult | None:
        landmarks = landmarks_from_flat(values, max_abs_input_m=self.config.max_abs_input_m)
        if landmarks is None:
            return None

        curls = compute_finger_curls(
            landmarks,
            open_ratio=self.config.open_ratio,
            closed_ratio=self.config.closed_ratio,
        )
        finger_features = compute_finger_features(
            landmarks,
            open_ratio=self.config.open_ratio,
            closed_ratio=self.config.closed_ratio,
        )
        unclipped_targets = curls_to_motor_targets(
            curls, self.config.open_targets, self.config.closed_targets
        )
        motor_limited_targets, motor_limit_clipped = _apply_motor_limits(
            unclipped_targets, self.config.motor_limits
        )
        step_limited_targets, max_step_clipped = _apply_max_step(
            motor_limited_targets, self.previous_targets, self.config.max_step
        )
        self.previous_targets = step_limited_targets

        return RightHandMappingResult(
            curls=curls,
            finger_features=finger_features,
            unclipped_targets=unclipped_targets,
            limited_targets=step_limited_targets,
            action=dict(zip(RIGHT_HAND_ACTION_KEYS, step_limited_targets, strict=True)),
            motor_limit_clipped=motor_limit_clipped,
            max_step_clipped=max_step_clipped,
        )


def format_result(result: RightHandMappingResult) -> str:
    feature_text = " ".join(
        f"{name}:tip={feature.tip_distance:.4f},joints={_format_float_tuple(feature.joint_distances)},score={feature.open_close_score:.3f}"
        for name, feature in result.finger_features.items()
    )
    return (
        "curls="
        f"thumb:{result.curls.thumb:.3f} "
        f"index:{result.curls.index:.3f} "
        f"middle:{result.curls.middle:.3f} "
        f"ring:{result.curls.ring:.3f} "
        f"little:{result.curls.little:.3f} "
        f"finger_features=[{feature_text}] "
        f"action={{{', '.join(f'{k}: {v:+.3f}' for k, v in result.action.items())}}} "
        f"motor_limit_clipped={result.motor_limit_clipped} "
        f"max_step_clipped={result.max_step_clipped}"
    )


def _format_float_tuple(values: Sequence[float]) -> str:
    return "(" + ",".join(f"{value:.4f}" for value in values) + ")"


def _parse_float_tuple(text: str, expected_len: int, name: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(values) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} must contain {expected_len} comma-separated floats.")
    if not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError(f"{name} must contain finite floats.")
    return values


def _parse_motor_limits(text: str) -> tuple[tuple[float, float], ...]:
    values = _parse_float_tuple(text, 16, "motor-limits")
    limits = tuple((values[idx], values[idx + 1]) for idx in range(0, 16, 2))
    for min_value, max_value in limits:
        if min_value > max_value:
            raise argparse.ArgumentTypeError("motor-limits min values must be <= max values.")
    return limits


def handle_hts_line(line: str, mapper: RightHandDryRunMapper) -> RightHandMappingResult | None:
    parsed = parse_hts_line(line)
    if parsed is None:
        logging.debug("Ignored malformed or unsupported HTS line")
        return None

    side, kind, values = parsed
    if side != "right" or kind != "landmarks":
        return None

    result = mapper.map_landmarks(values)
    if result is None:
        logging.warning("Rejected right landmarks: invalid shape, non-finite value, or too-large input")
    return result


def serve_tcp(host: str, port: int, mapper: RightHandDryRunMapper, print_raw: bool) -> None:
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

    logging.info("Phase 2 dry-run only: no serial ports, motors, or LeRobot Robot will be used.")
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
    parser = argparse.ArgumentParser(description="Phase 2 right AmazingHand dry-run mapper for HTS.")
    parser.add_argument("--host", default="0.0.0.0", help="TCP bind host.")
    parser.add_argument("--port", type=int, default=8000, help="TCP bind port.")
    parser.add_argument("--max-step", type=float, default=5.0, help="Max target change per frame per motor.")
    parser.add_argument(
        "--max-abs-input-m",
        type=float,
        default=2.0,
        help="Reject landmark xyz values whose absolute value exceeds this many meters.",
    )
    parser.add_argument(
        "--hand-open-target",
        type=lambda text: _parse_float_tuple(text, 8, "hand-open-target"),
        default=DEFAULT_OPEN_TARGETS,
        help="Comma-separated 8D open hand target.",
    )
    parser.add_argument(
        "--hand-closed-target",
        type=lambda text: _parse_float_tuple(text, 8, "hand-closed-target"),
        default=DEFAULT_CLOSED_TARGETS,
        help="Comma-separated 8D closed hand target.",
    )
    parser.add_argument(
        "--motor-limits",
        type=_parse_motor_limits,
        default=DEFAULT_MOTOR_LIMITS,
        help="Comma-separated min,max pairs for eight motors.",
    )
    parser.add_argument("--print-raw", action="store_true", help="Print every raw HTS CSV line.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    mapper = RightHandDryRunMapper(
        RightHandMappingConfig(
            open_targets=args.hand_open_target,
            closed_targets=args.hand_closed_target,
            motor_limits=args.motor_limits,
            max_step=args.max_step,
            max_abs_input_m=args.max_abs_input_m,
        )
    )
    serve_tcp(host=args.host, port=args.port, mapper=mapper, print_raw=args.print_raw)


if __name__ == "__main__":
    main()
