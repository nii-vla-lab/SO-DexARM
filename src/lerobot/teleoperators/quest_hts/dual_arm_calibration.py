#!/usr/bin/env python
"""DualArm calibration and startup helpers (originally scripts/quest_hts/phase31).

Subcommands:
  capture-hand  -- Record AmazingHand + human hand pose (fist/mid/open)
  capture-startup -- Record SO-101 startup position
  startup       -- Connect Quest HTS, move to start position, begin teleop/record
  map-hand      -- Print current hand mapping (for diagnostics)

Generated hand calibration and startup positions are saved under `.cache/so_dexarm/` by default.

Usage example:
  python -m lerobot.teleoperators.quest_hts.dual_arm_calibration capture-hand --side right --pose fist
  python -m lerobot.teleoperators.quest_hts.dual_arm_calibration capture-hand --side right --pose mid
  python -m lerobot.teleoperators.quest_hts.dual_arm_calibration capture-hand --side right --pose open
  python -m lerobot.teleoperators.quest_hts.dual_arm_calibration capture-startup --side right
  python -m lerobot.teleoperators.quest_hts.dual_arm_calibration startup --side right
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_HAND_CALIB_FILE = Path(".cache/so_dexarm/quest_hts_dual_arm_hand_calibration.yaml")
DEFAULT_STARTUP_FILE = Path(".cache/so_dexarm/quest_hts_dual_arm_startup.yaml")
DEFAULT_HTS_HOST = "0.0.0.0"
DEFAULT_HTS_PORT = 8000
# Capture order: open (パー) → mid (half-open) → fist (グー). All three poses are anchors of a
# piecewise-linear open/close mapping.
POSES = ("open", "mid", "fist")

DEFAULT_ARM_PORTS = {
    "right": "/dev/ttyso101_amazinghand_r_arm",
    "left": "/dev/ttyso101_amazinghand_l_arm",
}
DEFAULT_HAND_PORTS = {
    "right": "/dev/ttyso101_amazinghand_r_hand",
    "left": "/dev/ttyso101_amazinghand_l_hand",
}
DEFAULT_ARM_CALIB_DIRS = {
    "right": Path(".cache/calibration/robots/so101_amazinghand_right"),
    "left": Path(".cache/calibration/robots/so101_amazinghand_left"),
}

HTS_ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
HTS_HAND_MOTORS = tuple(f"finger{i}_motor{j}" for i in range(1, 5) for j in range(1, 3))


def side_prefix(side: str) -> str:
    return "r_" if side == "right" else "l_"


def hand_keys(side: str) -> tuple[str, ...]:
    p = side_prefix(side)
    return tuple(f"{p}finger{i}_motor{j}.pos" for i in range(1, 5) for j in range(1, 3))


def arm_keys(side: str) -> tuple[str, ...]:
    p = side_prefix(side)
    return tuple(f"{p}{j}.pos" for j in HTS_ARM_JOINTS)


def all_dual_arm_keys() -> tuple[str, ...]:
    return arm_keys("right") + arm_keys("left") + hand_keys("right") + hand_keys("left")


def resolve_input_hand(side: str, *, input_assignment_mode: str = "native_right") -> str:
    """Return the HTS raw label to use for the given physical side."""
    if input_assignment_mode == "native_right":
        return side  # right->right, left->left (in native mode both match physical)
    return side


def resolve_input_hand_map(
    sides: Sequence[str],
    *,
    input_assignment_mode: str = "native_right",
) -> dict[str, str]:
    return {s: resolve_input_hand(s, input_assignment_mode=input_assignment_mode) for s in sides}


# --------------------------------------------------------------------------- #
# Port map
# --------------------------------------------------------------------------- #


@dataclass
class PhysicalPortMap:
    right_arm_port: str = DEFAULT_ARM_PORTS["right"]
    left_arm_port: str = DEFAULT_ARM_PORTS["left"]
    right_hand_port: str = DEFAULT_HAND_PORTS["right"]
    left_hand_port: str = DEFAULT_HAND_PORTS["left"]
    right_calib_dir: Path = DEFAULT_ARM_CALIB_DIRS["right"]
    left_calib_dir: Path = DEFAULT_ARM_CALIB_DIRS["left"]

    def for_side(self, side: str) -> dict[str, str | Path]:
        if side == "right":
            return {
                "arm_port": self.right_arm_port,
                "hand_port": self.right_hand_port,
                "calib_dir": self.right_calib_dir,
            }
        return {
            "arm_port": self.left_arm_port,
            "hand_port": self.left_hand_port,
            "calib_dir": self.left_calib_dir,
        }

    def to_metadata(self) -> dict:
        return {
            "right_arm_port": self.right_arm_port,
            "left_arm_port": self.left_arm_port,
            "right_hand_port": self.right_hand_port,
            "left_hand_port": self.left_hand_port,
        }


@dataclass
class StartupConfig:
    sides: tuple[str, ...] = ("right",)
    port_map: PhysicalPortMap = field(default_factory=PhysicalPortMap)
    hts_host: str = DEFAULT_HTS_HOST
    hts_port: int = DEFAULT_HTS_PORT
    hand_calib_file: Path = DEFAULT_HAND_CALIB_FILE
    startup_file: Path = DEFAULT_STARTUP_FILE
    mode: str = "arm-and-hand"
    require_hardware: bool = False
    enable_motion: bool = False
    move_to_start: bool = True
    move_speed: float = 0.5
    hts_wait_timeout_s: float = 30.0
    baseline_wait_timeout_s: float = 30.0


# --------------------------------------------------------------------------- #
# Lightweight adapters for arm/hand bus
# --------------------------------------------------------------------------- #


class ArmAdapter:
    """Minimal arm bus adapter for reading/writing positions."""

    def __init__(self, *, port: str, calib_dir: Path):
        self._port = port
        self._calib_dir = calib_dir
        self._robot = None

    def connect(self) -> None:
        from lerobot.robots.so101_amazinghand_right import SO101AmazingHandRightConfig
        from lerobot.robots.utils import make_robot_from_config

        self._robot = make_robot_from_config(
            SO101AmazingHandRightConfig(
                id="right",
                port=self._port,
                mode="arm-only",
                calibration_dir=self._calib_dir,
                require_calibration=True,
            )
        )
        self._robot.connect(calibrate=False)

    def read_current_action(self) -> dict[str, float]:
        if self._robot is None:
            return {}
        obs = self._robot.get_observation()
        return {k: float(v) for k, v in obs.items() if ".pos" in k}

    def send_action(self, action: dict[str, float]) -> None:
        if self._robot is not None:
            self._robot.send_action(action)

    def disconnect(self) -> None:
        if self._robot is not None and self._robot.is_connected:
            self._robot.disconnect()
        self._robot = None


class HandAdapter:
    """Minimal hand bus adapter for reading/writing positions."""

    def __init__(self, *, port: str):
        self._port = port
        self._bus = None

    def connect(self) -> None:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        motors = {
            f"finger{i}_motor{j}": Motor((i - 1) * 2 + j, "scs0009", MotorNormMode.RANGE_0_100)
            for i in range(1, 5)
            for j in range(1, 3)
        }
        self._bus = FeetechMotorsBus(port=self._port, motors=motors, protocol_version=1)
        self._bus.connect(handshake=False)

    def read_current_action(self) -> dict[str, float]:
        if self._bus is None:
            return {}
        result = {}
        for motor in self._bus.motors:
            try:
                result[f"r_{motor}.pos"] = float(self._bus.read("Present_Position", motor))
            except Exception:
                result[f"r_{motor}.pos"] = 0.0
        return result

    def disconnect(self) -> None:
        if self._bus is not None and self._bus.is_connected:
            self._bus.disconnect(disable_torque=False)
        self._bus = None


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def _validate_side(side: str) -> None:
    if side not in {"right", "left", "both"}:
        raise ValueError(f"--side must be right, left, or both; got {side!r}")


def _validate_pose(pose: str) -> None:
    if pose not in POSES:
        raise ValueError(f"--pose must be one of {POSES}; got {pose!r}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# YAML helpers
# --------------------------------------------------------------------------- #


def read_yaml(path: Path) -> dict:
    import yaml

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def write_yaml(path: Path, payload: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _merge_pose_payload(existing: dict, side: str, category: str, pose: str, data: dict) -> dict:
    """Merge a pose measurement into existing payload without overwriting other entries."""
    existing.setdefault("version", 2)
    existing.setdefault("created_at", _utc_now())
    existing["updated_at"] = _utc_now()
    side_entry = existing.setdefault(side, {})
    cat_entry = side_entry.setdefault(category, {})
    cat_entry[pose] = data
    return existing


# --------------------------------------------------------------------------- #
# Hand calibration: robot poses
# --------------------------------------------------------------------------- #


def save_robot_hand_pose(
    *,
    path: Path,
    side: str,
    pose: str,
    positions: dict[str, float],
) -> dict:
    """Save robot AmazingHand positions for a given hand pose."""
    payload = read_yaml(path)
    clean = {k: float(v) for k, v in positions.items() if "finger" in k}
    data = {
        "timestamp": _utc_now(),
        "positions": clean,
    }
    payload = _merge_pose_payload(payload, side, "robot", pose, data)
    write_yaml(path, payload)
    print(f"Saved robot {side} hand {pose} pose -> {path}", flush=True)
    return payload


# --------------------------------------------------------------------------- #
# Hand calibration: human hand from HTS landmarks
# --------------------------------------------------------------------------- #

# Per-finger landmark chains (MediaPipe 21-point convention), wrist-anchored so the
# metacarpal (wrist→MCP) is the curl reference for the knuckle joint.
#   thumb : 0(wrist) 1(CMC) 2(MCP) 3(IP)  4(TIP)
#   index : 0(wrist) 5(MCP) 6(PIP) 7(DIP) 8(TIP)
#   middle: 0(wrist) 9(MCP) 10(PIP) 11(DIP) 12(TIP)
#   ring  : 0(wrist) 13(MCP) 14(PIP) 15(DIP) 16(TIP)
_FINGER_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
}

# Default human-hand feature used for new calibrations and runtime.
DEFAULT_HAND_FEATURE_MODE = "curl"


def _turn_angle(a, b, c) -> float:
    """Bend angle at b for the path a→b→c, in radians. 0 = straight, π = folded back.

    Scale-invariant (depends only on direction), so it is robust to how far the hand is
    from the Quest camera — unlike a raw tip-to-MCP distance.
    """
    v1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    v2 = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    n1 = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) ** 0.5
    n2 = (v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2) ** 0.5
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    dot = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


def hand_feature_vector_from_landmarks(
    landmarks: Sequence[float],
    mode: str = DEFAULT_HAND_FEATURE_MODE,
) -> dict[str, float]:
    """Extract per-finger open/close feature from raw HTS landmark values.

    Only the 4 fingers that drive AmazingHand motors are tracked: thumb, index, middle, ring.
    The human pinky (little) is intentionally not tracked — the AmazingHand has
    only 4 finger pairs and the pinky added jitter to finger3.

    Two feature modes (the same mode MUST be used at calibration capture and at runtime — this
    is guaranteed by stamping the mode into the calibration file; see ``save_human_hand_pose`` /
    ``build_hand_mapping`` / ``PiecewiseHandMapper``):

    - ``"curl"`` (default): sum of the three joint bend angles along the finger
      (MCP + PIP + DIP), in radians. This rises roughly linearly from an open hand (~0) to a
      fist (~3–4 rad), so it keeps resolution across the WHOLE range — including the closed/grip
      end, where the old distance feature saturated and made grips "雑" (indistinguishable).
    - ``"distance"`` (legacy): tip-to-MCP straight-line distance / palm_scale. Kept so existing
      calibration files (which have no stamped mode) keep behaving exactly as before.
    """
    # HTS landmarks are 21 points x 3 values = 63 floats (x, y, z, ...)
    if len(landmarks) < 63:
        return {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0}

    def point(idx: int) -> tuple[float, float, float]:
        base = idx * 3
        return float(landmarks[base]), float(landmarks[base + 1]), float(landmarks[base + 2])

    def dist(a: tuple, b: tuple) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b, strict=False)) ** 0.5

    if mode == "curl":
        features: dict[str, float] = {}
        for name, (i0, i1, i2, i3, i4) in _FINGER_CHAINS.items():
            p0, p1, p2, p3, p4 = point(i0), point(i1), point(i2), point(i3), point(i4)
            features[name] = (
                _turn_angle(p0, p1, p2)  # knuckle (MCP) flexion vs metacarpal
                + _turn_angle(p1, p2, p3)  # PIP / thumb-MCP
                + _turn_angle(p2, p3, p4)  # DIP / thumb-IP
            )
        return features

    # Legacy distance feature.
    wrist = point(0)
    features = {
        "thumb": dist(point(4), point(1)),
        "index": dist(point(8), point(5)),
        "middle": dist(point(12), point(9)),
        "ring": dist(point(16), point(13)),
    }
    # Normalize by wrist-to-middle-MCP distance
    palm_scale = dist(wrist, point(9)) or 1.0
    return {k: v / palm_scale for k, v in features.items()}


def save_human_hand_pose(
    *,
    path: Path,
    side: str,
    pose: str,
    landmarks: Sequence[float] | None,
    feature_mode: str = DEFAULT_HAND_FEATURE_MODE,
) -> dict:
    """Save human hand landmark features for a given pose.

    The active ``feature_mode`` is stamped at the payload top level so that runtime mapping
    recomputes the live feature with the SAME formula the anchors were captured with. Mixing
    feature modes between capture and runtime silently breaks the grip, so this is the single
    source of truth — never recompute features without consulting it.
    """
    payload = read_yaml(path)
    features = hand_feature_vector_from_landmarks(landmarks, mode=feature_mode) if landmarks else {}
    data = {
        "timestamp": _utc_now(),
        "features": features,
        "raw_landmarks_count": len(landmarks) if landmarks else 0,
    }
    payload = _merge_pose_payload(payload, side, "human", pose, data)
    # Stamp the feature mode PER SIDE (primary, wins at runtime) so re-capturing ONE hand can no
    # longer relabel the OTHER side's anchors. Keep the global stamp too for backward compatibility.
    payload.setdefault(side, {})["feature_mode"] = feature_mode
    payload["feature_mode"] = feature_mode
    write_yaml(path, payload)
    print(f"Saved human {side} hand {pose} features -> {path}", flush=True)
    return payload


# --------------------------------------------------------------------------- #
# Piecewise linear hand mapping
# --------------------------------------------------------------------------- #


def _feature_for_group(features: dict[str, float], fingers: Sequence[str]) -> float:
    scores = [features[f] for f in fingers if f in features]
    return sum(scores) / len(scores) if scores else 0.0


def _piecewise_linear(x: float, anchors: list[tuple[float, float]]) -> float:
    """Interpolate x through anchor points (x_in, y_out) sorted by x_in."""
    anchors = sorted(anchors, key=lambda p: p[0])
    if x <= anchors[0][0]:
        return anchors[0][1]
    if x >= anchors[-1][0]:
        return anchors[-1][1]
    for i in range(len(anchors) - 1):
        x0, y0 = anchors[i]
        x1, y1 = anchors[i + 1]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 + t * (y1 - y0)
    return anchors[-1][1]


class PiecewiseHandMapper:
    """Maps human hand feature values to robot AmazingHand motor targets."""

    # Fixed human-finger → robot motor-pair binding.
    #   finger1 = motor IDs 1-2  ← thumb
    #   finger2 = motor IDs 3-4  ← index
    #   finger3 = motor IDs 5-6  ← middle
    #   finger4 = motor IDs 7-8  ← ring
    # The human pinky (little) is no longer tracked.
    FINGER_GROUPS = {
        "finger1": ("thumb",),
        "finger2": ("index",),
        "finger3": ("middle",),
        "finger4": ("ring",),
    }

    def __init__(self, mapping: dict, side: str):
        self._mapping = mapping
        self._side = side
        self._prefix = side_prefix(side)
        # Feature mode the anchors were captured with. Runtime MUST compute live features with
        # this same mode (see hand_feature_vector_from_landmarks). Legacy mappings without the
        # stamp predate the curl feature, so they default to "distance".
        self.feature_mode = mapping.get("_feature_mode", "distance")

    def map_features(self, features: dict[str, float]) -> dict[str, float]:
        """Return motor target positions for the given finger features."""
        targets: dict[str, float] = {}
        for finger, source_fingers in self.FINGER_GROUPS.items():
            feature_val = _feature_for_group(features, list(source_fingers))
            for motor in (1, 2):
                key = f"{self._prefix}{finger}_motor{motor}.pos"
                anchors_data = self._mapping.get(finger, {}).get(f"motor{motor}", [])
                if anchors_data:
                    anchors = [(float(p["in"]), float(p["out"])) for p in anchors_data]
                    targets[key] = _piecewise_linear(feature_val, anchors)
        return targets


def build_hand_mapping(
    *,
    side: str,
    hand_calib_file: Path,
) -> dict:
    """Build piecewise linear mapping from calibration file."""
    payload = read_yaml(hand_calib_file)
    side_data = payload.get(side, {})
    human = side_data.get("human", {})
    robot = side_data.get("robot", {})
    mapping: dict = {}
    # Carry the capture-time feature mode so runtime recomputes the live feature identically.
    # PER-SIDE stamp (side_data["feature_mode"]) wins over the global top-level one, so the two
    # hands can use DIFFERENT feature modes (e.g. one still distance-era, one re-captured under curl)
    # without the global stamp silently mislabeling the other side. Falls back to the global stamp,
    # then "distance" for legacy files with no stamp at all.
    mapping["_feature_mode"] = side_data.get("feature_mode", payload.get("feature_mode", "distance"))
    # Must match PiecewiseHandMapper.FINGER_GROUPS:
    # finger1←thumb (IDs 1-2), finger2←index (IDs 3-4), finger3←middle (IDs 5-6), finger4←ring (IDs 7-8).
    finger_groups = {
        "finger1": ("thumb",),
        "finger2": ("index",),
        "finger3": ("middle",),
        "finger4": ("ring",),
    }
    for finger, source_fingers in finger_groups.items():
        mapping[finger] = {}
        for motor_num in (1, 2):
            motor_key = f"motor{motor_num}"
            anchors = []
            for pose in POSES:
                h = human.get(pose, {}).get("features", {})
                r = robot.get(pose, {}).get("positions", {})
                if not h or not r:
                    continue
                in_val = _feature_for_group(h, list(source_fingers))
                p = side_prefix(side)
                out_val = r.get(f"{p}{finger}_{motor_key}.pos", 0.0)
                anchors.append({"in": round(in_val, 5), "out": round(float(out_val), 3)})
            mapping[finger][motor_key] = anchors
    return mapping


def mapper_from_mapping_payload(mapping: dict, side: str) -> PiecewiseHandMapper:
    return PiecewiseHandMapper(mapping, side)


# --------------------------------------------------------------------------- #
# Startup position
# --------------------------------------------------------------------------- #


def save_start_position(
    *,
    path: Path,
    side: str,
    arm_positions: dict[str, float],
    hand_positions: dict[str, float] | None = None,
    wrist_xyz: tuple[float, float, float] | None = None,
) -> dict:
    """Save arm/hand start position for a side."""
    payload = read_yaml(path)
    payload.setdefault("version", 1)
    payload.setdefault("created_at", _utc_now())
    payload["updated_at"] = _utc_now()
    side_entry = payload.setdefault(side, {})
    side_entry["arm_positions"] = {k: float(v) for k, v in arm_positions.items()}
    if hand_positions:
        side_entry["hand_positions"] = {k: float(v) for k, v in hand_positions.items()}
    if wrist_xyz:
        side_entry["human_wrist_xyz"] = {"x": wrist_xyz[0], "y": wrist_xyz[1], "z": wrist_xyz[2]}
    write_yaml(path, payload)
    print(f"Saved {side} startup position -> {path}", flush=True)
    return payload


def load_start_position(path: Path, side: str) -> dict:
    """Load startup position for a side."""
    payload = read_yaml(path)
    return payload.get(side, {})


# --------------------------------------------------------------------------- #
# HTS stream helpers
# --------------------------------------------------------------------------- #


def summarize_hts_lines(lines: list[str]) -> dict:
    """Count received HTS streams."""
    from lerobot.teleoperators.quest_hts.hts_protocol import parse_hts_line

    counts: dict[tuple[str, str], int] = {}
    landmarks_by_side: dict[str, list[float]] = {}
    wrist_by_side: dict[str, list[float]] = {}
    for line in lines:
        parsed = parse_hts_line(line)
        if parsed is None:
            continue
        side, kind, values = parsed
        counts[(side, kind)] = counts.get((side, kind), 0) + 1
        if kind == "landmarks":
            landmarks_by_side[side] = list(values)
        elif kind == "wrist":
            wrist_by_side[side] = list(values)
    return {
        "counts": {f"{s}_{k}": v for (s, k), v in counts.items()},
        "landmarks": landmarks_by_side,
        "wrist": wrist_by_side,
        "has_right_landmarks": ("right", "landmarks") in counts,
        "has_left_landmarks": ("left", "landmarks") in counts,
        "has_right_wrist": ("right", "wrist") in counts,
        "has_left_wrist": ("left", "wrist") in counts,
    }


def _format_hts_debug(summary: dict) -> str:
    parts = []
    for key in ("has_right_wrist", "has_right_landmarks", "has_left_wrist", "has_left_landmarks"):
        parts.append(f"{key}={summary.get(key, False)}")
    return " ".join(parts)


def _consume_hts_records(
    sides: Sequence[str],
    *,
    host: str,
    port: int,
    timeout_s: float = 30.0,
    max_lines: int = 100,
) -> tuple[list[str], dict]:
    """Open a TCP server, accept one connection, read HTS lines until sides are seen."""
    import socket

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    server.settimeout(timeout_s)
    lines: list[str] = []
    try:
        conn, _ = server.accept()
        conn.settimeout(0.5)
        buf = ""
        deadline = time.monotonic() + timeout_s
        with conn:
            while time.monotonic() < deadline and len(lines) < max_lines:
                try:
                    data = conn.recv(4096)
                except TimeoutError:
                    continue
                if not data:
                    break
                buf += data.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        lines.append(line)
                summary = summarize_hts_lines(lines)
                ready = all(
                    summary.get(f"has_{s}_wrist") and summary.get(f"has_{s}_landmarks") for s in sides
                )
                if ready:
                    break
    finally:
        server.close()
    summary = summarize_hts_lines(lines)
    return lines, summary


def quest_ready_from_lines(lines: list[str], sides: Sequence[str]) -> tuple[bool, dict]:
    """Return (ready, summary) for the given sides."""
    summary = summarize_hts_lines(lines)
    ready = all(summary.get(f"has_{s}_wrist") and summary.get(f"has_{s}_landmarks") for s in sides)
    return ready, summary


def collect_hts_lines(
    sides: Sequence[str],
    *,
    host: str,
    port: int,
    timeout_s: float = 30.0,
) -> tuple[list[str], dict]:
    """Convenience: collect HTS lines and summarize."""
    lines, summary = _consume_hts_records(sides, host=host, port=port, timeout_s=timeout_s)
    return lines, summary


def _required_for_startup(sides: Sequence[str]) -> set[str]:
    required = set()
    for s in sides:
        required.add(f"has_{s}_wrist")
        required.add(f"has_{s}_landmarks")
    return required


# --------------------------------------------------------------------------- #
# Startup motions
# --------------------------------------------------------------------------- #


def interpolate_joint_steps(
    start: dict[str, float],
    goal: dict[str, float],
    *,
    steps: int = 20,
) -> list[dict[str, float]]:
    """Generate linearly interpolated joint waypoints from start to goal."""
    keys = list(goal.keys())
    waypoints = []
    for i in range(1, steps + 1):
        t = i / steps
        waypoint = {k: start.get(k, goal[k]) + t * (goal[k] - start.get(k, goal[k])) for k in keys}
        waypoints.append(waypoint)
    return waypoints


def require_hardware_motion_flags(require_hardware: bool, enable_motion: bool) -> None:
    if require_hardware and not enable_motion:
        raise RuntimeError(
            "Hardware motion requires --enable-motion flag. Pass --enable-motion to confirm motion is safe."
        )


def move_to_start_position(
    adapter: ArmAdapter,
    start_positions: dict[str, float],
    *,
    steps: int = 20,
    step_delay_s: float = 0.05,
) -> None:
    """Gradually move the arm to the saved startup position."""
    current = adapter.read_current_action()
    waypoints = interpolate_joint_steps(current, start_positions, steps=steps)
    for wp in waypoints:
        adapter.send_action(wp)
        time.sleep(step_delay_s)


def run_startup_sequence(
    config: StartupConfig,
    *,
    arm_adapter_factory: Callable | None = None,
    hts_lines: list[str] | None = None,
) -> dict:
    """Full startup: check HTS, move to start, baseline human wrist."""
    result: dict = {
        "sides": list(config.sides),
        "hts_ready": False,
        "motion_completed": dict.fromkeys(config.sides, False),
        "baseline_captured": dict.fromkeys(config.sides, False),
        "errors": [],
    }

    # Step 1: Check HTS stream
    if hts_lines is None:
        print("Waiting for Quest HTS stream...", flush=True)
        try:
            hts_lines, summary = collect_hts_lines(
                config.sides,
                host=config.hts_host,
                port=config.hts_port,
                timeout_s=config.hts_wait_timeout_s,
            )
        except Exception as exc:
            result["errors"].append(f"HTS collect error: {exc}")
            return result
    else:
        ready, summary = quest_ready_from_lines(hts_lines, config.sides)

    ready, summary = quest_ready_from_lines(hts_lines, config.sides)
    result["hts_summary"] = summary
    if not ready:
        missing = [key for key in _required_for_startup(config.sides) if not summary.get(key)]
        result["errors"].append(f"Quest HTS not ready. Missing: {missing}")
        print(f"Quest HTS not ready. Missing: {missing}", flush=True)
        return result

    result["hts_ready"] = True
    print(f"Quest HTS ready: {_format_hts_debug(summary)}", flush=True)

    # Step 2: Move to start position per side
    if config.enable_motion or config.require_hardware:
        require_hardware_motion_flags(config.require_hardware, config.enable_motion)
        startup_positions = load_start_position(config.startup_file, config.sides[0])
        if not startup_positions:
            result["errors"].append(f"No startup position saved for sides {config.sides}")
            return result

        for side in config.sides:
            positions = load_start_position(config.startup_file, side)
            arm_positions = positions.get("arm_positions", {})
            if not arm_positions:
                result["errors"].append(f"No arm positions for side {side}")
                continue

            cfg = config.port_map.for_side(side)
            make_adapter = arm_adapter_factory or (
                lambda cfg=cfg: ArmAdapter(
                    port=str(cfg["arm_port"]),
                    calib_dir=Path(cfg["calib_dir"]),
                )
            )
            adapter = make_adapter(cfg)
            try:
                adapter.connect()
                move_to_start_position(adapter, arm_positions)
                result["motion_completed"][side] = True
                print(f"Moved {side} arm to start position.", flush=True)
            finally:
                adapter.disconnect()

    # Step 3: Baseline human wrist
    for side in config.sides:
        wrist = summary.get("wrist", {}).get(side)
        if wrist:
            result["baseline_captured"][side] = True
            print(f"Captured {side} wrist baseline: {wrist[:3]}", flush=True)

    return result


# --------------------------------------------------------------------------- #
# DualArm record/teleop command builders
# --------------------------------------------------------------------------- #


def _command(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def build_dual_arm_record_command(
    port_map: PhysicalPortMap,
    *,
    repo_id: str,
    num_episodes: int = 5,
    episode_time_s: int = 30,
    reset_time_s: int = 5,
    hts_host: str = DEFAULT_HTS_HOST,
    hts_port: int = DEFAULT_HTS_PORT,
) -> list[str]:
    """Build lerobot record command for dual_arm SO-101 + AmazingHand."""
    return _command(
        "lerobot.scripts.control_robot",
        "record",
        "--robot.type",
        "dual_arm",
        "--robot.right-arm-port",
        port_map.right_arm_port,
        "--robot.left-arm-port",
        port_map.left_arm_port,
        "--robot.right-hand-port",
        port_map.right_hand_port,
        "--robot.left-hand-port",
        port_map.left_hand_port,
        "--teleop.type",
        "quest_hts_right",
        "--teleop.input-assignment-mode",
        "native_right",
        "--teleop.host",
        hts_host,
        "--teleop.port",
        str(hts_port),
        "--repo-id",
        repo_id,
        "--num-episodes",
        str(num_episodes),
        "--episode-time-s",
        str(episode_time_s),
        "--reset-time-s",
        str(reset_time_s),
        "--push-to-hub",
        "false",
    )


def build_dual_arm_teleop_command(
    port_map: PhysicalPortMap,
    *,
    hts_host: str = DEFAULT_HTS_HOST,
    hts_port: int = DEFAULT_HTS_PORT,
) -> list[str]:
    return _command(
        "lerobot.scripts.control_robot",
        "teleoperate",
        "--robot.type",
        "dual_arm",
        "--robot.right-arm-port",
        port_map.right_arm_port,
        "--robot.left-arm-port",
        port_map.left_arm_port,
        "--robot.right-hand-port",
        port_map.right_hand_port,
        "--robot.left-hand-port",
        port_map.left_hand_port,
        "--teleop.type",
        "quest_hts_right",
        "--teleop.input-assignment-mode",
        "native_right",
        "--teleop.host",
        hts_host,
        "--teleop.port",
        str(hts_port),
    )


def command_to_string(cmd: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(part)) for part in cmd)


def _print_or_run(cmd: list[str], *, run: bool = False) -> int:
    if run:
        import subprocess

        return subprocess.run(cmd, check=False).returncode
    print(command_to_string(cmd), flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Hardware read helpers for CLI subcommands
# --------------------------------------------------------------------------- #


def _parse_key_values(text: str) -> dict[str, float]:
    """Parse 'key=value key=value ...' from arm read result."""
    result = {}
    for part in text.split():
        if "=" in part:
            k, v = part.split("=", 1)
            with contextlib.suppress(ValueError):
                result[k] = float(v)
    return result


def _build_raw_adapter_factory(port_map: PhysicalPortMap, side: str) -> Callable:
    cfg = port_map.for_side(side)

    def factory(*_args, **_kwargs) -> ArmAdapter:
        return ArmAdapter(port=str(cfg["arm_port"]), calib_dir=Path(cfg["calib_dir"]))

    return factory


def read_robot_hand_positions_from_hardware(side: str, port_map: PhysicalPortMap) -> dict[str, float]:
    """Read current hand motor positions directly from hardware."""
    adapter = HandAdapter(port=port_map.for_side(side)["hand_port"])
    try:
        adapter.connect()
        return adapter.read_current_action()
    finally:
        adapter.disconnect()


def read_start_position_from_hardware(side: str, port_map: PhysicalPortMap) -> dict[str, float]:
    """Read current arm joint positions from hardware."""
    adapter = ArmAdapter(
        port=str(port_map.for_side(side)["arm_port"]),
        calib_dir=Path(port_map.for_side(side)["calib_dir"]),
    )
    try:
        adapter.connect()
        return adapter.read_current_action()
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase31: DualArm calibration and startup.",
    )
    parser.add_argument("--right-arm-port", default=DEFAULT_ARM_PORTS["right"])
    parser.add_argument("--left-arm-port", default=DEFAULT_ARM_PORTS["left"])
    parser.add_argument("--right-hand-port", default=DEFAULT_HAND_PORTS["right"])
    parser.add_argument("--left-hand-port", default=DEFAULT_HAND_PORTS["left"])
    parser.add_argument("--right-calib-dir", type=Path, default=DEFAULT_ARM_CALIB_DIRS["right"])
    parser.add_argument("--left-calib-dir", type=Path, default=DEFAULT_ARM_CALIB_DIRS["left"])
    parser.add_argument("--hts-host", default=DEFAULT_HTS_HOST)
    parser.add_argument("--hts-port", type=int, default=DEFAULT_HTS_PORT)
    parser.add_argument("--hand-calib-file", type=Path, default=DEFAULT_HAND_CALIB_FILE)
    parser.add_argument("--startup-file", type=Path, default=DEFAULT_STARTUP_FILE)

    sub = parser.add_subparsers(dest="command", required=True)

    # -- capture-hand --
    p_ch = sub.add_parser("capture-hand", help="Record hand pose (fist/mid/open) for one side.")
    p_ch.add_argument("--side", choices=("right", "left"), required=True)
    p_ch.add_argument("--pose", choices=POSES, required=True)
    p_ch.add_argument("--from-hardware", action="store_true", help="Also read robot hand positions.")

    # -- capture-startup --
    p_cs = sub.add_parser("capture-startup", help="Record arm startup position for one side.")
    p_cs.add_argument("--side", choices=("right", "left", "both"), default="right")
    p_cs.add_argument(
        "--from-hardware", action="store_true", help="Read current arm positions from hardware."
    )

    # -- startup --
    p_startup = sub.add_parser("startup", help="Run the full startup sequence.")
    p_startup.add_argument("--side", choices=("right", "left", "both"), default="right")
    p_startup.add_argument(
        "--enable-motion", action="store_true", help="Allow robot to move to start position."
    )
    p_startup.add_argument("--require-hardware", action="store_true")
    p_startup.add_argument("--hts-wait-timeout-s", type=float, default=30.0)
    p_startup.add_argument("--move-speed", type=float, default=0.5)

    # -- map-hand --
    p_map = sub.add_parser("map-hand", help="Print computed hand mapping from calibration file.")
    p_map.add_argument("--side", choices=("right", "left", "both"), default="right")

    return parser


def port_map_from_args(args: argparse.Namespace) -> PhysicalPortMap:
    return PhysicalPortMap(
        right_arm_port=args.right_arm_port,
        left_arm_port=args.left_arm_port,
        right_hand_port=args.right_hand_port,
        left_hand_port=args.left_hand_port,
        right_calib_dir=Path(args.right_calib_dir),
        left_calib_dir=Path(args.left_calib_dir),
    )


def run(args: argparse.Namespace) -> int:
    port_map = port_map_from_args(args)
    cmd = args.command

    if cmd == "capture-hand":
        _validate_side(args.side)
        _validate_pose(args.pose)

        print(f"[capture-hand] side={args.side} pose={args.pose}", flush=True)
        print("[capture-hand] Connecting to Quest HTS...", flush=True)

        # Collect HTS landmarks
        try:
            lines, summary = collect_hts_lines(
                [args.side],
                host=args.hts_host,
                port=args.hts_port,
                timeout_s=30.0,
            )
        except Exception as exc:
            print(f"[capture-hand] HTS error: {exc}", flush=True)
            return 1

        landmarks = summary.get("landmarks", {}).get(args.side)
        save_human_hand_pose(
            path=args.hand_calib_file,
            side=args.side,
            pose=args.pose,
            landmarks=landmarks,
        )

        if getattr(args, "from_hardware", False):
            print("[capture-hand] Reading robot hand positions...", flush=True)
            try:
                positions = read_robot_hand_positions_from_hardware(args.side, port_map)
                save_robot_hand_pose(
                    path=args.hand_calib_file,
                    side=args.side,
                    pose=args.pose,
                    positions=positions,
                )
            except Exception as exc:
                print(f"[capture-hand] Hand read error: {exc}", flush=True)
                return 1

        print("[capture-hand] Done.", flush=True)
        return 0

    if cmd == "capture-startup":
        sides = ("right", "left") if args.side == "both" else (args.side,)
        for side in sides:
            _validate_side(side)
            if getattr(args, "from_hardware", False):
                print(f"[capture-startup] Reading {side} arm positions...", flush=True)
                try:
                    arm_positions = read_start_position_from_hardware(side, port_map)
                except Exception as exc:
                    print(f"[capture-startup] Arm read error: {exc}", flush=True)
                    return 1
            else:
                print("[capture-startup] No hardware read. Saving empty positions.", flush=True)
                arm_positions = {}
            save_start_position(
                path=args.startup_file,
                side=side,
                arm_positions=arm_positions,
            )
        print("[capture-startup] Done.", flush=True)
        return 0

    if cmd == "startup":
        sides = ("right", "left") if args.side == "both" else (args.side,)
        config = StartupConfig(
            sides=sides,
            port_map=port_map,
            hts_host=args.hts_host,
            hts_port=args.hts_port,
            hand_calib_file=args.hand_calib_file,
            startup_file=args.startup_file,
            enable_motion=getattr(args, "enable_motion", False),
            require_hardware=getattr(args, "require_hardware", False),
            hts_wait_timeout_s=getattr(args, "hts_wait_timeout_s", 30.0),
        )
        result = run_startup_sequence(config)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if not result.get("errors") else 1

    if cmd == "map-hand":
        sides = ("right", "left") if args.side == "both" else (args.side,)
        for side in sides:
            mapping = build_hand_mapping(side=side, hand_calib_file=args.hand_calib_file)
            print(json.dumps({side: mapping}, indent=2), flush=True)
        return 0

    print(f"Unknown command: {cmd}", flush=True)
    return 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
