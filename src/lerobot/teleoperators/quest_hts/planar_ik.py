#!/usr/bin/env python

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import yaml

from lerobot.model.so101_kinematics import SO101Kinematics, SO101LeRobotCalibration

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ARM_ACTION_KEYS = tuple(f"{joint}.pos" for joint in ARM_JOINTS)
HAND_ACTION_KEYS = tuple(f"finger{i}_motor{j}.pos" for i in range(1, 5) for j in range(1, 3))
SIDE_PREFIX = {"right": "r_", "left": "l_"}
DEFAULT_CONTROLLED_JOINTS = ("shoulder_lift.pos", "elbow_flex.pos")
DEFAULT_FIXED_JOINTS = ("shoulder_pan.pos", "wrist_flex.pos", "wrist_roll.pos")
DEFAULT_LINK_LENGTHS = {"l1": 0.1159, "l2": 0.1350}
DEFAULT_SIDE_MAPPING_FILE = Path("scripts/configs/quest_hts_physical_side_mapping.yaml")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def side_names(side: str) -> tuple[str, ...]:
    if side == "both":
        return ("right", "left")
    if side not in {"right", "left"}:
        raise ValueError("--side must be right, left, or both.")
    return (side,)


def side_prefix(side: str) -> str:
    if side not in SIDE_PREFIX:
        raise ValueError("side must be right or left.")
    return SIDE_PREFIX[side]


def prefixed_key(side: str, key: str) -> str:
    key = strip_side_prefix(key)
    return f"{side_prefix(side)}{key}"


def strip_side_prefix(key: str) -> str:
    if key.startswith(("r_", "l_")):
        return key[2:]
    return key


def prefixed_action(action: Mapping[str, object], side: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in action.items():
        if value is None:
            continue
        if not str(key).endswith(".pos"):
            continue
        output[prefixed_key(side, str(key))] = float(value)
    return output


def unprefixed_action(action: Mapping[str, object], side: str) -> dict[str, float]:
    prefix = side_prefix(side)
    output: dict[str, float] = {}
    for key, value in action.items():
        if value is None:
            continue
        key = str(key)
        if not key.endswith(".pos"):
            continue
        if key.startswith(prefix):
            output[strip_side_prefix(key)] = float(value)
        elif not key.startswith(("r_", "l_")):
            output[key] = float(value)
    return output


def joint_from_action_key(key: str) -> str:
    return strip_side_prefix(key).removesuffix(".pos")


def default_start_pose_file(side: str) -> Path:
    if side == "both":
        return Path(".cache/so_dexarm/quest_hts_start_pose_dual_arm.yaml")
    return Path(f".cache/so_dexarm/quest_hts_start_pose_{side}.yaml")


def default_constraints_file(side: str) -> Path:
    if side == "both":
        return Path("scripts/configs/quest_hts_dual_arm_so101_constraints.yaml")
    return Path(f"scripts/configs/quest_hts_{side}_so101_constraints.yaml")


def default_arm_calibration_dir(side: str) -> Path:
    return Path(f".cache/calibration/robots/so101_amazinghand_{side}")


def default_hand_calibration_file(side: str) -> Path:
    return Path(f".cache/so_dexarm/quest_hts_{side}_hand_calibration.yaml")


def default_side_mapping_file() -> Path:
    return DEFAULT_SIDE_MAPPING_FILE


def default_robot_port(side: str) -> str:
    return "/dev/ttyso101_amazinghand_r_arm" if side == "right" else "/dev/ttyso101_amazinghand_l_arm"


def default_hand_port(side: str) -> str:
    return "/dev/ttyso101_amazinghand_r_hand" if side == "right" else "/dev/ttyso101_amazinghand_l_hand"


def calibration_file_from_dir(path: Path | str) -> Path:
    path = Path(path)
    if path.suffix == ".json":
        return path
    return path / "right.json"


def read_yaml(path: Path | str) -> dict:
    with Path(path).open("r", encoding="utf-8") as input_file:
        data = yaml.safe_load(input_file)
    return data if isinstance(data, dict) else {}


def write_yaml(path: Path | str, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(dict(payload), output_file, sort_keys=False)


def build_side_mapping_payload(
    *,
    physical_right_raw_label: str | None,
    physical_left_raw_label: str | None,
    created_at: str | None = None,
) -> dict[str, object]:
    mapping: dict[str, object] = {}
    if physical_right_raw_label is not None:
        mapping["physical_right_raw_label"] = physical_right_raw_label
    if physical_left_raw_label is not None:
        mapping["physical_left_raw_label"] = physical_left_raw_label
    return {
        "version": 1,
        "created_at": created_at or now_iso(),
        "canonical_convention": {
            "right": "physical human RIGHT hand controls physical RIGHT robot",
            "left": "physical human LEFT hand controls physical LEFT robot",
        },
        "transport_mapping": mapping,
    }


def load_side_mapping(path: Path | str) -> dict[str, object]:
    payload = read_yaml(path)
    mapping = payload.get("transport_mapping", {})
    if not isinstance(mapping, dict):
        raise ValueError(f"side mapping file {path} must contain transport_mapping.")
    return payload


def raw_label_for_physical_side(mapping_payload: Mapping[str, object], side: str) -> str:
    if side not in {"right", "left"}:
        raise ValueError("physical side must be right or left.")
    mapping = mapping_payload.get("transport_mapping", {})
    if not isinstance(mapping, dict):
        raise ValueError("side mapping payload must contain transport_mapping.")
    key = f"physical_{side}_raw_label"
    label = mapping.get(key)
    if label not in {"left", "right"}:
        raise ValueError(f"side mapping missing {key}=left|right.")
    return str(label)


def lookup_action_value(
    action: Mapping[str, object], key: str, side: str, default: float | None = None
) -> float | None:
    candidates = (key, strip_side_prefix(key), prefixed_key(side, key))
    for candidate in candidates:
        if candidate in action and action[candidate] is not None:
            return float(action[candidate])
    return default


def normalize_arm_action(action: Mapping[str, object], side: str) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key in ARM_ACTION_KEYS:
        value = lookup_action_value(action, key, side)
        if value is not None:
            normalized[key] = value
    return normalized


def prefixed_arm_action(action: Mapping[str, object], side: str) -> dict[str, float | None]:
    return {prefixed_key(side, key): lookup_action_value(action, key, side) for key in ARM_ACTION_KEYS}


def prefixed_hand_action(action: Mapping[str, object], side: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in HAND_ACTION_KEYS:
        value = lookup_action_value(action, key, side)
        if value is not None:
            output[prefixed_key(side, key)] = value
    return output


def load_side_calibration(
    path: Path | str | None, side: str, *, action_units: str = "range_m100_100"
) -> SO101LeRobotCalibration:
    if path is None or not Path(path).exists():
        return SO101LeRobotCalibration(action_units=action_units)
    with Path(path).open("r", encoding="utf-8") as input_file:
        loaded = json.load(input_file)
    data = loaded if isinstance(loaded, dict) else {}
    if any(key.startswith(side_prefix(side)) for key in data):
        data = {
            strip_side_prefix(key): value for key, value in data.items() if key.startswith(side_prefix(side))
        }
    return SO101LeRobotCalibration(data, action_units=action_units)


def planar_ee_from_action(
    action: Mapping[str, object],
    side: str,
    *,
    calibration_file: Path | str | None,
    action_units: str = "range_m100_100",
    l1: float = DEFAULT_LINK_LENGTHS["l1"],
    l2: float = DEFAULT_LINK_LENGTHS["l2"],
) -> dict[str, float]:
    calibration = load_side_calibration(calibration_file, side, action_units=action_units)
    normalized = normalize_arm_action(action, side)
    joint_degrees = calibration.action_dict_to_degrees(normalized)
    x, y = SO101Kinematics(l1=l1, l2=l2).forward_kinematics(
        joint_degrees.get("shoulder_lift", 0.0),
        joint_degrees.get("elbow_flex", 0.0),
    )
    return {"x": float(x), "y": float(y)}


def wrist_to_payload(
    wrist: Sequence[float] | None,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    if wrist is None:
        return (
            {"x": None, "y": None, "z": None},
            {"x": None, "y": None, "z": None, "w": None},
        )
    values = [float(value) for value in wrist]
    return (
        {"x": values[0], "y": values[1], "z": values[2]},
        {"x": values[3], "y": values[4], "z": values[5], "w": values[6]},
    )


def landmarks_payload(landmarks: Sequence[float] | None) -> dict[str, object]:
    return {"values": [] if landmarks is None else [float(value) for value in landmarks]}


def build_start_pose_side(
    side: str,
    *,
    physical_hand: str | None = None,
    bound_input_label: str | None = None,
    human_side: str | None = None,
    robot_side: str | None = None,
    input_assignment_mode: str | None = None,
    physical_side_verified: bool | None = None,
    assignment_assumption: str | None = None,
    side_mapping_file: Path | str | None = None,
    input_stream: Mapping[str, object] | None = None,
    transport_raw_label: str | None = None,
    include_transport_debug: bool = False,
    wrist: Sequence[float] | None,
    landmarks: Sequence[float] | None,
    hand_flex_features: Mapping[str, float] | None,
    robot_observation: Mapping[str, object],
    arm_calibration_dir: Path | str,
    arm_id: str,
    hand_calibration_file: Path | str,
    action_units: str = "range_m100_100",
) -> dict[str, object]:
    calibration_file = calibration_file_from_dir(arm_calibration_dir)
    wrist_position, wrist_rotation = wrist_to_payload(wrist)
    arm_action = prefixed_arm_action(robot_observation, side)
    planar_start = planar_ee_from_action(
        arm_action,
        side,
        calibration_file=calibration_file,
        action_units=action_units,
    )
    payload = {
        "human_side": human_side or physical_hand or side,
        "robot_side": robot_side or side,
        "input_assignment_mode": input_assignment_mode,
        "physical_side_verified": physical_side_verified,
        "assignment_assumption": assignment_assumption,
        "side_mapping_file": None if side_mapping_file is None else str(side_mapping_file),
        "input_stream": dict(input_stream or {}),
        "human": {
            "wrist_position": wrist_position,
            "wrist_rotation": wrist_rotation,
            "landmarks_baseline": landmarks_payload(landmarks),
            "hand_flex_baseline": dict(hand_flex_features or {}),
        },
        "robot": {
            "arm_calibration_dir": str(arm_calibration_dir),
            "arm_id": arm_id,
            "action_units": action_units,
            "arm_start_action": arm_action,
            "planar_ee_start": planar_start,
            "fixed_joint_start_action": {
                prefixed_key(side, key): arm_action[prefixed_key(side, key)] for key in DEFAULT_FIXED_JOINTS
            },
            "controlled_joint_start_action": {
                prefixed_key(side, key): arm_action[prefixed_key(side, key)]
                for key in DEFAULT_CONTROLLED_JOINTS
            },
            "hand_calibration_file": str(hand_calibration_file),
            "hand_start_action": prefixed_hand_action(robot_observation, side),
        },
    }
    if include_transport_debug:
        payload["transport_debug"] = {
            f"raw_label_for_{side}": transport_raw_label
            if transport_raw_label is not None
            else bound_input_label,
        }
    return payload


def build_start_pose_payload(
    *,
    name: str,
    sides: Mapping[str, Mapping[str, object]],
    created_at: str | None = None,
) -> dict[str, object]:
    return {
        "version": 1,
        "name": name,
        "created_at": created_at or now_iso(),
        "sides": {side: dict(payload) for side, payload in sides.items()},
    }


def load_start_pose_side(path: Path | str, side: str) -> dict[str, object]:
    payload = read_yaml(path)
    sides = payload.get("sides", {})
    if not isinstance(sides, dict) or side not in sides:
        raise ValueError(f"start pose file {path} does not contain side {side}.")
    side_payload = sides[side]
    if not isinstance(side_payload, dict):
        raise ValueError(f"start pose side {side} is not a mapping.")
    return side_payload


def _limits_for_keys(
    calibration: SO101LeRobotCalibration, side: str, keys: Sequence[str]
) -> dict[str, dict[str, float]]:
    limits: dict[str, dict[str, float]] = {}
    for key in keys:
        joint = joint_from_action_key(key)
        low, high = calibration.action_limits(joint)
        limits[prefixed_key(side, key)] = {"min": float(low), "max": float(high)}
    return limits


def build_constraints_side(
    side: str,
    *,
    calibration_file: Path | str | None,
    controlled_joints: Sequence[str] | None = None,
    fixed_joints: Sequence[str] | None = None,
    action_units: str = "range_m100_100",
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    workspace_margin_m: float = 0.01,
) -> dict[str, object]:
    controlled = tuple(controlled_joints or DEFAULT_CONTROLLED_JOINTS)
    fixed = tuple(fixed_joints or DEFAULT_FIXED_JOINTS)
    all_keys = tuple(prefixed_key(side, key) for key in ARM_ACTION_KEYS)
    calibration = load_side_calibration(calibration_file, side, action_units=action_units)
    full_limits = _limits_for_keys(calibration, side, ARM_ACTION_KEYS)
    return {
        "arm_id": side,
        "action_units": "lerobot_calibrated",
        "arm_action_units": action_units,
        "motor_ids": {
            "shoulder_pan": 11,
            "shoulder_lift": 12,
            "elbow_flex": 13,
            "wrist_flex": 14,
            "wrist_roll": 15,
        },
        "controlled_joints": [prefixed_key(side, key) for key in controlled],
        "fixed_joints": {prefixed_key(side, key): {"source": "start_pose"} for key in fixed},
        "full_calibration_limits": full_limits,
        "task_limits": dict(full_limits),
        "joint_limits": dict(full_limits),
        "all_arm_joints": list(all_keys),
        "planar_ik": {
            "l1": DEFAULT_LINK_LENGTHS["l1"],
            "l2": DEFAULT_LINK_LENGTHS["l2"],
            "human_axis_map": {
                "robot_x_from": "wrist_forward",
                "robot_y_from": "wrist_up",
            },
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "workspace_margin_m": float(workspace_margin_m),
        },
    }


def build_constraints_payload(
    *,
    sides: Mapping[str, Mapping[str, object]],
    preset: str = "planar_ik_record_v1",
) -> dict[str, object]:
    return {
        "version": 1,
        "preset": preset,
        "created_at": now_iso(),
        "sides": {side: dict(payload) for side, payload in sides.items()},
    }


def load_constraints_side(path: Path | str, side: str) -> dict[str, object]:
    payload = read_yaml(path)
    sides = payload.get("sides", {})
    if not isinstance(sides, dict) or side not in sides:
        raise ValueError(f"constraints file {path} does not contain side {side}.")
    side_payload = sides[side]
    if not isinstance(side_payload, dict):
        raise ValueError(f"constraints side {side} is not a mapping.")
    return side_payload


def validate_constraints_payload(payload: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    sides = payload.get("sides")
    if not isinstance(sides, dict):
        return ["constraints payload must contain a sides mapping"]
    for side, side_payload in sides.items():
        if side not in {"right", "left"}:
            errors.append(f"unsupported side: {side}")
            continue
        if not isinstance(side_payload, dict):
            errors.append(f"{side}: side payload must be a mapping")
            continue
        controlled = set(side_payload.get("controlled_joints", []))
        fixed = set((side_payload.get("fixed_joints") or {}).keys())
        expected_controlled = {prefixed_key(side, key) for key in DEFAULT_CONTROLLED_JOINTS}
        expected_fixed = {prefixed_key(side, key) for key in DEFAULT_FIXED_JOINTS}
        if controlled != expected_controlled:
            errors.append(f"{side}: controlled_joints must be {sorted(expected_controlled)}")
        if fixed != expected_fixed:
            errors.append(f"{side}: fixed_joints must be {sorted(expected_fixed)}")
        limits = side_payload.get("task_limits") or side_payload.get("full_calibration_limits")
        if not isinstance(limits, dict):
            errors.append(f"{side}: missing task_limits/full_calibration_limits")
            continue
        for key in expected_controlled | expected_fixed:
            entry = limits.get(key)
            if not isinstance(entry, dict) or "min" not in entry or "max" not in entry:
                errors.append(f"{side}: missing min/max limit for {key}")
    return errors


def load_limits_for_side(constraints: Mapping[str, object], side: str) -> dict[str, tuple[float, float]]:
    source = constraints.get("task_limits") or constraints.get("full_calibration_limits") or {}
    limits: dict[str, tuple[float, float]] = {}
    if not isinstance(source, dict):
        return limits
    for key, entry in source.items():
        if not isinstance(entry, dict):
            continue
        low = entry.get("min")
        high = entry.get("max")
        if low is None or high is None:
            continue
        limits[strip_side_prefix(key)] = (float(low), float(high))
    return limits
