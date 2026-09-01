"""Hand-calibration helpers for the Quest HTS teleoperators.

Extracted from scripts/quest_hts/real_teleop_common.py when the Quest HTS stack was
integrated into the lerobot package. The calibration YAML maps human finger features
(open/closed poses) to AmazingHand raw motor positions.
"""

from __future__ import annotations

from pathlib import Path

import yaml

FINGER_GROUPS = {
    "finger1": ("index",),
    "finger2": ("middle",),
    "finger3": ("ring", "little"),
    "finger4": ("thumb",),
}


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return min(max(value, min_value), max_value)


def _finger_features_for_side(result, side: str) -> dict:
    side_result = result.right if side == "right" else result.left if side == "left" else None
    if side_result is None or side_result.hand_result is None:
        return {}
    return _serialize_finger_features(side_result.hand_result.finger_features)


def _serialize_finger_features(features: dict) -> dict[str, dict]:
    return {
        name: {
            "tip_distance": float(feature.tip_distance),
            "joint_distances": [float(value) for value in feature.joint_distances],
            "open_close_score": float(feature.open_close_score),
        }
        for name, feature in features.items()
    }


def _load_hand_calibration(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as input_file:
        data = yaml.safe_load(input_file)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def _tip_distance_for_group(features: dict, fingers: tuple[str, ...]) -> float:
    distances = [float(features[name]["tip_distance"]) for name in fingers if name in features]
    if not distances:
        raise ValueError(f"missing finger tip distances for {fingers}")
    return sum(distances) / len(distances)


def _raw_value(mapping: dict, key: str) -> int:
    if key not in mapping:
        raise ValueError(f"hand calibration missing raw position for {key}")
    return int(round(float(mapping[key])))


def _apply_hand_calibration(
    *,
    calibration: dict,
    side: str,
    finger_features: dict,
    hand_flex_gain: float,
    hand_flex_saturation_threshold: float,
) -> tuple[dict[str, float], dict[str, int], dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
    human_poses = calibration.get("human_poses", {})
    robot = calibration.get("robot", {})
    open_features = human_poses.get("open", {}).get("features", {})
    closed_features = human_poses.get("closed", {}).get("features", {})
    open_raw = robot.get("open_raw_positions", {})
    closed_raw = robot.get("closed_raw_positions", {})
    if not open_features or not closed_features:
        raise ValueError("hand calibration requires human open and closed poses.")
    if not open_raw or not closed_raw:
        raise ValueError("hand calibration requires robot open_raw_positions and closed_raw_positions.")

    safe_min = robot.get("safe_min_raw_positions", {})
    safe_max = robot.get("safe_max_raw_positions", {})
    calibrated_action: dict[str, float] = {}
    target_raw: dict[str, int] = {}
    raw_human_flex_by_key: dict[str, float] = {}
    adjusted_human_flex_by_key: dict[str, float] = {}
    calibration_flex_debug_by_key: dict[str, dict[str, float]] = {}
    prefix = "l" if side == "left" else "r"
    for finger_name, source_fingers in FINGER_GROUPS.items():
        open_tip_distance = _tip_distance_for_group(open_features, source_fingers)
        closed_tip_distance = _tip_distance_for_group(closed_features, source_fingers)
        current_tip_distance = _tip_distance_for_group(finger_features, source_fingers)
        denominator = open_tip_distance - closed_tip_distance
        raw_human_flex = (
            0.0
            if abs(denominator) <= 1e-9
            else _clamp((open_tip_distance - current_tip_distance) / denominator, 0.0, 1.0)
        )
        adjusted_human_flex = raw_human_flex * hand_flex_gain
        if adjusted_human_flex >= hand_flex_saturation_threshold:
            adjusted_human_flex = 1.0
        adjusted_human_flex = _clamp(adjusted_human_flex, 0.0, 1.0)
        for motor in (1, 2):
            key = f"{prefix}_{finger_name}_motor{motor}.pos"
            robot_open_raw = _raw_value(open_raw, key)
            robot_closed_raw = _raw_value(closed_raw, key)
            raw = robot_open_raw + adjusted_human_flex * (robot_closed_raw - robot_open_raw)
            min_raw = int(safe_min.get(key, min(robot_open_raw, robot_closed_raw)))
            max_raw = int(safe_max.get(key, max(robot_open_raw, robot_closed_raw)))
            target_raw[key] = int(round(_clamp(raw, min_raw, max_raw)))
            calibrated_action[key] = adjusted_human_flex * 100.0
            raw_human_flex_by_key[key] = raw_human_flex
            adjusted_human_flex_by_key[key] = adjusted_human_flex
            calibration_flex_debug_by_key[key] = {
                "current_tip_distance": current_tip_distance,
                "open_tip_distance": open_tip_distance,
                "closed_tip_distance": closed_tip_distance,
                "raw_flex": raw_human_flex,
                "adjusted_flex": adjusted_human_flex,
            }
    return (
        calibrated_action,
        target_raw,
        raw_human_flex_by_key,
        adjusted_human_flex_by_key,
        calibration_flex_debug_by_key,
    )
