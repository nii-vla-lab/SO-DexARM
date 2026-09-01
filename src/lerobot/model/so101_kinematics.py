#!/usr/bin/env python

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SO101_ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
SO101_ARM_ACTION_KEYS = tuple(f"{joint}.pos" for joint in SO101_ARM_JOINTS)
STS3215_MAX_POSITION = 4095


@dataclass(frozen=True)
class TcpOffset:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def as_transform(self) -> np.ndarray:
        return transform_from_xyz_rpy(self.x, self.y, self.z, self.roll, self.pitch, self.yaw)

    def as_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
        }


@dataclass(frozen=True)
class SO101IKResult:
    success: bool
    joint_degrees: dict[str, float]
    target_flange_pose: np.ndarray
    projected: bool = False
    reason: str = ""


def transform_from_xyz_rpy(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_matrix_from_rpy(roll, pitch, yaw)
    transform[:3, 3] = [x, y, z]
    return transform


def transform_from_xyz_quat(
    x: float,
    y: float,
    z: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_matrix_from_quat(qx, qy, qz, qw)
    transform[:3, 3] = [x, y, z]
    return transform


def pose_to_dict(transform: np.ndarray) -> dict[str, object]:
    roll, pitch, yaw = rpy_from_rotation_matrix(transform[:3, :3])
    return {
        "translation": {
            "x": float(transform[0, 3]),
            "y": float(transform[1, 3]),
            "z": float(transform[2, 3]),
        },
        "rpy": {"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)},
        "matrix": transform.tolist(),
    }


def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def rotation_matrix_from_quat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        return np.eye(3, dtype=float)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def rpy_from_rotation_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=float)
    rotation = transform[:3, :3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ transform[:3, 3])
    return inverse


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle_rad(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


class SO101Kinematics:
    """SO-101 analytical kinematics based on XLeRobot's public 2-link solver.

    XLeRobot's published solver solves only the vertical shoulder/elbow plane and
    returns LeRobot-style joint angles in degrees for ``shoulder_lift`` and
    ``elbow_flex``.  This wrapper keeps those equations and adds the remaining
    SO-101 arm joints for practical 5-axis end-effector control:

    1. ``shoulder_pan`` tracks horizontal bearing.
    2. The XLeRobot planar IK tracks radial/vertical TCP position.
    3. ``wrist_flex`` tracks palm pitch relative to shoulder/elbow posture.
    4. ``wrist_roll`` tracks palm roll.
    """

    def __init__(self, l1: float = 0.1159, l2: float = 0.1350):
        self.l1 = l1
        self.l2 = l2
        self.theta1_offset = math.atan2(0.028, 0.11257)
        self.theta2_offset = math.atan2(0.0052, 0.1349) + self.theta1_offset

    @property
    def max_planar_reach(self) -> float:
        return self.l1 + self.l2

    @property
    def min_planar_reach(self) -> float:
        return abs(self.l1 - self.l2)

    def planar_reach_status(self, x: float, y: float) -> tuple[bool, str]:
        radius = math.hypot(x, y)
        if radius > self.max_planar_reach + 1e-9:
            return False, "target beyond SO-101 planar reach"
        if 0.0 < radius < self.min_planar_reach - 1e-9:
            return False, "target inside SO-101 planar minimum reach"
        return True, ""

    def inverse_kinematics(
        self,
        x: float,
        y: float,
        l1: float | None = None,
        l2: float | None = None,
        *,
        clamp_to_workspace: bool = True,
    ) -> tuple[float, float]:
        """Solve XLeRobot's SO-101/SO-100 2D planar IK.

        Args:
            x: Planar forward/radial coordinate in meters.
            y: Planar vertical coordinate in meters.
            l1: Upper-arm link length in meters.
            l2: Lower-arm link length in meters.
            clamp_to_workspace: Match XLeRobot examples when true by projecting
                unreachable planar targets to the closest radial boundary.

        Returns:
            ``(shoulder_lift_deg, elbow_flex_deg)`` in degrees.
        """

        l1 = self.l1 if l1 is None else l1
        l2 = self.l2 if l2 is None else l2
        radius = math.hypot(x, y)
        max_radius = l1 + l2
        min_radius = abs(l1 - l2)

        if radius > max_radius:
            if not clamp_to_workspace:
                raise ValueError("target beyond SO-101 planar reach")
            scale = max_radius / radius
            x *= scale
            y *= scale
            radius = max_radius
        if 0.0 < radius < min_radius:
            if not clamp_to_workspace:
                raise ValueError("target inside SO-101 planar minimum reach")
            scale = min_radius / radius
            x *= scale
            y *= scale
            radius = min_radius

        cos_theta2 = -((radius * radius - l1 * l1 - l2 * l2) / (2.0 * l1 * l2))
        cos_theta2 = clamp(cos_theta2, -1.0, 1.0)
        theta2 = math.pi - math.acos(cos_theta2)
        beta = math.atan2(y, x)
        gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
        theta1 = beta + gamma

        joint2 = clamp(theta1 + self.theta1_offset, -0.1, 3.45)
        joint3 = clamp(theta2 + self.theta2_offset, -0.2, math.pi)

        joint2_deg = 90.0 - math.degrees(joint2)
        joint3_deg = math.degrees(joint3) - 90.0
        return joint2_deg, joint3_deg

    def forward_kinematics(
        self,
        joint2_deg: float,
        joint3_deg: float,
        l1: float | None = None,
        l2: float | None = None,
    ) -> tuple[float, float]:
        """Return XLeRobot planar ``(radial_x, vertical_y)`` in meters."""

        l1 = self.l1 if l1 is None else l1
        l2 = self.l2 if l2 is None else l2
        joint2_rad = math.radians(90.0 - joint2_deg)
        joint3_rad = math.radians(joint3_deg + 90.0)
        theta1 = joint2_rad - self.theta1_offset
        theta2 = joint3_rad - self.theta2_offset
        # Second-link absolute angle is (theta1 - theta2), matching inverse_kinematics()'s
        # convention (theta1 = beta + gamma, gamma measured from link1 toward the EE).
        # The previous (theta1 + theta2 - π) MIRRORED the elbow, so forward_kinematics() and
        # inverse_kinematics() were NOT inverses: IK(FK(sh,el)) returned el' = C - el (the elbow
        # flipped about ~16°). Because start_ee is built from FK while the live command is built
        # from IK, that mismatch jumped the arm to a wrong elbow pose the instant hand-control
        # engaged and distorted every delta (mirrored-elbow Jacobian) → "雑" tracking. With
        # (theta1 - theta2) the two are exact inverses (verified round-trip), so IK(start_ee)==start.
        x = l1 * math.cos(theta1) + l2 * math.cos(theta1 - theta2)
        y = l1 * math.sin(theta1) + l2 * math.sin(theta1 - theta2)
        return x, y

    def forward_tcp_pose(
        self, joint_degrees: Mapping[str, float], tcp_offset: TcpOffset | None = None
    ) -> np.ndarray:
        pan = math.radians(float(joint_degrees.get("shoulder_pan", 0.0)))
        shoulder = float(joint_degrees.get("shoulder_lift", 0.0))
        elbow = float(joint_degrees.get("elbow_flex", 0.0))
        wrist_flex = float(joint_degrees.get("wrist_flex", 0.0))
        wrist_roll = float(joint_degrees.get("wrist_roll", 0.0))
        radial, vertical = self.forward_kinematics(shoulder, elbow)
        pitch = math.radians(wrist_flex + shoulder + elbow)
        roll = math.radians(wrist_roll)

        flange = np.eye(4, dtype=float)
        flange[:3, :3] = rotation_matrix_from_rpy(roll, pitch, pan)
        flange[:3, 3] = [radial * math.cos(pan), radial * math.sin(pan), vertical]
        return flange @ (tcp_offset or TcpOffset()).as_transform()

    def solve_tcp_ik(
        self,
        target_tcp_pose: np.ndarray,
        *,
        current_joint_degrees: Mapping[str, float] | None = None,
        tcp_offset: TcpOffset | None = None,
        clamp_to_workspace: bool = False,
    ) -> SO101IKResult:
        tcp_offset = tcp_offset or TcpOffset()
        target_flange_pose = target_tcp_pose @ invert_transform(tcp_offset.as_transform())
        x, y, z = (float(v) for v in target_flange_pose[:3, 3])
        radial = math.hypot(x, y)
        pan_rad = (
            math.atan2(y, x)
            if radial > 1e-9
            else math.radians(float((current_joint_degrees or {}).get("shoulder_pan", 0.0)))
        )

        reachable, reason = self.planar_reach_status(radial, z)
        if not reachable and not clamp_to_workspace:
            return SO101IKResult(
                success=False,
                joint_degrees=dict(current_joint_degrees or {}),
                target_flange_pose=target_flange_pose,
                projected=False,
                reason=reason,
            )

        try:
            shoulder_lift, elbow_flex = self.inverse_kinematics(
                radial,
                z,
                clamp_to_workspace=clamp_to_workspace,
            )
        except ValueError as exc:
            return SO101IKResult(
                success=False,
                joint_degrees=dict(current_joint_degrees or {}),
                target_flange_pose=target_flange_pose,
                projected=False,
                reason=str(exc),
            )

        pan_unrot = rotation_matrix_from_rpy(0.0, 0.0, -pan_rad)
        local_rotation = pan_unrot @ target_flange_pose[:3, :3]
        target_roll, target_pitch, _target_yaw = rpy_from_rotation_matrix(local_rotation)
        wrist_flex = math.degrees(target_pitch) - shoulder_lift - elbow_flex
        wrist_roll = math.degrees(wrap_angle_rad(target_roll))

        return SO101IKResult(
            success=True,
            joint_degrees={
                "shoulder_pan": math.degrees(pan_rad),
                "shoulder_lift": shoulder_lift,
                "elbow_flex": elbow_flex,
                "wrist_flex": wrist_flex,
                "wrist_roll": wrist_roll,
            },
            target_flange_pose=target_flange_pose,
            projected=not reachable,
            reason="" if reachable else reason,
        )


class SO101LeRobotCalibration:
    """Convert XLeRobot/SO-101 degree angles to LeRobot calibrated action units."""

    def __init__(
        self,
        calibration: Mapping[str, Mapping[str, float]] | None = None,
        *,
        action_units: str = "range_m100_100",
        max_position: int = STS3215_MAX_POSITION,
    ):
        self.calibration = dict(calibration or {})
        self.action_units = action_units
        self.max_position = max_position

    @classmethod
    def from_file(
        cls,
        path: Path | str | None,
        *,
        action_units: str = "range_m100_100",
        max_position: int = STS3215_MAX_POSITION,
    ) -> SO101LeRobotCalibration:
        if path is None:
            return cls(action_units=action_units, max_position=max_position)
        cal_path = Path(path)
        if not cal_path.exists():
            return cls(action_units=action_units, max_position=max_position)
        with cal_path.open("r", encoding="utf-8") as input_file:
            data = json.load(input_file)
        return cls(
            data if isinstance(data, dict) else {}, action_units=action_units, max_position=max_position
        )

    @property
    def has_calibration(self) -> bool:
        return all(joint in self.calibration for joint in SO101_ARM_JOINTS)

    def action_to_degrees(self, joint: str, value: float) -> float:
        if self.action_units == "degrees":
            return float(value)
        entry = self.calibration.get(joint)
        if not entry:
            return float(value)
        span = float(entry["range_max"]) - float(entry["range_min"])
        if span == 0.0:
            return float(value)
        normalized = float(value)
        if self.action_units == "range_m100_100":
            return normalized * span * 360.0 / (200.0 * self.max_position)
        raise ValueError(f"Unsupported SO-101 action units: {self.action_units}")

    def degrees_to_action(self, joint: str, degrees: float) -> float:
        if self.action_units == "degrees":
            return float(degrees)
        entry = self.calibration.get(joint)
        if not entry:
            return float(degrees)
        span = float(entry["range_max"]) - float(entry["range_min"])
        if span == 0.0:
            return float(degrees)
        if self.action_units == "range_m100_100":
            return float(degrees) * 200.0 * self.max_position / (360.0 * span)
        raise ValueError(f"Unsupported SO-101 action units: {self.action_units}")

    def action_dict_to_degrees(self, action: Mapping[str, float]) -> dict[str, float]:
        return {
            joint: self.action_to_degrees(joint, float(action.get(f"{joint}.pos", action.get(joint, 0.0))))
            for joint in SO101_ARM_JOINTS
        }

    def degrees_dict_to_action(self, joint_degrees: Mapping[str, float]) -> dict[str, float]:
        return {
            f"{joint}.pos": self.degrees_to_action(joint, float(joint_degrees[joint]))
            for joint in SO101_ARM_JOINTS
            if joint in joint_degrees
        }

    def action_limits(self, joint: str) -> tuple[float, float]:
        if self.action_units == "degrees":
            entry = self.calibration.get(joint)
            if not entry:
                return (-180.0, 180.0)
            span = float(entry["range_max"]) - float(entry["range_min"])
            limit = span * 360.0 / (2.0 * self.max_position)
            return (-limit, limit)
        if self.action_units == "range_m100_100":
            return (-100.0, 100.0)
        raise ValueError(f"Unsupported SO-101 action units: {self.action_units}")

    def clamp_action(
        self, action: Mapping[str, float]
    ) -> tuple[dict[str, float], dict[str, tuple[float, float, float]]]:
        clamped: dict[str, float] = {}
        clamp_log: dict[str, tuple[float, float, float]] = {}
        for key, value in action.items():
            joint = key.removesuffix(".pos")
            if joint not in SO101_ARM_JOINTS:
                clamped[key] = float(value)
                continue
            low, high = self.action_limits(joint)
            bounded = clamp(float(value), low, high)
            clamped[key] = bounded
            if abs(bounded - float(value)) > 1e-9:
                clamp_log[key] = (float(value), low, high)
        return clamped, clamp_log


def load_tcp_offset_yaml(path: Path | str | None, overrides: Mapping[str, float] | None = None) -> TcpOffset:
    data: dict = {}
    if path is not None and Path(path).exists():
        import yaml

        with Path(path).open("r", encoding="utf-8") as input_file:
            loaded = yaml.safe_load(input_file)
        data = loaded if isinstance(loaded, dict) else {}
    tcp = data.get("tcp_offset", {}) if isinstance(data.get("tcp_offset", {}), dict) else {}
    values = {
        "x": float(tcp.get("x", 0.0)),
        "y": float(tcp.get("y", 0.0)),
        "z": float(tcp.get("z", 0.0)),
        "roll": float(tcp.get("roll", 0.0)),
        "pitch": float(tcp.get("pitch", 0.0)),
        "yaw": float(tcp.get("yaw", 0.0)),
    }
    if overrides:
        for key in values:
            override = overrides.get(key)
            if override is not None:
                values[key] = float(override)
    return TcpOffset(**values)


def vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))
