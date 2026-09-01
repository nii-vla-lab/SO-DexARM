#!/usr/bin/env python

from __future__ import annotations

import argparse
import contextlib
import errno
import logging
import math
import socket
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.model.so101_kinematics import (
    SO101Kinematics,
    SO101LeRobotCalibration,
    TcpOffset,
    invert_transform,
    load_tcp_offset_yaml,
    pose_to_dict,
    rotation_matrix_from_rpy,
    rpy_from_rotation_matrix,
    transform_from_xyz_quat,
    vector_norm,
)
from lerobot.teleoperators.quest_hts.dual_arm_mapping import (
    DUAL_ARM_ACTION_KEYS,
    LEROBOT_DUAL_ARM_ACTION_KEYS,
    DualArmDryRunConfig,
    DualArmDryRunMapper,
    SideConfig,
    format_result,
    provisional_to_lerobot_action,
)
from lerobot.teleoperators.quest_hts.hand_calibration import (
    _apply_hand_calibration,
    _finger_features_for_side,
    _load_hand_calibration,
)
from lerobot.teleoperators.quest_hts.hts_protocol import parse_hts_line
from lerobot.teleoperators.quest_hts.planar_ik import (
    load_constraints_side,
    load_limits_for_side,
    load_side_mapping,
    load_start_pose_side,
    normalize_arm_action,
    prefixed_action,
    prefixed_key,
    raw_label_for_physical_side,
    strip_side_prefix,
    unprefixed_action,
)
from lerobot.teleoperators.quest_hts.right_arm_mapping import RightArmMappingConfig
from lerobot.teleoperators.quest_hts.right_hand_mapping import RightHandMappingConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_quest_hts import QuestHTSRightTeleoperatorConfig, QuestHTSTeleoperatorConfig

logger = logging.getLogger(__name__)

RIGHT_ARM_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
)
RIGHT_HAND_KEYS = tuple(f"finger{i}_motor{j}.pos" for i in range(1, 5) for j in range(1, 3))
RIGHT_ACTION_KEYS = RIGHT_ARM_KEYS + RIGHT_HAND_KEYS
RIGHT_CANONICAL_ARM_KEYS = tuple(prefixed_key("right", key) for key in RIGHT_ARM_KEYS)
RIGHT_CANONICAL_HAND_KEYS = tuple(prefixed_key("right", key) for key in RIGHT_HAND_KEYS)
RIGHT_CANONICAL_ACTION_KEYS = RIGHT_CANONICAL_ARM_KEYS + RIGHT_CANONICAL_HAND_KEYS
HTS_RAW_LABELS = ("left", "right")


def _canonical_keys(keys: tuple[str, ...], side: str = "right") -> tuple[str, ...]:
    return tuple(prefixed_key(side, key) for key in keys)


class QuestHTSTeleoperator(Teleoperator):
    """LeRobot Teleoperator for Quest Hand Tracking Streamer.

    This class consumes HTS TCP CSV lines and returns LeRobot dual_arm
    SO-101 + AmazingHand action keys. It does not connect to robot hardware.
    """

    config_class = QuestHTSTeleoperatorConfig
    name = "quest_hts"

    def __init__(self, config: QuestHTSTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self._mapper = DualArmDryRunMapper(_make_mapper_config(config))
        self._lock = threading.Lock()
        self._connected = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._server_sock: socket.socket | None = None
        self._last_update_s = 0.0
        self._action_count = 0
        self._received_counts: dict[tuple[str, str], int] = {}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(LEROBOT_DUAL_ARM_ACTION_KEYS, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._stop_event.clear()
        if self.config.start_receiver:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((self.config.host, self.config.port))
                server.listen(1)
                server.settimeout(0.5)
            except OSError:
                server.close()
                raise
            self._server_sock = server
            self._thread = threading.Thread(
                target=self._tcp_server_loop,
                args=(server,),
                daemon=True,
                name="quest_hts_tcp",
            )
            self._thread.start()
            self._emit("TCP server listening on %s:%d", self.config.host, self.config.port, force_print=True)
        self._connected = True
        logger.info("%s connected (dry-run HTS teleoperator)", self)

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._stop_event.set()
        if self._server_sock is not None:
            with contextlib.suppress(OSError):
                self._server_sock.close()
            self._server_sock = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._connected = False
        logger.info("%s disconnected", self)

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        with self._lock:
            combined_action = self._mapper.combined_action()
            action = provisional_to_lerobot_action(combined_action)
            action_dict = {key: float(value) for key, value in action.items()}
            self._action_count += 1
            if self.config.print_debug:
                self._emit("action count=%d", self._action_count)
                self._emit("combined26=%s", _format_combined_action(combined_action))
                self._emit("26D action summary=%s", _format_action(action_dict))
            return action_dict

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    def handle_hts_line(self, line: str) -> bool:
        """Inject one HTS CSV line. Useful for tests and non-socket dry-runs."""
        with self._lock:
            parsed = parse_hts_line(line)
            if parsed is None:
                return False
            side, kind, values = parsed
            self._received_counts[(side, kind)] = self._received_counts.get((side, kind), 0) + 1
            if self.config.print_debug:
                self._emit(
                    "received %s %s count=%d",
                    side.title(),
                    kind,
                    self._received_counts[(side, kind)],
                )
            result = self._mapper.update_from_parsed(side, kind, values)
            if result is None:
                return False
            self._last_update_s = time.monotonic()
            if self.config.print_debug:
                self._emit("%s", format_result(result))
            return True

    @property
    def last_update_s(self) -> float:
        return self._last_update_s

    @property
    def action_count(self) -> int:
        return self._action_count

    def _emit(self, message: str, *args: object, force_print: bool = False) -> None:
        text = message % args if args else message
        logger.info(text)
        if force_print or self.config.print_debug:
            print(text, flush=True)

    def _tcp_server_loop(self, server: socket.socket) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    if self.config.print_debug:
                        self._emit("TCP accept waiting on %s:%d", self.config.host, self.config.port)
                    conn, addr = server.accept()
                except TimeoutError:
                    if self.config.print_debug:
                        self._emit("TCP accept timeout")
                    continue
                except OSError:
                    break

                self._emit("HTS client connected from %s", addr)
                self._handle_connection(conn)
        except OSError as exc:
            logger.error("Quest HTS TCP server stopped after socket error: %s", exc)
        finally:
            with contextlib.suppress(OSError):
                server.close()
            if self._server_sock is server:
                self._server_sock = None

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(0.5)
            buffer = ""
            while not self._stop_event.is_set():
                try:
                    if self.config.print_debug:
                        self._emit("TCP recv waiting")
                    data = conn.recv(8192)
                except TimeoutError:
                    if self.config.print_debug:
                        self._emit("TCP recv timeout")
                    continue
                except OSError as exc:
                    logger.warning("Connection error: %s", exc)
                    break
                if not data:
                    if self.config.print_debug:
                        self._emit("TCP recv returned EOF")
                    break
                if self.config.print_debug:
                    self._emit("TCP recv returned bytes=%d", len(data))
                try:
                    buffer += data.decode("utf-8")
                except UnicodeDecodeError:
                    logger.warning("Rejected non-UTF-8 HTS packet")
                    continue
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        self.handle_hts_line(line)


class QuestHTSRightTeleoperator(Teleoperator):
    """Right-side QuestHTS teleoperator that emits calibrated LeRobot actions."""

    config_class = QuestHTSRightTeleoperatorConfig
    name = "quest_hts_right"

    def __init__(self, config: QuestHTSRightTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self._lock = threading.Lock()
        self._connected = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._server_sock: socket.socket | None = None
        self._active_conn = None
        self._closing_active_conn_reason: str | None = None
        self._receiver_alive = False
        self._client_connected = False
        self._client_connect_count = 0
        self._last_receiver_error: str | None = None
        self._last_stale_warning_print_s = 0.0
        self._stream_stale_count = 0
        self._last_stream_stale_seq: int | None = None
        self._reconnect_count = 0
        self._human_side = "right"
        self._robot_side = "right"
        self._mapper = DualArmDryRunMapper()
        self._hand_calibration = _load_hand_calibration(config.right_hand_calibration_file)
        self._raw_inputs = {label: _new_raw_input_state() for label in HTS_RAW_LABELS}
        self._wrist_history: list[tuple[float, ...]] = []
        self._latest_wrist: tuple[float, ...] | None = None
        self._latest_wrist_s: float | None = None
        self._latest_landmarks: tuple[float, ...] | None = None
        self._latest_landmarks_s: float | None = None
        self._latest_seq = 0
        self._wrist_baseline: tuple[float, ...] | None = None
        self._bound_input_label: str | None = self._initial_bound_input_label(config)
        self._binding_completed_at: str | None = _now_iso() if self._bound_input_label is not None else None
        self._binding_confidence: float | None = 1.0 if self._bound_input_label is not None else None
        self._binding_stats: dict[str, object] = {}
        self._binding_error: str | None = None
        self._live_side_confirmed = (
            True if self._is_native_right_mode() else not config.require_live_side_confirmation
        )
        self._live_side_confirmation_stats: dict[str, object] = {}
        self._live_side_confirmation_error: str | None = None
        self._live_side_confirmation_completed_at: str | None = None
        self._single_visible_warning_printed = False
        self._single_visible_assignment_completed_at: str | None = None
        self._single_visible_assignment_error: str | None = None
        self._episode_abort_reason: str | None = None
        self._baseline_completed_at: str | None = None
        self._human_wrist_baseline_pose: np.ndarray | None = None
        self._robot_tcp_baseline_pose: np.ndarray | None = None
        self._ee_baseline_captured = False
        self._neutral_action: dict[str, float] | None = None
        self._last_feedback: dict[str, float] = {}
        self._smoothed_arm_delta = dict.fromkeys(RIGHT_ARM_KEYS, 0.0)
        self._previous_arm_target: dict[str, float] | None = None
        self._kinematics = SO101Kinematics()
        self._tcp_offset = self._load_tcp_offset()
        self._calibration = SO101LeRobotCalibration.from_file(
            config.robot_calibration_file,
            action_units=config.arm_action_units,
        )
        self._workspace_limits = _load_workspace_limits(config.workspace_limits_file)
        self._last_valid_ik_action: dict[str, float] | None = None
        self._last_controlled_joint_action: dict[str, float] | None = None
        self._last_target_tcp_pose: np.ndarray | None = None
        self._last_ik_joint_degrees: dict[str, float] | None = None
        self._ik_failure_count = 0
        self._joint_limit_clamp_count = 0
        self._phase28_side = self._robot_side
        self._planar_start_pose_warnings: list[str] = []
        self._phase28_start_pose = self._load_phase28_start_pose()
        self._phase28_constraints = self._load_phase28_constraints()
        self._planar_start_action = self._load_phase28_start_action()
        self._planar_start_ee = self._load_phase28_start_ee()
        self._planar_controlled_joints = self._load_phase28_controlled_joints()
        self._planar_fixed_joints = self._load_phase28_fixed_joints()
        self._planar_joint_limits = self._load_phase28_joint_limits()
        self._planar_ik_config = self._load_phase28_planar_ik_config()
        self._planar_live_human_baseline: np.ndarray | None = None
        self._last_planar_ee_target: dict[str, float] | None = None
        self._start_pose_mismatch_abort: bool = False
        self._start_pose_mismatch_details: list[str] = []
        self._last_hand_joint_action: dict[str, float] | None = None
        self._hand_start_action: dict[str, float] | None = None
        self.last_action_debug: dict = {}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self._mode_keys(canonical=True), float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return dict.fromkeys(RIGHT_CANONICAL_ACTION_KEYS, float)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        if self.config.mode == "arm-only":
            return True
        return bool(self._hand_calibration)

    @staticmethod
    def _initial_bound_input_label(config: QuestHTSRightTeleoperatorConfig) -> str | None:
        if config.input_assignment_mode == "native_right":
            return "right"
        if config.input_assignment_mode == "single_visible_right":
            return None
        if config.binding_mode == "explicit":
            return config.bound_input_label or config.input_hand
        if config.binding_mode == "saved_map" and Path(config.side_mapping_file).exists():
            return raw_label_for_physical_side(
                load_side_mapping(config.side_mapping_file), config.physical_hand
            )
        return None

    def _is_single_visible_right_mode(self) -> bool:
        return self.config.input_assignment_mode == "single_visible_right"

    def _is_native_right_mode(self) -> bool:
        return self.config.input_assignment_mode == "native_right"

    def _physical_side_verified(self) -> bool:
        if self._is_native_right_mode():
            return True
        if self._is_single_visible_right_mode():
            return False
        return self._live_side_confirmed

    def _active_hts_wrist_label(self) -> str | None:
        label = self._active_input_label_locked()
        return None if label is None else f"{label.title()} wrist"

    def _active_hts_landmarks_label(self) -> str | None:
        label = self._active_input_label_locked()
        return None if label is None else f"{label.title()} landmarks"

    def _stale_timeout_s(self) -> float:
        return float(self.config.stale_timeout_s)

    def reset_input_capture_state(self, *, clear_binding: bool = False) -> None:
        """Clear accumulated QuestHTS frames so the next decision uses fresh input only."""
        with self._lock:
            self._reset_input_capture_state_locked(clear_binding=clear_binding)

    def _reset_input_capture_state_locked(self, *, clear_binding: bool = False) -> None:
        self._raw_inputs = {label: _new_raw_input_state() for label in HTS_RAW_LABELS}
        self._wrist_history = []
        self._latest_wrist = None
        self._latest_wrist_s = None
        self._latest_landmarks = None
        self._latest_landmarks_s = None
        self._latest_seq = 0
        self._wrist_baseline = None
        self._binding_stats = {}
        self._binding_error = None
        self._last_stream_stale_seq = None
        if clear_binding:
            self._bound_input_label = None
            self._binding_confidence = None
            self._binding_completed_at = None
        if self._is_native_right_mode():
            self._bound_input_label = "right"
            self._binding_confidence = 1.0
            self._binding_completed_at = _now_iso()
        if self.config.require_live_side_confirmation:
            self._live_side_confirmed = False
            self._live_side_confirmation_stats = {}
            self._live_side_confirmation_error = None
            self._live_side_confirmation_completed_at = None
        if self._is_native_right_mode():
            self._live_side_confirmed = True
        if self._is_single_visible_right_mode():
            self._single_visible_assignment_error = None
            if clear_binding:
                self._single_visible_assignment_completed_at = None

    def _evaluate_expected_fresh_binding_locked(
        self,
        expected_label: str,
    ) -> tuple[bool, dict[str, object], str | None]:
        stats = {label: _binding_side_stats(self._raw_inputs[label]) for label in HTS_RAW_LABELS}
        expected_stats = stats[expected_label]
        min_frames = int(self.config.side_confirmation_min_frames)
        min_movement = float(self.config.min_binding_movement_m)
        if int(expected_stats["frames"]) < min_frames:
            any_frames = any(int(side_stats["frames"]) > 0 for side_stats in stats.values())
            return False, stats, "not_enough_frames" if any_frames else "no_frames"
        if float(expected_stats["movement_m"]) < min_movement:
            return False, stats, "movement_too_small"
        unexpected = [
            label
            for label, side_stats in stats.items()
            if label != expected_label
            and int(side_stats["frames"]) >= min_frames
            and float(side_stats["movement_m"]) >= min_movement
        ]
        if unexpected:
            return False, stats, "unexpected_other_hand_movement"
        return True, stats, None

    def _single_visible_active_labels_locked(self) -> list[str]:
        now = time.monotonic()
        active_window_s = max(float(self.config.hold_last_target_s), 0.5)
        active = []
        for label in HTS_RAW_LABELS:
            last_frame_s = self._raw_inputs[label].get("last_frame_s")
            if last_frame_s is not None and now - float(last_frame_s) <= active_window_s:
                active.append(label)
        return active

    def _single_visible_assignment_state_locked(self) -> tuple[str | None, dict[str, object], str | None]:
        stats = {label: _binding_side_stats(self._raw_inputs[label]) for label in HTS_RAW_LABELS}
        active = self._single_visible_active_labels_locked()
        if len(active) == 1:
            return active[0], stats, None
        if len(active) > 1:
            return None, stats, "multiple_visible_hand_streams"
        any_frames = any(int(side_stats["frames"]) > 0 for side_stats in stats.values())
        return None, stats, "stale_input" if any_frames else "no_fresh_input"

    def _emit_single_visible_warning_once(self) -> None:
        if self._single_visible_warning_printed:
            return
        self._single_visible_warning_printed = True
        print(
            "SINGLE-VISIBLE-RIGHT MODE:\n"
            "Only your physical RIGHT hand may be visible to Quest.\n"
            "Any visible left hand may also control the RIGHT robot because the streamer does not reliably distinguish physical sides in the current setup.",
            flush=True,
        )

    def assign_single_visible_right_input(self, *, reset_capture: bool = False) -> bool:
        if not self._is_single_visible_right_mode():
            return True
        self._emit_single_visible_warning_once()
        if reset_capture:
            with self._lock:
                self._reset_input_capture_state_locked(clear_binding=True)
        deadline = time.monotonic() + float(self.config.side_confirmation_timeout_s)
        last_reason = None
        while time.monotonic() < deadline and not self._stop_event.is_set():
            with self._lock:
                label, stats, reason = self._single_visible_assignment_state_locked()
                self._binding_stats = stats
                self._single_visible_assignment_error = reason
                if label is not None:
                    self._set_bound_input_label_locked(label, 1.0, stats)
                    self._single_visible_assignment_completed_at = _now_iso()
                    self._live_side_confirmed = True
                    self._live_side_confirmation_error = None
                    self._sync_bound_latest_locked()
                    print(
                        "Single visible hand assigned to RIGHT robot input. Keep your LEFT hand out of view.",
                        flush=True,
                    )
                    return True
                if reason == "multiple_visible_hand_streams":
                    print(
                        "More than one visible hand stream detected. Hide your LEFT hand before controlling the RIGHT robot.",
                        flush=True,
                    )
                    break
                last_reason = reason
            time.sleep(0.05)
        self._bound_input_label = None
        self._live_side_confirmed = False
        self._single_visible_assignment_error = last_reason or self._single_visible_assignment_error
        return False

    def _single_visible_runtime_block_locked(self) -> dict[str, object] | None:
        if not self._is_single_visible_right_mode():
            return None
        active = self._single_visible_active_labels_locked()
        if len(active) > 1:
            self._single_visible_assignment_error = "multiple_visible_hand_streams"
            return {
                "action_block_reason": "multiple_visible_hand_streams",
                "single_visible_active_streams": list(active),
                "target_source": "none",
                "ik_success": False,
                "ik_failure": False,
            }
        if self._bound_input_label is None:
            return {
                "action_block_reason": "single_visible_assignment_not_completed",
                "single_visible_active_streams": list(active),
                "target_source": "none",
                "ik_success": False,
                "ik_failure": False,
            }
        if len(active) == 1 and active[0] != self._bound_input_label:
            self._single_visible_assignment_error = "assigned_stream_not_visible"
            return {
                "action_block_reason": "assigned_stream_not_visible",
                "single_visible_active_streams": list(active),
                "target_source": "none",
                "ik_success": False,
                "ik_failure": False,
            }
        return None

    def _stale_runtime_block_locked(self) -> dict[str, object] | None:
        label = self._active_input_label_locked()
        if label is None:
            return None
        raw = self._raw_inputs[label]
        latest_frame_s = raw.get("last_frame_s")
        if latest_frame_s is None:
            return None
        latest_input_age_s = time.monotonic() - float(latest_frame_s)
        if latest_input_age_s <= self._stale_timeout_s():
            return None
        self._mark_stream_stale_locked()
        return {
            **self._receiver_debug_locked(),
            "action_block_reason": "stale_input",
            "latest_input_age_s": latest_input_age_s,
            "target_source": "none",
            "ik_success": False,
            "ik_failure": False,
        }

    def _is_constrained_planar_mode(self) -> bool:
        return self.config.arm_control_mode == "constrained_planar_ik"

    def _load_phase28_start_pose(self) -> dict[str, object]:
        if self.config.start_pose_file is None:
            return {}
        try:
            payload = load_start_pose_side(self.config.start_pose_file, self._phase28_side)
            if "bound_input_label" in payload:
                warning = (
                    f"legacy start pose format contains bound_input_label in {self.config.start_pose_file}; "
                    "recapture with canonical side mapping before production record."
                )
                self._planar_start_pose_warnings.append(warning)
                logger.warning(warning)
            return payload
        except Exception as exc:
            if self._is_constrained_planar_mode():
                raise
            logger.warning("Ignoring Phase28 start pose load failure: %s", exc)
            return {}

    def _load_phase28_constraints(self) -> dict[str, object]:
        if self.config.constraints_file is None:
            return {}
        try:
            return load_constraints_side(self.config.constraints_file, self._phase28_side)
        except Exception as exc:
            if self._is_constrained_planar_mode():
                raise
            logger.warning("Ignoring Phase28 constraints load failure: %s", exc)
            return {}

    def _load_phase28_start_action(self) -> dict[str, float]:
        robot = self._phase28_start_pose.get("robot", {})
        if not isinstance(robot, dict):
            return {}
        arm_start = robot.get("arm_start_action", {})
        return normalize_arm_action(arm_start if isinstance(arm_start, dict) else {}, self._phase28_side)

    def _load_phase28_start_ee(self) -> dict[str, float] | None:
        robot = self._phase28_start_pose.get("robot", {})
        if not isinstance(robot, dict):
            return None
        planar = robot.get("planar_ee_start", {})
        if isinstance(planar, dict) and planar.get("x") is not None and planar.get("y") is not None:
            return {"x": float(planar["x"]), "y": float(planar["y"])}
        if not self._planar_start_action:
            return None
        joint_degrees = self._calibration.action_dict_to_degrees(self._planar_start_action)
        x, y = self._kinematics.forward_kinematics(
            joint_degrees.get("shoulder_lift", 0.0),
            joint_degrees.get("elbow_flex", 0.0),
        )
        return {"x": float(x), "y": float(y)}

    def _load_phase28_controlled_joints(self) -> tuple[str, ...]:
        controlled = self._phase28_constraints.get("controlled_joints", [])
        if not controlled:
            return ("shoulder_lift.pos", "elbow_flex.pos")
        return tuple(strip_side_prefix(str(key)) for key in controlled)

    def _load_phase28_fixed_joints(self) -> tuple[str, ...]:
        fixed = self._phase28_constraints.get("fixed_joints", {})
        if isinstance(fixed, dict) and fixed:
            return tuple(strip_side_prefix(str(key)) for key in fixed)
        return ("shoulder_pan.pos", "wrist_flex.pos", "wrist_roll.pos")

    def _load_phase28_joint_limits(self) -> dict[str, tuple[float, float]]:
        if self._phase28_constraints:
            return load_limits_for_side(self._phase28_constraints, self._phase28_side)
        return {key: self._calibration.action_limits(key.removesuffix(".pos")) for key in RIGHT_ARM_KEYS}

    def _load_phase28_planar_ik_config(self) -> dict[str, object]:
        planar = self._phase28_constraints.get("planar_ik", {})
        if not isinstance(planar, dict):
            planar = {}
        return {
            "l1": float(planar.get("l1", self._kinematics.l1)),
            "l2": float(planar.get("l2", self._kinematics.l2)),
            "scale_x": float(planar.get("scale_x", 1.0)),
            "scale_y": float(planar.get("scale_y", 1.0)),
            "workspace_margin_m": float(planar.get("workspace_margin_m", 0.01)),
            "human_axis_map": planar.get("human_axis_map", {}),
        }

    def calibrate(self) -> None:
        # QuestHTS wrist baseline is captured automatically from live stream.
        if self.config.mode != "arm-only":
            self._hand_calibration = _load_hand_calibration(self.config.right_hand_calibration_file)
        self._tcp_offset = self._load_tcp_offset()
        self._calibration = SO101LeRobotCalibration.from_file(
            self.config.robot_calibration_file,
            action_units=self.config.arm_action_units,
        )

    def configure(self) -> None:
        pass

    def is_control_ready(self) -> bool:
        if self._is_single_visible_right_mode():
            with self._lock:
                if self._single_visible_runtime_block_locked() is not None:
                    return False
                if self._stale_runtime_block_locked() is not None:
                    return False
            if self.config.mode == "hand-only" or self.config.arm_control_mode == "joint_delta":
                return self._bound_input_label is not None
            return self._bound_input_label is not None and self._ee_baseline_captured
        with self._lock:
            if self._stale_runtime_block_locked() is not None:
                return False
        if self.config.require_live_side_confirmation and not self._live_side_confirmed:
            return False
        if self.config.mode == "hand-only" or self.config.arm_control_mode == "joint_delta":
            return self._bound_input_label is not None
        return self._bound_input_label is not None and self._ee_baseline_captured

    def _wait_for_fresh_wrist(self, *, timeout_s: float = 30.0) -> bool:
        """Poll until a fresh wrist input is available. Returns True if received within timeout."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not self._stop_event.is_set():
            with self._lock:
                if self._latest_wrist is not None:
                    return True
            time.sleep(0.05)
        return False

    def prepare_control_session(self, robot=None) -> None:
        """Optional hook used by LeRobot teleoperate/record before sending actions."""
        observation = robot.get_observation() if robot is not None else dict(self._last_feedback)
        self.send_feedback(observation)
        if self._is_single_visible_right_mode():
            self._emit_single_visible_warning_once()
            print(
                "Show only your physical RIGHT hand. Keep your LEFT hand out of view, then press ENTER to continue.",
                flush=True,
            )
            input()
            if not self.assign_single_visible_right_input():
                raise RuntimeError(
                    "Could not assign a single visible hand to the RIGHT robot input. "
                    "Hide your LEFT hand and keep only your physical RIGHT hand visible."
                )
        else:
            self.confirm_live_side_mapping()
        if self.config.mode == "hand-only" or self.config.arm_control_mode == "joint_delta":
            if self._bound_input_label is None:
                self._run_binding_step()
            return
        if self.config.print_current_tcp_pose or self.config.save_current_tcp_as_home:
            current_tcp = self._current_tcp_pose_from_feedback_locked()
            if current_tcp is not None:
                payload = {
                    "version": 1,
                    "side": "right",
                    "ee_frame": "wrist_roll_flange",
                    "tool_name": "amazinghand",
                    "tcp_offset": self._tcp_offset.as_dict(),
                    "tcp_pose": pose_to_dict(current_tcp),
                    "arm_action_units": self.config.arm_action_units,
                    "robot_calibration_file": str(self.config.robot_calibration_file),
                }
                if self.config.print_current_tcp_pose:
                    self._emit("Current right SO-101 AmazingHand TCP pose: %s", payload, force_print=True)
                if self.config.save_current_tcp_as_home:
                    _write_yaml(self.config.home_tcp_pose_file, payload)
                    self._emit(
                        "Saved current right SO-101 AmazingHand TCP home pose to %s",
                        self.config.home_tcp_pose_file,
                        force_print=True,
                    )
        if not self.config.manual_baseline or self._ee_baseline_captured:
            return
        if self._bound_input_label is None:
            self._run_binding_step()
        if self._is_constrained_planar_mode():
            self._warn_if_robot_not_at_planar_start_locked()
            hand_label = _physical_hand_label(self.config.physical_hand)
            if self._is_single_visible_right_mode():
                print(
                    "Hold your physical RIGHT hand and the right SO-101 + AmazingHand at the start pose, then press ENTER to capture baseline.",
                    flush=True,
                )
            else:
                print(
                    f"Hold your physical {hand_label} hand at the saved start pose, then press ENTER to capture the live planar IK baseline.",
                    flush=True,
                )
            input()
            if robot is not None:
                self.send_feedback(robot.get_observation())
            with self._lock:
                self._sync_bound_latest_locked()
                wrist_ready = self._latest_wrist is not None
            if not wrist_ready:
                print(
                    "Waiting for fresh HTS Right wrist / Right landmarks input. "
                    "Connect the Quest streamer before capturing baseline."
                    if self._is_native_right_mode()
                    else "Waiting for fresh wrist input before capturing baseline...",
                    flush=True,
                )
                wrist_ready = self._wait_for_fresh_wrist(timeout_s=30.0)
            if not wrist_ready:
                raise RuntimeError(
                    "Cannot capture constrained planar IK baseline: timed out waiting for fresh wrist pose."
                )
            with self._lock:
                if self._latest_wrist is None:
                    raise RuntimeError(
                        "Cannot capture constrained planar IK baseline: no fresh bound wrist pose received."
                    )
                if not self._capture_constrained_planar_baseline_locked(self._latest_wrist):
                    raise RuntimeError(
                        "Cannot capture constrained planar IK baseline: start pose file is incomplete."
                    )
            print("Constrained planar IK baseline captured. Teleoperation started.", flush=True)
            return
        hand_label = _physical_hand_label(self.config.physical_hand)
        print(
            f"Place the right SO-101 + AmazingHand at the start pose and hold your physical {hand_label} hand at the corresponding neutral pose, then press ENTER to capture the end-effector baseline.",
            flush=True,
        )
        input()
        if robot is not None:
            self.send_feedback(robot.get_observation())
        with self._lock:
            self._sync_bound_latest_locked()
            wrist_ready = self._latest_wrist is not None
        if not wrist_ready:
            print(
                "Waiting for fresh HTS Right wrist / Right landmarks input. "
                "Connect the Quest streamer before capturing baseline."
                if self._is_native_right_mode()
                else "Waiting for fresh wrist input before capturing baseline...",
                flush=True,
            )
            wrist_ready = self._wait_for_fresh_wrist(timeout_s=30.0)
        if not wrist_ready:
            raise RuntimeError(
                "Cannot capture QuestHTS end-effector baseline: timed out waiting for fresh wrist pose."
            )
        with self._lock:
            if self._latest_wrist is None:
                raise RuntimeError(
                    "Cannot capture QuestHTS end-effector baseline: no fresh bound wrist pose received."
                )
            if not self._capture_ee_baseline_locked(self._latest_wrist):
                raise RuntimeError(
                    "Cannot capture QuestHTS end-effector baseline: robot arm feedback is incomplete."
                )
        print("End-effector baseline captured. IK teleoperation started.", flush=True)

    def capture_end_effector_baseline(
        self,
        *,
        wrist: tuple[float, ...] | None = None,
        observation: dict[str, float] | None = None,
    ) -> bool:
        """Capture baseline from tests or non-interactive wrappers without touching hardware."""
        with self._lock:
            if observation is not None:
                observation = unprefixed_action(observation, self._robot_side)
                self._last_feedback.update(
                    {key: float(value) for key, value in observation.items() if key in RIGHT_ACTION_KEYS}
                )
                if self._neutral_action is None and all(key in self._last_feedback for key in RIGHT_ARM_KEYS):
                    self._neutral_action = {
                        key: float(self._last_feedback.get(key, 0.0)) for key in RIGHT_ACTION_KEYS
                    }
            wrist_pose = wrist or self._latest_wrist
            if wrist_pose is None:
                return False
            if self._is_constrained_planar_mode():
                self._warn_if_robot_not_at_planar_start_locked()
                return self._capture_constrained_planar_baseline_locked(wrist_pose)
            return self._capture_ee_baseline_locked(wrist_pose)

    def confirm_live_side_mapping(self) -> bool:
        """Verify the saved/raw binding with fresh motion before robot control starts."""
        if self._is_native_right_mode():
            with self._lock:
                self._bound_input_label = "right"
                self._binding_confidence = 1.0
                self._binding_error = None
                self._live_side_confirmed = True
                self._live_side_confirmation_error = None
                self._live_side_confirmation_completed_at = _now_iso()
                self._sync_bound_latest_locked()
            return True
        if self._is_single_visible_right_mode():
            return self.assign_single_visible_right_input()
        if not self.config.require_live_side_confirmation:
            self._live_side_confirmed = True
            return True
        if self._bound_input_label is None:
            self._run_binding_step()
        with self._lock:
            expected_label = self._bound_input_label
        if expected_label not in HTS_RAW_LABELS:
            raise RuntimeError("Cannot confirm physical side mapping: no saved input label is available.")

        physical_label = _physical_hand_label(self.config.physical_hand)
        robot_label = _physical_hand_label(self.config.physical_hand)
        other_label = "LEFT" if self.config.physical_hand == "right" else "RIGHT"
        print(
            f"Move only your physical {physical_label} hand. Verifying that it controls the {robot_label} robot input. "
            f"Keep your {other_label} hand hidden.",
            flush=True,
        )
        with self._lock:
            self._reset_input_capture_state_locked(clear_binding=False)

        deadline = time.monotonic() + float(self.config.side_confirmation_timeout_s)
        last_reason = None
        while time.monotonic() < deadline and not self._stop_event.is_set():
            with self._lock:
                ok, stats, reason = self._evaluate_expected_fresh_binding_locked(expected_label)
                self._live_side_confirmation_stats = stats
                self._live_side_confirmation_error = reason
                if ok:
                    self._live_side_confirmed = True
                    self._live_side_confirmation_error = None
                    self._live_side_confirmation_completed_at = _now_iso()
                    self._sync_bound_latest_locked()
                    print(
                        f"Physical {physical_label} hand confirmed for physical {robot_label} robot.",
                        flush=True,
                    )
                    return True
                last_reason = reason
            time.sleep(0.05)

        self._live_side_confirmed = False
        self._live_side_confirmation_error = last_reason
        message = (
            f"Physical {physical_label} hand could not be confirmed. "
            f"Refusing to control the {robot_label} robot with an unverified hand mapping."
        )
        print(message, flush=True)
        raise RuntimeError(message if last_reason is None else f"{message} confirmation_reason={last_reason}")

    def get_episode_metadata(self) -> dict[str, object]:
        metadata = {
            "arm_control_mode": self.config.arm_control_mode,
            "arm_calibration_source": "lerobot",
            "hand_calibration_source": "quest_yaml",
            "robot_calibration_file": str(self.config.robot_calibration_file),
            "right_hand_calibration_file": str(self.config.right_hand_calibration_file),
            "human_side": self._human_side,
            "robot_side": self._robot_side,
            "input_assignment_mode": self.config.input_assignment_mode,
            "physical_side_verified": self._physical_side_verified(),
            "canonical_action_keys": list(self._mode_keys(canonical=True)),
            "baseline_completed_at": self._baseline_completed_at,
            "stale_policy": self.config.stale_policy,
            "stale_timeout_s": self.config.stale_timeout_s,
            "rebaseline_after_reconnect": self.config.rebaseline_after_reconnect,
            "episode_abort_reason": self._episode_abort_reason,
            "stream_stale_count": self._stream_stale_count,
            "stale_input_count": self._stream_stale_count,
            "reconnect_count": self._reconnect_count,
            "start_pose_file": None
            if self.config.start_pose_file is None
            else str(self.config.start_pose_file),
            "constraints_file": None
            if self.config.constraints_file is None
            else str(self.config.constraints_file),
            "planar_ik": self._planar_ik_config,
            "planar_workspace_policy": self.config.planar_workspace_policy,
            "planar_ee_start": self._planar_start_ee,
            "planar_ee_target": self._last_planar_ee_target,
            "controlled_joints": [
                prefixed_key(self._robot_side, key) for key in self._planar_controlled_joints
            ],
            "fixed_joints": [prefixed_key(self._robot_side, key) for key in self._planar_fixed_joints],
            "robot_start_action": self._prefix_keyed_mapping(self._planar_start_action),
            "live_human_baseline": None
            if self._planar_live_human_baseline is None
            else pose_to_dict(self._planar_live_human_baseline),
            "start_pose_warnings": list(self._planar_start_pose_warnings),
            "tcp_config_file": str(self.config.tcp_config_file),
            "tcp_offset": self._tcp_offset.as_dict(),
            "baseline_human_wrist_pose": None
            if self._human_wrist_baseline_pose is None
            else pose_to_dict(self._human_wrist_baseline_pose),
            "baseline_robot_tcp_pose": None
            if self._robot_tcp_baseline_pose is None
            else pose_to_dict(self._robot_tcp_baseline_pose),
            "target_tcp_pose": None
            if self._last_target_tcp_pose is None
            else pose_to_dict(self._last_target_tcp_pose),
            "ik_output_joint_target": self._last_ik_joint_degrees,
            "ik_failure_count": self._ik_failure_count,
            "hand_disabled_status": False,
            "hand_errors": [],
        }
        if self._is_native_right_mode():
            metadata.update(
                {
                    "active_hts_wrist_label": "Right wrist",
                    "active_hts_landmarks_label": "Right landmarks",
                    **self._receiver_debug_locked(),
                }
            )
        else:
            metadata.update(
                {
                    "assignment_assumption": (
                        "only_visible_hand_is_physical_right"
                        if self._is_single_visible_right_mode()
                        else None
                    ),
                    "side_mapping_file": None
                    if self._is_single_visible_right_mode()
                    else str(self.config.side_mapping_file),
                    "side_mapping_ready": self._bound_input_label is not None,
                    "binding_mode": self.config.binding_mode,
                    "binding_confidence": self._binding_confidence,
                    "binding_completed_at": self._binding_completed_at,
                    "require_live_side_confirmation": self.config.require_live_side_confirmation,
                    "live_side_confirmed": self._live_side_confirmed,
                    "live_side_confirmation_completed_at": self._live_side_confirmation_completed_at,
                    "live_side_confirmation_error": self._live_side_confirmation_error,
                    "single_visible_assignment_completed_at": self._single_visible_assignment_completed_at,
                    "single_visible_assignment_error": self._single_visible_assignment_error,
                }
            )
            if self.config.print_debug_transport_labels:
                metadata["transport_debug"] = {
                    f"raw_label_for_{self._human_side}": self._bound_input_label,
                    "binding_stats": self._binding_stats,
                }
        return metadata

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        if self.config.require_calibration and self.config.mode != "arm-only" and not self._hand_calibration:
            raise RuntimeError(
                f"QuestHTS hand calibration missing: {self.config.right_hand_calibration_file}"
            )
        if (
            not self._is_native_right_mode()
            and not self._is_single_visible_right_mode()
            and self.config.binding_mode == "saved_map"
            and self._bound_input_label is None
        ):
            self._run_binding_step()
        self._stop_event.clear()
        self._receiver_alive = False
        self._client_connected = False
        self._last_receiver_error = None
        if self.config.start_receiver:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((self.config.host, self.config.port))
                server.listen(1)
                server.settimeout(0.5)
            except OSError:
                server.close()
                raise
            self._server_sock = server
            self._thread = threading.Thread(target=self._tcp_server_loop, args=(server,), daemon=True)
            self._thread.start()
            self._emit("TCP server listening on %s:%d", self.config.host, self.config.port, force_print=True)
        self._connected = True

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._stop_event.set()
        if self._server_sock is not None:
            with contextlib.suppress(OSError):
                self._server_sock.close()
            self._server_sock = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._connected = False

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        with self._lock:
            normalized_feedback = unprefixed_action(feedback, self._robot_side)
            numeric = {
                key: float(value) for key, value in normalized_feedback.items() if key in RIGHT_ACTION_KEYS
            }
            self._last_feedback.update(numeric)
            if self._neutral_action is None and all(key in self._last_feedback for key in RIGHT_ARM_KEYS):
                self._neutral_action = {
                    key: float(self._last_feedback.get(key, 0.0)) for key in RIGHT_ACTION_KEYS
                }
            if (
                self.config.arm_control_mode in {"ik_ee", "constrained_planar_ik"}
                and not self.config.manual_baseline
                and not self._ee_baseline_captured
                and self._wrist_baseline is not None
                and all(key in self._last_feedback for key in RIGHT_ARM_KEYS)
            ):
                if self._is_constrained_planar_mode():
                    self._capture_constrained_planar_baseline_locked(self._wrist_baseline)
                else:
                    self._capture_ee_baseline_locked(self._wrist_baseline)

    def handle_hts_line(self, line: str) -> bool:
        parsed = parse_hts_line(line)
        if parsed is None:
            return False
        side, kind, values = parsed
        if side not in HTS_RAW_LABELS:
            return False
        now = time.monotonic()
        with self._lock:
            self._update_raw_input_locked(side, kind, tuple(values), now)
            self._maybe_auto_bind_locked()
            if side != self._active_input_label_locked():
                return True
            self._sync_bound_latest_locked()
        return True

    def _update_raw_input_locked(self, side: str, kind: str, values: tuple[float, ...], now: float) -> None:
        raw = self._raw_inputs[side]
        raw["seq"] += 1
        raw["last_frame_s"] = now
        raw["frame_count"] += 1
        self._latest_seq += 1
        self._last_stream_stale_seq = None
        if kind == "wrist":
            previous_wrist = raw.get("wrist")
            raw["wrist"] = values
            raw["wrist_s"] = now
            raw["wrist_count"] += 1
            raw["wrist_history"].append(values)
            if raw.get("first_wrist") is None:
                raw["first_wrist"] = values
            if previous_wrist is not None:
                raw["movement_m"] += _wrist_translation_distance(previous_wrist, values)
            first_wrist = raw.get("first_wrist")
            raw["span_m"] = (
                _wrist_translation_distance(first_wrist, values) if first_wrist is not None else 0.0
            )
        elif kind == "landmarks":
            previous_landmarks = raw.get("landmarks")
            raw["landmarks"] = values
            raw["landmarks_s"] = now
            raw["landmarks_count"] += 1
            if raw.get("first_landmarks") is None:
                raw["first_landmarks"] = values
            if previous_landmarks is not None:
                raw["landmark_movement"] += _landmark_average_distance(previous_landmarks, values)
        else:
            return

    def _sync_bound_latest_locked(self) -> None:
        label = self._active_input_label_locked()
        if label is None:
            return
        raw = self._raw_inputs[label]
        self._latest_wrist = raw.get("wrist")
        self._latest_wrist_s = raw.get("wrist_s")
        self._latest_landmarks = raw.get("landmarks")
        self._latest_landmarks_s = raw.get("landmarks_s")
        self._wrist_history = list(raw.get("wrist_history", []))
        if self._wrist_baseline is None and len(self._wrist_history) >= self.config.baseline_samples:
            self._wrist_baseline = _average_wrist(self._wrist_history[: self.config.baseline_samples])
        if (
            self.config.arm_control_mode in {"ik_ee", "constrained_planar_ik"}
            and not self.config.manual_baseline
            and not self._ee_baseline_captured
            and self._wrist_baseline is not None
            and all(key in self._last_feedback for key in RIGHT_ARM_KEYS)
        ):
            if self._is_constrained_planar_mode():
                self._capture_constrained_planar_baseline_locked(self._wrist_baseline)
            else:
                self._capture_ee_baseline_locked(self._wrist_baseline)

    def _active_input_label_locked(self) -> str | None:
        if self._bound_input_label is not None:
            return self._bound_input_label
        if self.config.binding_mode == "explicit":
            return self.config.bound_input_label or self.config.input_hand
        return None

    def _maybe_auto_bind_locked(self) -> None:
        if (
            self._bound_input_label is not None
            or self.config.binding_mode != "auto"
            or self.config.manual_baseline
        ):
            return
        label, confidence, stats, reason = self._evaluate_binding_locked()
        self._binding_stats = stats
        self._binding_error = reason
        if label is None:
            return
        self._set_bound_input_label_locked(label, confidence, stats)

    def _set_bound_input_label_locked(self, label: str, confidence: float, stats: dict[str, object]) -> None:
        self._bound_input_label = label
        self._binding_confidence = confidence
        self._binding_stats = stats
        self._binding_error = None
        self._binding_completed_at = _now_iso()
        if self.config.require_live_side_confirmation:
            self._live_side_confirmed = False
            self._live_side_confirmation_completed_at = None
            self._live_side_confirmation_error = None
        self._wrist_baseline = None
        self._sync_bound_latest_locked()
        if self.config.print_hand_binding:
            hand_label = _physical_hand_label(self.config.physical_hand)
            verb = "loaded" if self.config.binding_mode == "saved_map" else "captured"
            self._emit("Physical %s hand mapping %s successfully.", hand_label, verb, force_print=True)
            if self.config.print_debug_transport_labels:
                self._emit(
                    "transport_debug.raw_label_for_%s=%s", self.config.physical_hand, label, force_print=True
                )

    def _evaluate_binding_locked(self) -> tuple[str | None, float | None, dict[str, object], str | None]:
        stats = {label: _binding_side_stats(self._raw_inputs[label]) for label in HTS_RAW_LABELS}
        candidates = [
            label
            for label, side_stats in stats.items()
            if side_stats["frames"] >= self.config.min_binding_frames
            and side_stats["movement_m"] >= self.config.min_binding_movement_m
        ]
        if not candidates:
            any_frames = any(side_stats["frames"] > 0 for side_stats in stats.values())
            reason = "movement_too_small" if any_frames else "no_frames"
            return None, None, stats, reason
        candidates.sort(key=lambda label: stats[label]["movement_m"], reverse=True)
        best = candidates[0]
        if len(candidates) > 1:
            second = candidates[1]
            best_movement = float(stats[best]["movement_m"])
            second_movement = float(stats[second]["movement_m"])
            if best_movement <= 0.0 or second_movement / best_movement >= 0.5:
                return None, None, stats, "ambiguous"
        best_movement = float(stats[best]["movement_m"])
        other_movement = max(float(stats[label]["movement_m"]) for label in HTS_RAW_LABELS if label != best)
        confidence = best_movement / max(best_movement + other_movement, 1e-9)
        return best, confidence, stats, None

    def _run_binding_step(self) -> None:
        if self._is_native_right_mode():
            with self._lock:
                self._set_bound_input_label_locked(
                    "right", 1.0, {"right": _binding_side_stats(self._raw_inputs["right"])}
                )
                self._live_side_confirmed = True
            return
        if self._is_single_visible_right_mode():
            if not self.assign_single_visible_right_input():
                raise RuntimeError("Could not assign single visible hand to RIGHT robot input.")
            return
        if self.config.binding_mode == "saved_map":
            label = raw_label_for_physical_side(
                load_side_mapping(self.config.side_mapping_file), self.config.physical_hand
            )
            with self._lock:
                self._set_bound_input_label_locked(
                    label, 1.0, {label: _binding_side_stats(self._raw_inputs[label])}
                )
            return
        if self.config.binding_mode == "explicit":
            label = self.config.bound_input_label or self.config.input_hand
            if label is None:
                raise RuntimeError("--teleop.binding-mode=explicit requires --teleop.bound-input-label.")
            with self._lock:
                self._set_bound_input_label_locked(
                    label, 1.0, {label: _binding_side_stats(self._raw_inputs[label])}
                )
            return
        hand_label = _physical_hand_label(self.config.physical_hand)
        self._emit(
            "Show only your physical %s hand and gently move it to bind the input hand.",
            hand_label,
            force_print=True,
        )
        self._emit(
            "Show only your physical %s hand and gently move it. Detecting the corresponding QuestHTS hand label...",
            hand_label,
            force_print=True,
        )
        with self._lock:
            self._reset_input_capture_state_locked(clear_binding=True)
        deadline = time.monotonic() + self.config.hand_binding_timeout_s
        last_reason = None
        while time.monotonic() < deadline and not self._stop_event.is_set():
            with self._lock:
                label, confidence, stats, reason = self._evaluate_binding_locked()
                self._binding_stats = stats
                self._binding_error = reason
                if label is not None and confidence is not None:
                    self._set_bound_input_label_locked(label, confidence, stats)
                    return
                last_reason = reason
            time.sleep(0.05)
        message = f"Could not identify physical {hand_label} hand. Keep the other hand hidden and move only your {hand_label} hand."
        self._emit(message, force_print=True)
        raise RuntimeError(message if last_reason is None else f"{message} binding_reason={last_reason}")

    def _prefix_keyed_mapping(self, mapping: Mapping[str, object]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in mapping.items():
            if isinstance(key, str) and key.endswith(".pos"):
                output[prefixed_key(self._robot_side, key)] = value
            else:
                output[key] = value
        return output

    def _binding_debug_marker_locked(self) -> dict[str, object]:
        if self._is_native_right_mode():
            return {
                "active_hts_wrist_label": "Right wrist",
                "active_hts_landmarks_label": "Right landmarks",
            }
        return {"side_mapping_ready": self._bound_input_label is not None}

    def _public_debug_locked(self, debug: dict[str, object]) -> dict[str, object]:
        public = dict(debug)
        public.pop("input_hand", None)
        public.pop("bound_input_label", None)
        public["human_side"] = self._human_side
        public["robot_side"] = self._robot_side
        public["input_assignment_mode"] = self.config.input_assignment_mode
        public["physical_side_verified"] = self._physical_side_verified()
        if self._is_single_visible_right_mode():
            public["assignment_assumption"] = "only_visible_hand_is_physical_right"
            public["side_mapping_ready"] = self._bound_input_label is not None
            public["require_live_side_confirmation"] = self.config.require_live_side_confirmation
            public["live_side_confirmed"] = self._live_side_confirmed
        elif self._is_native_right_mode():
            public.pop("assignment_assumption", None)
            public.pop("binding_mode", None)
            public.pop("binding_confidence", None)
            public.pop("side_mapping_ready", None)
            public.pop("require_live_side_confirmation", None)
            public.pop("live_side_confirmed", None)
            public["active_hts_wrist_label"] = "Right wrist"
            public["active_hts_landmarks_label"] = "Right landmarks"
        else:
            public["assignment_assumption"] = None
            public["side_mapping_ready"] = self._bound_input_label is not None
            public["require_live_side_confirmation"] = self.config.require_live_side_confirmation
            public["live_side_confirmed"] = self._live_side_confirmed
        public["canonical_action_keys"] = list(self._mode_keys(canonical=True))
        for field in (
            "arm_neutral_calibrated",
            "arm_controller_delta",
            "arm_controller_delta_smoothed",
            "arm_target_before_step_limit",
            "arm_target",
            "robot_start_action",
            "ik_joint_target_before_limit",
            "ik_joint_target_after_limit",
            "calibrated_arm_action",
            "action_sent_to_robot",
            "joint_limit_clamp_log",
            "hand_target",
            "calibrated_hand_action",
        ):
            value = public.get(field)
            if isinstance(value, dict):
                public[field] = self._prefix_keyed_mapping(value)
        for field in ("controlled_joints", "fixed_joints"):
            value = public.get(field)
            if isinstance(value, list | tuple):
                public[field] = [prefixed_key(self._robot_side, str(key)) for key in value]
        if self.config.print_debug_transport_labels:
            public["transport_debug"] = {
                f"raw_label_for_{self._human_side}": self._bound_input_label,
                "binding_stats": self._binding_stats,
            }
        return public

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        with self._lock:
            debug = {
                "human_side": self._human_side,
                "robot_side": self._robot_side,
                "input_assignment_mode": self.config.input_assignment_mode,
                "physical_side_verified": self._physical_side_verified(),
                "assignment_assumption": (
                    "only_visible_hand_is_physical_right" if self._is_single_visible_right_mode() else None
                ),
                "binding_mode": self.config.binding_mode,
                "binding_confidence": self._binding_confidence,
                "arm_control_mode": self.config.arm_control_mode,
                **self._receiver_debug_locked(),
                "output_units": "lerobot_calibrated",
                "raw_tick_path_used": False,
                "target_source": "none",
                "wrist_present": self._latest_wrist is not None,
                "landmarks_present": self._latest_landmarks is not None,
            }
            single_visible_block = self._single_visible_runtime_block_locked()
            if single_visible_block is not None:
                debug.update(
                    {
                        **single_visible_block,
                        "baseline_ready": self._ee_baseline_captured,
                    }
                )
                self.last_action_debug = self._public_debug_locked(debug)
                return {}
            if self._bound_input_label is None:
                debug.update(
                    {
                        "action_block_reason": "binding_not_completed",
                        "baseline_ready": self._ee_baseline_captured,
                        "ik_success": False,
                        "ik_failure": False,
                    }
                )
                self.last_action_debug = self._public_debug_locked(debug)
                return {}
            if self.config.require_live_side_confirmation and not self._live_side_confirmed:
                debug.update(
                    {
                        "action_block_reason": "live_side_confirmation_not_completed",
                        "baseline_ready": self._ee_baseline_captured,
                        "ik_success": False,
                        "ik_failure": False,
                        "live_side_confirmed": False,
                    }
                )
                self.last_action_debug = self._public_debug_locked(debug)
                return {}
            if self._start_pose_mismatch_abort:
                debug.update(
                    {
                        "action_block_reason": "start_pose_mismatch_abort",
                        "start_pose_mismatch_details": list(self._start_pose_mismatch_details),
                        "baseline_ready": self._ee_baseline_captured,
                        "ik_success": False,
                        "ik_failure": False,
                    }
                )
                self.last_action_debug = self._public_debug_locked(debug)
                return {}
            action: dict[str, float] = {}
            if self.config.mode == "hand-only":
                action.update({key: float(self._last_feedback.get(key, 0.0)) for key in RIGHT_HAND_KEYS})
            elif self.config.arm_control_mode == "joint_delta":
                action.update({key: float(self._last_feedback.get(key, 0.0)) for key in self._mode_keys()})
                if self._neutral_action is not None:
                    for key in self._mode_keys():
                        action.setdefault(key, float(self._neutral_action.get(key, 0.0)))
            else:
                if self.config.mode != "arm-only":
                    action.update({key: float(self._last_feedback.get(key, 0.0)) for key in RIGHT_HAND_KEYS})
            if self.config.mode != "hand-only":
                arm_targets, arm_debug = self._build_arm_action_locked()
                action.update(arm_targets)
                debug.update(arm_debug)
                if (
                    self.config.arm_control_mode == "constrained_planar_ik"
                    and not arm_targets
                    and arm_debug.get("action_block_reason") is not None
                ):
                    self.last_action_debug = debug
                    self.last_action_debug = self._public_debug_locked(debug)
                    if self.config.print_debug or self.config.print_ik_debug:
                        self._emit(
                            "quest_hts_right action=%s debug=%s",
                            {},
                            self.last_action_debug,
                            force_print=self.config.print_ik_debug,
                        )
                    return {}
            if self.config.mode != "arm-only":
                hand_targets, hand_debug = self._build_hand_action_locked()
                action.update(hand_targets)
                debug.update(hand_debug)
            action = {key: float(action[key]) for key in self._mode_keys() if key in action}
            public_action = prefixed_action(action, self._robot_side)
            self.last_action_debug = self._public_debug_locked(debug)
            if self.config.print_debug or self.config.print_ik_debug:
                self._emit(
                    "quest_hts_right action=%s debug=%s",
                    public_action,
                    self.last_action_debug,
                    force_print=self.config.print_ik_debug,
                )
            return public_action

    def _mode_keys(self, *, canonical: bool = False) -> tuple[str, ...]:
        if self.config.mode == "arm-only":
            keys = RIGHT_ARM_KEYS
        elif self.config.mode == "hand-only":
            keys = RIGHT_HAND_KEYS
        else:
            keys = RIGHT_ACTION_KEYS
        return _canonical_keys(keys, self._robot_side) if canonical else keys

    def _build_arm_action_locked(self) -> tuple[dict[str, float], dict]:
        if self.config.arm_control_mode == "ik_ee":
            return self._build_ik_ee_arm_action_locked()
        if self.config.arm_control_mode == "constrained_planar_ik":
            return self._build_constrained_planar_ik_arm_action_locked()
        return self._build_joint_delta_arm_action_locked()

    def _build_joint_delta_arm_action_locked(self) -> tuple[dict[str, float], dict]:
        if self._neutral_action is None or self._latest_wrist is None or self._wrist_baseline is None:
            return {}, {"arm_baseline_ready": self._wrist_baseline is not None, "arm_target_source": "none"}
        age = None if self._latest_wrist_s is None else time.monotonic() - self._latest_wrist_s
        if age is not None and age > self._stale_timeout_s():
            return {}, {"arm_baseline_ready": True, "arm_target_source": "none", "latest_input_age_s": age}
        delta, delta_debug = _calibrated_arm_delta(self.config, self._latest_wrist, self._wrist_baseline)
        target_before_step = {}
        step_limited = False
        cumulative_limited = False
        for index, key in enumerate(RIGHT_ARM_KEYS):
            limited_delta = max(
                -self.config.max_arm_cumulative, min(self.config.max_arm_cumulative, delta[key])
            )
            cumulative_limited = cumulative_limited or abs(limited_delta - delta[key]) > 1e-9
            alpha = self.config.arm_smoothing_alpha
            self._smoothed_arm_delta[key] = (
                alpha * limited_delta + (1.0 - alpha) * self._smoothed_arm_delta[key]
            )
            lo, hi = self.config.arm_joint_limits[index]
            target_before_step[key] = max(
                lo, min(hi, self._neutral_action[key] + self._smoothed_arm_delta[key])
            )
        previous = self._previous_arm_target or {key: self._neutral_action[key] for key in RIGHT_ARM_KEYS}
        target = {}
        for key in RIGHT_ARM_KEYS:
            lo = previous[key] - self.config.max_arm_step
            hi = previous[key] + self.config.max_arm_step
            value = max(lo, min(hi, target_before_step[key]))
            step_limited = step_limited or abs(value - target_before_step[key]) > 1e-9
            target[key] = value
        self._previous_arm_target = dict(target)
        return target, {
            **delta_debug,
            "arm_baseline_ready": True,
            "arm_target_source": "fresh",
            "arm_neutral_calibrated": {key: self._neutral_action[key] for key in RIGHT_ARM_KEYS},
            "arm_controller_delta": dict(delta),
            "arm_controller_delta_smoothed": dict(self._smoothed_arm_delta),
            "arm_target_before_step_limit": target_before_step,
            "arm_target": dict(target),
            "arm_step_limited": step_limited,
            "arm_cumulative_limited": cumulative_limited,
            "target_source": "fresh",
        }

    def _capture_constrained_planar_baseline_locked(self, wrist: tuple[float, ...]) -> bool:
        if not self._planar_start_action or self._planar_start_ee is None:
            return False
        self._human_wrist_baseline_pose = _wrist_transform(wrist)
        self._planar_live_human_baseline = self._human_wrist_baseline_pose
        saved_human = self._saved_human_baseline_pose()
        if saved_human is not None:
            saved_delta_m = vector_norm(self._human_wrist_baseline_pose[:3, 3] - saved_human[:3, 3])
            if saved_delta_m > 0.05:
                warning = (
                    "human live baseline differs from saved Phase28 wrist baseline: "
                    f"translation_diff_m={saved_delta_m:.4f}"
                )
                self._planar_start_pose_warnings.append(warning)
                self._emit(f"WARNING: {warning}", force_print=True)
        self._ee_baseline_captured = True
        self._baseline_completed_at = _now_iso()
        self._last_valid_ik_action = dict(self._planar_start_action)
        self._last_controlled_joint_action = None
        self._last_hand_joint_action = None
        # Capture current hand positions from feedback as the hand start action.
        # _last_feedback uses unprefixed keys (send_feedback strips the side prefix).
        self._hand_start_action = {
            key: float(val) for key, val in self._last_feedback.items() if key in RIGHT_HAND_KEYS
        }
        self.last_action_debug = {
            **self.last_action_debug,
            "arm_control_mode": self.config.arm_control_mode,
            "human_side": self._human_side,
            "robot_side": self._robot_side,
            **self._binding_debug_marker_locked(),
            "start_pose_file": None
            if self.config.start_pose_file is None
            else str(self.config.start_pose_file),
            "constraints_file": None
            if self.config.constraints_file is None
            else str(self.config.constraints_file),
            "live_human_baseline": pose_to_dict(self._planar_live_human_baseline),
            "planar_ee_start": dict(self._planar_start_ee),
            "robot_start_action": self._prefix_keyed_mapping(self._planar_start_action),
        }
        return True

    def _saved_human_baseline_pose(self) -> np.ndarray | None:
        human = self._phase28_start_pose.get("human", {})
        if not isinstance(human, dict):
            return None
        position = human.get("wrist_position", {})
        rotation = human.get("wrist_rotation", {})
        if not isinstance(position, dict) or not isinstance(rotation, dict):
            return None
        values = [
            position.get("x"),
            position.get("y"),
            position.get("z"),
            rotation.get("x"),
            rotation.get("y"),
            rotation.get("z"),
            rotation.get("w"),
        ]
        if any(value is None for value in values):
            return None
        return transform_from_xyz_quat(*(float(value) for value in values))

    def _warn_if_robot_not_at_planar_start_locked(self) -> None:
        if not self._planar_start_action:
            return
        # Check only the controlled joints (fixed joints are always overridden to start values).
        keys_to_check = self._planar_controlled_joints or list(self._planar_start_action.keys())
        threshold = self.config.start_pose_max_error
        # Fall back to the legacy threshold if start_pose_policy is still using the old API.
        if self.config.allow_start_pose_warning and self.config.start_pose_policy == "abort":
            threshold = self.config.start_pose_warning_threshold
        mismatches = []
        for key in keys_to_check:
            start_value = self._planar_start_action.get(key)
            if start_value is None:
                continue
            current = self._last_feedback.get(key)
            if current is None:
                continue
            diff = abs(float(current) - float(start_value))
            if diff > threshold:
                mismatches.append(
                    f"{key}: current={float(current):.3f}, saved_start={float(start_value):.3f}, diff={diff:.3f}"
                )
        if not mismatches:
            return
        self._planar_start_pose_warnings.extend(mismatches)
        policy = self.config.start_pose_policy
        # Legacy override: allow_start_pose_warning=True → warn mode
        if self.config.allow_start_pose_warning and policy == "abort":
            policy = "warn"
        if policy == "ignore":
            return
        detail = "; ".join(mismatches)
        if policy == "warn":
            self._emit(
                "WARNING: current robot observation differs from saved Phase28 start pose: %s",
                detail,
                force_print=True,
            )
            return
        # policy == "abort"
        self._start_pose_mismatch_abort = True
        self._start_pose_mismatch_details = mismatches
        abort_message = (
            "START POSE MISMATCH: robot is not at the saved start pose.\n"
            "Refusing to send teleop actions.\n"
            "Reposition the robot or recapture the start pose.\n"
            f"Details: {detail}"
        )
        print(abort_message, flush=True)
        raise RuntimeError(abort_message)

    def _clamp_planar_action_locked(
        self, action: Mapping[str, float]
    ) -> tuple[dict[str, float], dict[str, tuple[float, float, float]]]:
        clamped: dict[str, float] = {}
        clamp_log: dict[str, tuple[float, float, float]] = {}
        for key, value in action.items():
            low_high = self._planar_joint_limits.get(key)
            if low_high is None:
                low_high = self._calibration.action_limits(key.removesuffix(".pos"))
            low, high = low_high
            bounded = max(float(low), min(float(high), float(value)))
            clamped[key] = bounded
            if abs(bounded - float(value)) > 1e-9:
                clamp_log[key] = (float(value), float(low), float(high))
        return clamped, clamp_log

    def _build_constrained_planar_ik_arm_action_locked(self) -> tuple[dict[str, float], dict]:
        if not self._planar_start_action or self._planar_start_ee is None or not self._phase28_constraints:
            return {}, {
                "arm_baseline_ready": False,
                "baseline_ready": False,
                "arm_target_source": "none",
                "action_block_reason": "missing_phase28_start_pose_or_constraints",
                "ik_success": False,
                "ik_failure": False,
                "start_pose_file": None
                if self.config.start_pose_file is None
                else str(self.config.start_pose_file),
                "constraints_file": None
                if self.config.constraints_file is None
                else str(self.config.constraints_file),
            }
        if not self._ee_baseline_captured or self._human_wrist_baseline_pose is None:
            return {}, {
                "arm_baseline_ready": False,
                "baseline_ready": False,
                "arm_target_source": "none",
                "action_block_reason": "baseline_not_captured",
                "ik_success": False,
                "ik_failure": False,
                "ik_failure_count": self._ik_failure_count,
            }
        if self._latest_wrist is None:
            return {}, {
                "arm_baseline_ready": True,
                "baseline_ready": True,
                "arm_target_source": "none",
                "action_block_reason": "no_wrist_frame",
                "ik_success": False,
                "ik_failure": False,
                **self._receiver_debug_locked(),
            }
        age = None if self._latest_wrist_s is None else time.monotonic() - self._latest_wrist_s
        if age is not None and age > self._stale_timeout_s():
            self._emit_stale_warning_locked(age)
            return {}, {
                **self._receiver_debug_locked(),
                "arm_baseline_ready": True,
                "baseline_ready": True,
                "arm_target_source": "none",
                "action_block_reason": "stale_input",
                "latest_input_age_s": age,
                "ik_success": False,
                "ik_failure": False,
            }

        human_pose = _wrist_transform(self._latest_wrist)
        human_delta = invert_transform(self._human_wrist_baseline_pose) @ human_pose
        human_dx = float(human_delta[0, 3])
        human_dy = float(human_delta[1, 3])
        scale_x = float(self._planar_ik_config.get("scale_x", 1.0))
        scale_y = float(self._planar_ik_config.get("scale_y", 1.0))
        target_x = float(self._planar_start_ee["x"]) + scale_x * human_dx
        target_y = float(self._planar_start_ee["y"]) + scale_y * human_dy
        l1 = float(self._planar_ik_config.get("l1", self._kinematics.l1))
        l2 = float(self._planar_ik_config.get("l2", self._kinematics.l2))
        original_target_x = target_x
        original_target_y = target_y
        planar_projected = False
        planar_projection_reason = ""
        if self.config.planar_workspace_policy == "project":
            target_x, target_y, planar_projected, planar_projection_reason = _project_planar_target_to_reach(
                target_x,
                target_y,
                l1,
                l2,
            )
        self._last_planar_ee_target = {"x": target_x, "y": target_y}

        try:
            shoulder_lift_deg, elbow_flex_deg = self._kinematics.inverse_kinematics(
                target_x,
                target_y,
                l1=l1,
                l2=l2,
                clamp_to_workspace=self.config.planar_workspace_policy == "project",
            )
        except ValueError as exc:
            return self._handle_ik_failure_locked(
                reason=str(exc),
                debug={
                    "arm_baseline_ready": True,
                    "baseline_ready": True,
                    "arm_target_source": "ik-failed",
                    "action_block_reason": "ik_failed",
                    "target_source": "none",
                    "human_wrist_pose": pose_to_dict(human_pose),
                    "human_relative_pose_delta": pose_to_dict(human_delta),
                    "planar_ee_start": dict(self._planar_start_ee),
                    "planar_ee_target_requested": {"x": original_target_x, "y": original_target_y},
                    "planar_ee_target": {"x": target_x, "y": target_y},
                    "planar_workspace_policy": self.config.planar_workspace_policy,
                    "planar_workspace_projected": planar_projected,
                    "planar_workspace_projection_reason": planar_projection_reason,
                    "controlled_joints": self._planar_controlled_joints,
                    "fixed_joints": self._planar_fixed_joints,
                },
            )

        joint_degrees = {
            **self._calibration.action_dict_to_degrees(self._planar_start_action),
            "shoulder_lift": shoulder_lift_deg,
            "elbow_flex": elbow_flex_deg,
        }
        controlled_action = self._calibration.degrees_dict_to_action(
            {"shoulder_lift": shoulder_lift_deg, "elbow_flex": elbow_flex_deg}
        )
        action_before_limit = {
            key: float(self._planar_start_action[key])
            for key in RIGHT_ARM_KEYS
            if key in self._planar_start_action
        }
        action_before_limit.update(controlled_action)

        # --- Step continuity clamping ---
        max_step = self.config.max_controlled_joint_step
        max_delta_from_start = self.config.max_controlled_joint_delta_from_start
        controlled_keys = list(controlled_action.keys())

        reference = self._last_controlled_joint_action
        if reference is None:
            reference = {
                k: float(self._planar_start_action[k])
                for k in controlled_keys
                if k in self._planar_start_action
            }

        first_action = self._last_controlled_joint_action is None
        step_clamped: dict[str, float] = {}
        held_due_to_first_jump = False
        first_jump_detail: dict[str, float] = {}

        clamped_controlled = dict(controlled_action)
        for key in controlled_keys:
            raw_val = clamped_controlled.get(key, reference.get(key, 0.0))
            ref_val = reference.get(key, raw_val)
            delta = raw_val - ref_val
            if abs(delta) > max_step:
                clamped = ref_val + max_step * (1.0 if delta > 0 else -1.0)
                step_clamped[key] = clamped
                clamped_controlled[key] = clamped

        if first_action:
            for key in controlled_keys:
                start_val = float(self._planar_start_action.get(key, clamped_controlled[key]))
                delta_from_start = abs(clamped_controlled[key] - start_val)
                if delta_from_start > max_delta_from_start:
                    held_due_to_first_jump = True
                    first_jump_detail[key] = delta_from_start

        if held_due_to_first_jump:
            logger.warning(
                "Constrained planar IK: first action deviates too far from start pose (%s); holding start.",
                first_jump_detail,
            )
            fail_debug = {
                "arm_baseline_ready": True,
                "baseline_ready": True,
                "arm_target_source": "none",
                "action_block_reason": "first_action_jump_from_start",
                "target_source": "none",
                "ik_success": False,
                "ik_failure": True,
                "ik_failure_reason": "first_action_jump_from_start",
                "first_jump_detail": first_jump_detail,
                "max_controlled_joint_delta_from_start": max_delta_from_start,
                **self._receiver_debug_locked(),
            }
            self.last_action_debug = self._public_debug_locked(fail_debug)
            return {}, fail_debug

        action_before_limit.update(clamped_controlled)
        action_after_task_limit, task_clamp_log = self._clamp_planar_action_locked(action_before_limit)
        action_after_limit, calibration_clamp_log = self._calibration.clamp_action(action_after_task_limit)
        clamp_log = {**task_clamp_log, **calibration_clamp_log}
        self._joint_limit_clamp_count += len(clamp_log)
        if clamp_log:
            logger.warning("SO-101 constrained planar IK target clamped: %s", clamp_log)

        self._last_ik_joint_degrees = dict(joint_degrees)
        self._last_valid_ik_action = dict(action_after_limit)
        self._last_controlled_joint_action = {
            k: action_after_limit[k] for k in controlled_keys if k in action_after_limit
        }
        debug = {
            "arm_baseline_ready": True,
            "baseline_ready": True,
            "arm_target_source": "fresh",
            "action_block_reason": None,
            "target_source": "fresh",
            "ik_success": True,
            "ik_failure": False,
            "ik_failure_count": self._ik_failure_count,
            "start_pose_file": None
            if self.config.start_pose_file is None
            else str(self.config.start_pose_file),
            "constraints_file": None
            if self.config.constraints_file is None
            else str(self.config.constraints_file),
            "human_side": self._human_side,
            "robot_side": self._robot_side,
            **self._binding_debug_marker_locked(),
            "controlled_joints": self._planar_controlled_joints,
            "fixed_joints": self._planar_fixed_joints,
            "robot_start_action": dict(self._planar_start_action),
            "live_human_baseline": pose_to_dict(self._human_wrist_baseline_pose),
            "human_wrist_pose": pose_to_dict(human_pose),
            "human_relative_pose_delta": pose_to_dict(human_delta),
            "human_planar_delta_m": {"x": human_dx, "y": human_dy},
            "planar_ee_start": dict(self._planar_start_ee),
            "planar_ee_target_requested": {"x": original_target_x, "y": original_target_y},
            "planar_ee_target": {"x": target_x, "y": target_y},
            "planar_workspace_policy": self.config.planar_workspace_policy,
            "planar_workspace_projected": planar_projected,
            "planar_workspace_projection_reason": planar_projection_reason,
            "planar_ik": dict(self._planar_ik_config),
            "ik_output_degrees": {
                "shoulder_lift": shoulder_lift_deg,
                "elbow_flex": elbow_flex_deg,
            },
            "ik_joint_target_degrees": dict(joint_degrees),
            "ik_joint_target_before_limit": dict(action_before_limit),
            "ik_joint_target_after_limit": dict(action_after_limit),
            "calibrated_arm_action": dict(action_after_limit),
            "joint_limit_clamp_log": clamp_log,
            "joint_limit_clamp_count": self._joint_limit_clamp_count,
            "controlled_joint_step_clamped": step_clamped,
            "max_controlled_joint_step": max_step,
            "max_controlled_joint_delta_from_start": max_delta_from_start,
            "action_sent_to_robot": dict(action_after_limit),
            "start_pose_warnings": list(self._planar_start_pose_warnings),
        }
        if self.config.print_ik_debug:
            self._emit("Constrained planar IK debug=%s", self._public_debug_locked(debug), force_print=True)
        return action_after_limit, debug

    def _build_ik_ee_arm_action_locked(self) -> tuple[dict[str, float], dict]:
        if (
            not self._ee_baseline_captured
            or self._human_wrist_baseline_pose is None
            or self._robot_tcp_baseline_pose is None
        ):
            return {}, {
                "arm_baseline_ready": False,
                "baseline_ready": False,
                "arm_target_source": "none",
                "action_block_reason": "baseline_not_captured",
                "ik_success": False,
                "ik_failure": False,
                "ik_failure_count": self._ik_failure_count,
            }
        if self._latest_wrist is None:
            return {}, {
                "arm_baseline_ready": True,
                "baseline_ready": True,
                "arm_target_source": "none",
                "action_block_reason": "no_wrist_frame",
                "ik_success": False,
                "ik_failure": False,
                **self._receiver_debug_locked(),
            }
        age = None if self._latest_wrist_s is None else time.monotonic() - self._latest_wrist_s
        if age is not None and age > self._stale_timeout_s():
            self._emit_stale_warning_locked(age)
            return {}, {
                **self._receiver_debug_locked(),
                "arm_baseline_ready": True,
                "baseline_ready": True,
                "arm_target_source": "none",
                "action_block_reason": "stale_input",
                "latest_input_age_s": age,
                "ik_success": False,
                "ik_failure": False,
            }

        current_tcp_pose = self._current_tcp_pose_from_feedback_locked()
        current_joint_degrees = self._calibration.action_dict_to_degrees(self._last_feedback)
        human_pose = _wrist_transform(self._latest_wrist)
        human_delta = invert_transform(self._human_wrist_baseline_pose) @ human_pose
        mapped_delta, map_debug = self._map_human_delta_to_robot_delta(human_delta)
        target_tcp_pose = self._robot_tcp_baseline_pose @ mapped_delta
        self._last_target_tcp_pose = target_tcp_pose

        workspace_ok, workspace_reason = _workspace_contains(self._workspace_limits, target_tcp_pose)
        if not workspace_ok:
            return self._handle_ik_failure_locked(
                reason=workspace_reason,
                debug={
                    "arm_baseline_ready": True,
                    "baseline_ready": True,
                    "arm_target_source": "workspace-rejected",
                    "action_block_reason": "workspace_rejected",
                    "human_wrist_pose": pose_to_dict(human_pose),
                    "human_relative_pose_delta": pose_to_dict(human_delta),
                    "mapped_robot_tcp_target_pose": pose_to_dict(target_tcp_pose),
                    "target_tcp_pose": pose_to_dict(target_tcp_pose),
                    "robot_current_tcp_pose": None
                    if current_tcp_pose is None
                    else pose_to_dict(current_tcp_pose),
                    **map_debug,
                },
            )

        ik_result = self._kinematics.solve_tcp_ik(
            target_tcp_pose,
            current_joint_degrees=current_joint_degrees,
            tcp_offset=self._tcp_offset,
            clamp_to_workspace=False,
        )
        if not ik_result.success:
            return self._handle_ik_failure_locked(
                reason=ik_result.reason,
                debug={
                    "arm_baseline_ready": True,
                    "baseline_ready": True,
                    "arm_target_source": "ik-failed",
                    "action_block_reason": "ik_failed",
                    "human_wrist_pose": pose_to_dict(human_pose),
                    "human_relative_pose_delta": pose_to_dict(human_delta),
                    "mapped_robot_tcp_target_pose": pose_to_dict(target_tcp_pose),
                    "target_tcp_pose": pose_to_dict(target_tcp_pose),
                    "robot_current_tcp_pose": None
                    if current_tcp_pose is None
                    else pose_to_dict(current_tcp_pose),
                    "ik_target_flange_pose": pose_to_dict(ik_result.target_flange_pose),
                    **map_debug,
                },
            )

        self._last_ik_joint_degrees = dict(ik_result.joint_degrees)
        action_before_limit = self._calibration.degrees_dict_to_action(ik_result.joint_degrees)
        action_after_limit, clamp_log = self._calibration.clamp_action(action_before_limit)
        self._joint_limit_clamp_count += len(clamp_log)
        if clamp_log:
            logger.warning("SO-101 IK joint target clamped by LeRobot action limits: %s", clamp_log)
        self._last_valid_ik_action = dict(action_after_limit)
        debug = {
            "arm_baseline_ready": True,
            "baseline_ready": True,
            "arm_target_source": "fresh",
            "action_block_reason": None,
            "target_source": "fresh",
            "ik_success": True,
            "ik_failure": False,
            "ik_failure_count": self._ik_failure_count,
            "human_wrist_pose": pose_to_dict(human_pose),
            "baseline_human_wrist_pose": pose_to_dict(self._human_wrist_baseline_pose),
            "baseline_robot_tcp_pose": pose_to_dict(self._robot_tcp_baseline_pose),
            "human_relative_pose_delta": pose_to_dict(human_delta),
            "mapped_robot_tcp_target_pose": pose_to_dict(target_tcp_pose),
            "target_tcp_pose": pose_to_dict(target_tcp_pose),
            "robot_current_tcp_pose": None if current_tcp_pose is None else pose_to_dict(current_tcp_pose),
            "ik_target_flange_pose": pose_to_dict(ik_result.target_flange_pose),
            "ik_joint_target_degrees": dict(ik_result.joint_degrees),
            "ik_joint_target_before_limit": dict(action_before_limit),
            "ik_joint_target_after_limit": dict(action_after_limit),
            "calibrated_arm_action": dict(action_after_limit),
            "joint_limit_clamp_log": clamp_log,
            "joint_limit_clamp_count": self._joint_limit_clamp_count,
            "action_sent_to_robot": dict(action_after_limit),
            "tcp_offset": self._tcp_offset.as_dict(),
            "tcp_config_file": str(self.config.tcp_config_file),
            **map_debug,
        }
        if self.config.print_ik_debug:
            self._emit("IK EE debug=%s", self._public_debug_locked(debug), force_print=True)
        return action_after_limit, debug

    def _handle_ik_failure_locked(self, *, reason: str, debug: dict) -> tuple[dict[str, float], dict]:
        self._ik_failure_count += 1
        failure_debug = {
            **debug,
            "ik_success": False,
            "ik_failure": True,
            "ik_failure_reason": reason,
            "ik_failure_count": self._ik_failure_count,
            "ik_failure_policy": self.config.ik_failure_policy,
        }
        logger.warning("SO-101 IK failed; policy=%s reason=%s", self.config.ik_failure_policy, reason)
        if self.config.ik_failure_policy == "abort":
            raise RuntimeError(f"SO-101 IK failed: {reason}")
        if self.config.ik_failure_policy == "skip":
            return {}, failure_debug
        if self._last_valid_ik_action is None:
            return {}, {**failure_debug, "held_last_valid_action": False}
        return dict(self._last_valid_ik_action), {
            **failure_debug,
            "held_last_valid_action": True,
            "action_sent_to_robot": dict(self._last_valid_ik_action),
        }

    def _map_human_delta_to_robot_delta(
        self, human_delta: np.ndarray
    ) -> tuple[np.ndarray, dict[str, object]]:
        translation = np.array(human_delta[:3, 3], dtype=float)
        signs = np.array(self.config.tcp_axis_signs, dtype=float)
        mapped_translation = translation * signs * float(self.config.tcp_translation_scale)
        translation_norm = vector_norm(mapped_translation)
        translation_limited = False
        if translation_norm > self.config.max_tcp_translation_m:
            mapped_translation *= self.config.max_tcp_translation_m / translation_norm
            translation_limited = True

        roll, pitch, yaw = rpy_from_rotation_matrix(human_delta[:3, :3])
        mapped_rpy = np.array([roll, pitch, yaw], dtype=float) * float(self.config.tcp_rotation_scale)
        rotation_norm = vector_norm(mapped_rpy)
        rotation_limited = False
        if rotation_norm > self.config.max_tcp_rotation_rad:
            mapped_rpy *= self.config.max_tcp_rotation_rad / rotation_norm
            rotation_limited = True

        mapped = np.eye(4, dtype=float)
        mapped[:3, 3] = mapped_translation
        mapped[:3, :3] = rotation_matrix_from_rpy(
            float(mapped_rpy[0]), float(mapped_rpy[1]), float(mapped_rpy[2])
        )
        return mapped, {
            "mapped_robot_delta_pose": pose_to_dict(mapped),
            "quest_to_robot_axis_signs": tuple(float(v) for v in self.config.tcp_axis_signs),
            "human_delta_translation_m": tuple(float(v) for v in translation),
            "mapped_robot_delta_translation_m": tuple(float(v) for v in mapped_translation),
            "human_delta_rpy_rad": (float(roll), float(pitch), float(yaw)),
            "mapped_robot_delta_rpy_rad": tuple(float(v) for v in mapped_rpy),
            "tcp_translation_limited": translation_limited,
            "tcp_rotation_limited": rotation_limited,
        }

    def _capture_ee_baseline_locked(self, wrist: tuple[float, ...]) -> bool:
        robot_tcp = self._current_tcp_pose_from_feedback_locked()
        if robot_tcp is None:
            return False
        self._human_wrist_baseline_pose = _wrist_transform(wrist)
        self._robot_tcp_baseline_pose = robot_tcp
        self._ee_baseline_captured = True
        self._baseline_completed_at = _now_iso()
        self._last_valid_ik_action = {
            key: float(self._last_feedback.get(key, 0.0))
            for key in RIGHT_ARM_KEYS
            if key in self._last_feedback
        }
        self.last_action_debug = {
            **self.last_action_debug,
            "arm_control_mode": self.config.arm_control_mode,
            "human_side": self._human_side,
            "robot_side": self._robot_side,
            **self._binding_debug_marker_locked(),
            "baseline_human_wrist_pose": pose_to_dict(self._human_wrist_baseline_pose),
            "baseline_robot_tcp_pose": pose_to_dict(self._robot_tcp_baseline_pose),
            "tcp_offset": self._tcp_offset.as_dict(),
        }
        return True

    def _current_tcp_pose_from_feedback_locked(self) -> np.ndarray | None:
        if not all(key in self._last_feedback for key in RIGHT_ARM_KEYS):
            return None
        joint_degrees = self._calibration.action_dict_to_degrees(self._last_feedback)
        return self._kinematics.forward_tcp_pose(joint_degrees, self._tcp_offset)

    def _load_tcp_offset(self) -> TcpOffset:
        overrides = {
            "x": self.config.tcp_offset_x,
            "y": self.config.tcp_offset_y,
            "z": self.config.tcp_offset_z,
            "roll": self.config.tcp_offset_roll,
            "pitch": self.config.tcp_offset_pitch,
            "yaw": self.config.tcp_offset_yaw,
        }
        return load_tcp_offset_yaml(self.config.tcp_config_file, overrides)

    def _receiver_debug_locked(self) -> dict[str, object]:
        label = self._active_input_label_locked()
        raw = self._raw_inputs[label] if label is not None else None
        latest_frame_s = raw.get("last_frame_s") if raw is not None else None
        latest_input_age_s = None if latest_frame_s is None else time.monotonic() - latest_frame_s
        stream_receiving_fresh_frames = (
            latest_input_age_s is not None and latest_input_age_s <= self._stale_timeout_s()
        )
        debug = {
            "receiver_alive": self._receiver_alive,
            "receiver_thread_alive": self._receiver_alive
            and self._thread is not None
            and self._thread.is_alive(),
            "stream_receiving_fresh_frames": stream_receiving_fresh_frames,
            "client_connected": self._client_connected,
            "client_connect_count": self._client_connect_count,
            "reconnect_count": self._reconnect_count,
            "stream_stale_count": self._stream_stale_count,
            "input_frame_seq": 0 if raw is None else int(raw.get("seq", 0)),
            "latest_input_age_s": latest_input_age_s,
            "baseline_ready": self._ee_baseline_captured,
            "last_receiver_error": self._last_receiver_error,
        }
        if self._is_native_right_mode():
            debug["active_hts_wrist_label"] = "Right wrist"
            debug["active_hts_landmarks_label"] = "Right landmarks"
        return debug

    def _emit_stale_warning_locked(self, latest_input_age_s: float) -> None:
        now = time.monotonic()
        if now - self._last_stale_warning_print_s < 1.0:
            return
        self._last_stale_warning_print_s = now
        self._mark_stream_stale_locked()
        self._emit(
            "HTS input stale: latest_input_age_s=%.3f; skipping arm action",
            latest_input_age_s,
            force_print=True,
        )

    def _mark_stream_stale_locked(self) -> None:
        label = self._active_input_label_locked()
        if label is None:
            return
        raw_seq = int(self._raw_inputs[label].get("seq", 0))
        if self._last_stream_stale_seq == raw_seq:
            return
        self._last_stream_stale_seq = raw_seq
        self._stream_stale_count += 1
        if self.config.stale_policy == "abort-episode":
            self._episode_abort_reason = "stale_input"
        if self.config.stale_policy == "abort-run":
            self._episode_abort_reason = "stale_input"
        if self._client_connected:
            self._client_connected = False
            self._reconnect_count += 1
            self._close_active_connection_locked("stale_input")

    def _close_active_connection_locked(self, reason: str) -> None:
        conn = self._active_conn
        self._active_conn = None
        self._closing_active_conn_reason = reason
        if conn is None:
            return
        try:
            conn.close()
        except OSError:
            pass
        except AttributeError:
            pass

    @staticmethod
    def _is_bad_file_descriptor_error(exc: OSError) -> bool:
        return getattr(exc, "errno", None) == errno.EBADF or "Bad file descriptor" in str(exc)

    def should_abort_episode(self) -> bool:
        with self._lock:
            return self._episode_abort_reason is not None

    def clear_episode_abort_state(self) -> None:
        with self._lock:
            self._episode_abort_reason = None

    def _build_hand_action_locked(self) -> tuple[dict[str, float], dict]:
        input_label = self._active_input_label_locked()
        if input_label is None:
            return {}, {
                "hand_target_source": "none",
                "calibration_target_used": False,
                "action_block_reason": "binding_not_completed",
            }
        if self._latest_landmarks is None:
            return {}, {"hand_target_source": "none", "calibration_target_used": False}
        age = None if self._latest_landmarks_s is None else time.monotonic() - self._latest_landmarks_s
        if age is not None and age > self._stale_timeout_s():
            return {}, {
                "hand_target_source": "none",
                "calibration_target_used": False,
                "latest_landmarks_age_s": age,
            }
        result = self._mapper.update_from_parsed(input_label, "landmarks", self._latest_landmarks)
        features = _finger_features_for_side(result, input_label) if result is not None else {}
        if not features:
            return {}, {"hand_target_source": "none", "calibration_target_used": False}
        _cal_action, _raw_targets, raw_flex, adjusted_flex, _dbg = _apply_hand_calibration(
            calibration=self._hand_calibration,
            side="right",
            finger_features=features,
            hand_flex_gain=self.config.hand_flex_gain,
            hand_flex_saturation_threshold=self.config.hand_flex_saturation_threshold,
        )
        hand_targets_raw = {
            key.removeprefix("r_"): float(value)
            for key, value in _cal_action.items()
            if key.removeprefix("r_") in RIGHT_HAND_KEYS
        }

        # --- Hand step continuity clamping ---
        max_hand_step = self.config.max_hand_joint_step
        max_hand_delta_from_start = self.config.max_hand_joint_delta_from_start
        first_hand_action = self._last_hand_joint_action is None

        # Reference: last sent hand values, or hand start (current robot position at baseline).
        reference = self._last_hand_joint_action
        if reference is None:
            reference = self._hand_start_action or {}

        hand_step_clamped: dict[str, float] = {}
        hand_targets = {}
        for key, raw_val in hand_targets_raw.items():
            ref_val = reference.get(key, raw_val)
            delta = raw_val - ref_val
            if abs(delta) > max_hand_step:
                clamped = ref_val + max_hand_step * (1.0 if delta > 0 else -1.0)
                hand_step_clamped[key] = clamped
                hand_targets[key] = clamped
            else:
                hand_targets[key] = raw_val

        # On first action: also clamp against hand_start_action delta.
        if first_hand_action and self._hand_start_action:
            for key in list(hand_targets.keys()):
                start_val = self._hand_start_action.get(key, hand_targets[key])
                if abs(hand_targets[key] - start_val) > max_hand_delta_from_start:
                    # Clamp to stay within max_hand_delta_from_start of start.
                    overshoot = hand_targets[key] - start_val
                    hand_targets[key] = start_val + max_hand_delta_from_start * (
                        1.0 if overshoot > 0 else -1.0
                    )
                    hand_step_clamped[key] = hand_targets[key]

        self._last_hand_joint_action = dict(hand_targets)

        return hand_targets, {
            "hand_target_source": "fresh",
            "calibration_target_used": True,
            "raw_human_flex_by_key": raw_flex,
            "adjusted_human_flex_by_key": adjusted_flex,
            "hand_flex_features": features,
            "hand_start_action": dict(self._hand_start_action) if self._hand_start_action else {},
            "hand_target_before_clamp": dict(hand_targets_raw),
            "hand_target_after_clamp": dict(hand_targets),
            "hand_joint_step_clamped": dict(hand_step_clamped),
            "max_hand_joint_step": max_hand_step,
            "max_hand_joint_delta_from_start": max_hand_delta_from_start,
            "hand_target": dict(hand_targets),
            "calibrated_hand_action": dict(hand_targets),
            "target_source": "fresh",
        }

    def _emit(self, message: str, *args: object, force_print: bool = False) -> None:
        text = message % args if args else message
        logger.info(text)
        if force_print or self.config.print_debug:
            print(text, flush=True)

    def _tcp_server_loop(self, server: socket.socket) -> None:
        self._receiver_alive = True
        try:
            while not self._stop_event.is_set():
                try:
                    conn, addr = server.accept()
                except TimeoutError:
                    continue
                except OSError as exc:
                    if not self._stop_event.is_set():
                        self._last_receiver_error = str(exc)
                    break
                with self._lock:
                    self._client_connected = True
                    self._client_connect_count += 1
                    self._active_conn = conn
                    self._last_receiver_error = None
                self._emit("HTS client connected from %s", addr)
                try:
                    self._handle_connection(conn)
                finally:
                    with self._lock:
                        self._client_connected = False
                        if self._active_conn is conn:
                            self._active_conn = None
        finally:
            self._receiver_alive = False
            with self._lock:
                self._client_connected = False
                self._active_conn = None
            with contextlib.suppress(OSError):
                server.close()

    def _handle_connection(self, conn: socket.socket) -> None:
        try:
            with conn:
                conn.settimeout(0.5)
                buffer = ""
                while not self._stop_event.is_set():
                    try:
                        data = conn.recv(8192)
                    except TimeoutError:
                        continue
                    except OSError as exc:
                        with self._lock:
                            intentional_close = self._closing_active_conn_reason is not None
                            if (
                                not (intentional_close and self._is_bad_file_descriptor_error(exc))
                                and not self._stop_event.is_set()
                            ):
                                self._last_receiver_error = str(exc)
                        break
                    if not data:
                        break
                    try:
                        buffer += data.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line:
                            self.handle_hts_line(line)
        finally:
            with self._lock:
                if self._active_conn is conn:
                    self._active_conn = None
                self._closing_active_conn_reason = None


def _make_mapper_config(config: QuestHTSTeleoperatorConfig) -> DualArmDryRunConfig:
    return DualArmDryRunConfig(
        right=SideConfig(
            arm=RightArmMappingConfig(
                robot_home_joints=config.right_robot_home_joints,
                scale=config.scale,
                joint_limits=config.joint_limits,
                max_step=config.arm_max_step,
                max_abs_input_m=config.max_abs_input_m,
            ),
            hand=RightHandMappingConfig(
                open_targets=config.right_hand_open_target,
                closed_targets=config.right_hand_closed_target,
                motor_limits=config.motor_limits,
                max_step=config.hand_max_step,
                max_abs_input_m=config.max_abs_input_m,
            ),
            axis_signs=config.right_axis_signs,
        ),
        left=SideConfig(
            arm=RightArmMappingConfig(
                robot_home_joints=config.left_robot_home_joints,
                scale=config.scale,
                joint_limits=config.joint_limits,
                max_step=config.arm_max_step,
                max_abs_input_m=config.max_abs_input_m,
            ),
            hand=RightHandMappingConfig(
                open_targets=config.left_hand_open_target,
                closed_targets=config.left_hand_closed_target,
                motor_limits=config.motor_limits,
                max_step=config.hand_max_step,
                max_abs_input_m=config.max_abs_input_m,
            ),
            axis_signs=config.left_axis_signs,
        ),
    )


def _average_wrist(samples: list[tuple[float, ...]]) -> tuple[float, ...]:
    return tuple(sum(sample[i] for sample in samples) / len(samples) for i in range(7))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _physical_hand_label(physical_hand: str) -> str:
    return "RIGHT" if physical_hand == "right" else "LEFT" if physical_hand == "left" else "ACTIVE"


def _new_raw_input_state() -> dict[str, object]:
    return {
        "seq": 0,
        "frame_count": 0,
        "last_frame_s": None,
        "wrist": None,
        "wrist_s": None,
        "wrist_count": 0,
        "wrist_history": [],
        "first_wrist": None,
        "movement_m": 0.0,
        "span_m": 0.0,
        "landmarks": None,
        "landmarks_s": None,
        "landmarks_count": 0,
        "first_landmarks": None,
        "landmark_movement": 0.0,
    }


def _wrist_translation_distance(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float:
    if a is None or b is None:
        return 0.0
    return math.sqrt(sum((float(b[i]) - float(a[i])) ** 2 for i in range(3)))


def _landmark_average_distance(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float:
    if a is None or b is None:
        return 0.0
    count = min(len(a), len(b)) // 3
    if count <= 0:
        return 0.0
    total = 0.0
    for index in range(count):
        base = index * 3
        total += math.sqrt(sum((float(b[base + j]) - float(a[base + j])) ** 2 for j in range(3)))
    return total / count


def _project_planar_target_to_reach(
    x: float, y: float, l1: float, l2: float
) -> tuple[float, float, bool, str]:
    radius = math.hypot(x, y)
    max_radius = l1 + l2
    min_radius = abs(l1 - l2)
    if radius > max_radius:
        scale = max_radius / radius
        return x * scale, y * scale, True, "target beyond SO-101 planar reach"
    if 0.0 < radius < min_radius:
        scale = min_radius / radius
        return x * scale, y * scale, True, "target inside SO-101 planar minimum reach"
    if radius == 0.0 and min_radius > 0.0:
        return min_radius, 0.0, True, "target inside SO-101 planar minimum reach"
    return x, y, False, ""


def _binding_side_stats(raw: dict[str, object]) -> dict[str, object]:
    wrist_movement = max(float(raw.get("movement_m", 0.0)), float(raw.get("span_m", 0.0)))
    landmark_movement = float(raw.get("landmark_movement", 0.0))
    movement = max(wrist_movement, landmark_movement)
    return {
        "frames": int(raw.get("frame_count", 0)),
        "wrist_frames": int(raw.get("wrist_count", 0)),
        "landmark_frames": int(raw.get("landmarks_count", 0)),
        "movement_m": movement,
        "wrist_movement_m": wrist_movement,
        "landmark_movement_m": landmark_movement,
        "seq": int(raw.get("seq", 0)),
    }


def _quat_to_euler(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _angle_delta(current: float, baseline: float) -> float:
    return (current - baseline + math.pi) % (2.0 * math.pi) - math.pi


def _deadzone(value: float, deadzone: float) -> float:
    if abs(value) <= deadzone:
        return 0.0
    return math.copysign(abs(value) - deadzone, value)


def _calibrated_arm_delta(
    config: QuestHTSRightTeleoperatorConfig,
    wrist: tuple[float, ...],
    baseline: tuple[float, ...],
) -> tuple[dict[str, float], dict]:
    dx = _deadzone(float(wrist[0]) - float(baseline[0]), config.arm_deadzone_pos)
    dy = _deadzone(float(wrist[1]) - float(baseline[1]), config.arm_deadzone_pos)
    dz = _deadzone(float(wrist[2]) - float(baseline[2]), config.arm_deadzone_pos)
    roll, pitch, yaw = _quat_to_euler(float(wrist[3]), float(wrist[4]), float(wrist[5]), float(wrist[6]))
    b_roll, b_pitch, b_yaw = _quat_to_euler(
        float(baseline[3]),
        float(baseline[4]),
        float(baseline[5]),
        float(baseline[6]),
    )
    droll = _deadzone(_angle_delta(roll, b_roll), config.arm_deadzone_rot)
    dpitch = _deadzone(_angle_delta(pitch, b_pitch), config.arm_deadzone_rot)
    dyaw = _deadzone(_angle_delta(yaw, b_yaw), config.arm_deadzone_rot)
    delta = {
        "shoulder_pan.pos": dx * config.arm_pos_gain_x + dyaw * config.arm_rot_gain_yaw * 0.25,
        "shoulder_lift.pos": dz * config.arm_pos_gain_z - dy * config.arm_pos_gain_y * 0.35,
        "elbow_flex.pos": -dy * config.arm_pos_gain_y,
        "wrist_flex.pos": dpitch * config.arm_rot_gain_pitch,
        "wrist_roll.pos": droll * config.arm_rot_gain_roll + dyaw * config.arm_rot_gain_yaw * 0.5,
    }
    return delta, {
        "wrist_position": (float(wrist[0]), float(wrist[1]), float(wrist[2])),
        "wrist_rotation": (roll, pitch, yaw),
        "wrist_delta_position": {"x": dx, "y": dy, "z": dz},
        "wrist_delta_rotation": {"roll": droll, "pitch": dpitch, "yaw": dyaw},
    }


def _wrist_transform(wrist: tuple[float, ...]) -> np.ndarray:
    return transform_from_xyz_quat(
        float(wrist[0]),
        float(wrist[1]),
        float(wrist[2]),
        float(wrist[3]),
        float(wrist[4]),
        float(wrist[5]),
        float(wrist[6]),
    )


def _load_workspace_limits(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    import yaml

    with Path(path).open("r", encoding="utf-8") as input_file:
        data = yaml.safe_load(input_file)
    return data if isinstance(data, dict) else None


def _workspace_contains(limits: dict | None, target_tcp_pose: np.ndarray) -> tuple[bool, str]:
    if not limits:
        return True, ""
    position = target_tcp_pose[:3, 3]
    for index, axis in enumerate(("x", "y", "z")):
        axis_limits = limits.get(axis) or limits.get(f"{axis}_m")
        if not isinstance(axis_limits, list | tuple) or len(axis_limits) != 2:
            continue
        low, high = float(axis_limits[0]), float(axis_limits[1])
        if not low <= float(position[index]) <= high:
            return False, f"target TCP {axis}={position[index]:.4f} outside workspace [{low:.4f}, {high:.4f}]"
    return True, ""


def _write_yaml(path: Path, payload: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(payload, output_file, sort_keys=False)


def _format_action(action: dict[str, float]) -> str:
    ordered = OrderedDict((key, action[key]) for key in LEROBOT_DUAL_ARM_ACTION_KEYS)
    return "{" + ", ".join(f"{key}: {value:+.3f}" for key, value in ordered.items()) + "}"


def _format_combined_action(action: dict[str, float]) -> str:
    ordered = OrderedDict((key, action[key]) for key in DUAL_ARM_ACTION_KEYS)
    return "{" + ", ".join(f"{key}: {value:+.3f}" for key, value in ordered.items()) + "}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quest HTS LeRobot teleoperator dry-run.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--print-debug", action="store_true")
    parser.add_argument("--period-s", type=float, default=0.5)
    parser.add_argument(
        "--max-actions",
        type=int,
        default=0,
        help="Stop after logging this many get_action() samples. 0 means run until Ctrl+C.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("QuestHTS live dry-run: no Robot, serial port, or motor bus will be opened.")
    teleop = QuestHTSTeleoperator(
        QuestHTSTeleoperatorConfig(host=args.host, port=args.port, print_debug=args.print_debug)
    )
    teleop.connect()
    count = 0
    try:
        while teleop.is_connected:
            logger.info("action=%s", _format_action(teleop.get_action()))
            count += 1
            if args.max_actions > 0 and count >= args.max_actions:
                break
            time.sleep(max(args.period_s, 0.05))
    except KeyboardInterrupt:
        pass
    finally:
        if teleop.is_connected:
            teleop.disconnect()


if __name__ == "__main__":
    main()
