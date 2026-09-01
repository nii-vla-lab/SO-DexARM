#!/usr/bin/env python

from __future__ import annotations

import logging
import time
from functools import cached_property
from pathlib import Path

import yaml

from lerobot.motors import MotorCalibration
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.so101_amazinghand.so101_amazinghand_follower import (
    _HAND_CAL_MIN_SPAN,
    ALL_MOTORS,
    ARM_MOTORS,
    HAND_MOTORS,
    SO101AmazingHandFollower,
    _big_endian,
    _scs_def,
)
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.errors import DeviceAlreadyConnectedError

from .config_so101_amazinghand_right import SO101AmazingHandRightConfig

logger = logging.getLogger(__name__)


def _mode_motors(mode: str) -> list[str]:
    if mode == "arm-only":
        return list(ARM_MOTORS)
    if mode == "hand-only":
        return list(HAND_MOTORS)
    return list(ALL_MOTORS)


def _canonical_motor_key(motor: str, side: str = "right") -> str:
    prefix = "r_" if side == "right" else "l_"
    return f"{prefix}{motor}.pos"


def _motor_name_from_action_key(key: str, side: str = "right") -> str | None:
    if not key.endswith(".pos"):
        return None
    motor = key.removesuffix(".pos")
    expected_prefix = "r_" if side == "right" else "l_"
    other_prefix = "l_" if side == "right" else "r_"
    if motor.startswith(expected_prefix):
        return motor[2:]
    if motor.startswith(other_prefix):
        return None
    return motor


def _needs_arm_calibration(mode: str) -> bool:
    return mode in {"arm-only", "arm-and-hand"}


def _needs_hand_calibration(mode: str) -> bool:
    return mode in {"hand-only", "arm-and-hand"}


def _required_lerobot_motors(mode: str, hand_calibration_source: str) -> list[str]:
    motors: list[str] = []
    if _needs_arm_calibration(mode):
        motors.extend(ARM_MOTORS)
    if _needs_hand_calibration(mode) and hand_calibration_source == "lerobot":
        motors.extend(HAND_MOTORS)
    return motors


def _hand_yaml_key(motor: str, side: str = "right") -> str:
    return _canonical_motor_key(motor, side)


def _load_quest_hand_calibration(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as input_file:
        data = yaml.safe_load(input_file)
    return data if isinstance(data, dict) else {}


def _missing_quest_hand_features(
    calibration: dict,
    hand_motors: list[str] | tuple[str, ...] = HAND_MOTORS,
    side: str = "right",
) -> list[str]:
    missing: list[str] = []
    human_poses = (
        calibration.get("human_poses", {}) if isinstance(calibration.get("human_poses", {}), dict) else {}
    )
    robot = calibration.get("robot", {}) if isinstance(calibration.get("robot", {}), dict) else {}
    finger_mapping = calibration.get("finger_mapping", {})
    if not isinstance(finger_mapping, dict) or not finger_mapping:
        missing.append("finger_mapping")
    for pose in ("open", "closed"):
        features = (
            human_poses.get(pose, {}).get("features", {})
            if isinstance(human_poses.get(pose, {}), dict)
            else {}
        )
        if not features:
            missing.append(f"human_poses.{pose}.features")
    for section in (
        "open_raw_positions",
        "closed_raw_positions",
        "safe_min_raw_positions",
        "safe_max_raw_positions",
    ):
        values = robot.get(section, {}) if isinstance(robot.get(section, {}), dict) else {}
        for motor in hand_motors:
            key = _hand_yaml_key(motor, side)
            if key not in values:
                missing.append(f"robot.{section}.{key}")
    return missing


def _quest_hand_motor_calibration(
    calibration: dict,
    hand_motors: dict[str, object],
    side: str = "right",
) -> dict[str, MotorCalibration]:
    robot = calibration.get("robot", {}) if isinstance(calibration.get("robot", {}), dict) else {}
    safe_min = (
        robot.get("safe_min_raw_positions", {})
        if isinstance(robot.get("safe_min_raw_positions", {}), dict)
        else {}
    )
    safe_max = (
        robot.get("safe_max_raw_positions", {})
        if isinstance(robot.get("safe_max_raw_positions", {}), dict)
        else {}
    )
    open_raw = (
        robot.get("open_raw_positions", {}) if isinstance(robot.get("open_raw_positions", {}), dict) else {}
    )
    closed_raw = (
        robot.get("closed_raw_positions", {})
        if isinstance(robot.get("closed_raw_positions", {}), dict)
        else {}
    )
    result: dict[str, MotorCalibration] = {}
    for motor, motor_cfg in hand_motors.items():
        key = _hand_yaml_key(motor, side)
        lo = safe_min.get(key, min(float(open_raw[key]), float(closed_raw[key])))
        hi = safe_max.get(key, max(float(open_raw[key]), float(closed_raw[key])))
        result[motor] = MotorCalibration(
            id=motor_cfg.id,
            drive_mode=0,
            homing_offset=0,
            range_min=int(round(float(lo))),
            range_max=int(round(float(hi))),
        )
    return result


class SO101AmazingHandRight(SO101AmazingHandFollower):
    """Right-side calibrated SO-101 + AmazingHand robot.

    It reuses the existing SO101AmazingHand motor buses and calibration format,
    but provides right-side defaults, mode-specific feature sets, and arm-first
    send ordering so a hand write failure can be isolated by policy.
    """

    config_class = SO101AmazingHandRightConfig
    name = "so101_amazinghand_right"

    def __init__(self, config: SO101AmazingHandRightConfig):
        super().__init__(config)
        self.config = config
        self.side = config.side
        self.right_hand_disabled = False
        self.right_arm_disabled = False
        self.last_hand_error: str | None = None
        self.last_arm_error: str | None = None
        self.hand_error_history: list[str] = []
        self.arm_error_history: list[str] = []
        self.shutdown_warnings: list[str] = []
        self.hand_fault_motor_id: int | None = None
        self.hand_fault_key: str | None = None
        self.hand_fault_type: str | None = None
        self.hand_fault_first_timestamp: float | None = None
        self._quest_hand_calibration = _load_quest_hand_calibration(self.config.hand_calibration_file)
        if self.config.hand_calibration_source == "quest_yaml" and self._quest_hand_calibration:
            missing = _missing_quest_hand_features(self._quest_hand_calibration, side=self.side)
            if not missing:
                self.hand_bus.calibration = _quest_hand_motor_calibration(
                    self._quest_hand_calibration,
                    self.hand_bus.motors,
                    side=self.side,
                )

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        feats: dict = {_canonical_motor_key(m, self.side): float for m in _mode_motors(self.config.mode)}
        for cam_key, cam_cfg in self.config.cameras.items():
            feats[cam_key] = (cam_cfg.height, cam_cfg.width, 3)
        return feats

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {_canonical_motor_key(m, self.side): float for m in _mode_motors(self.config.mode)}

    @property
    def is_connected(self) -> bool:
        arm_ok = True if self.config.mode == "hand-only" else self.arm_bus.is_connected
        hand_ok = True if self.config.mode == "arm-only" else self.hand_bus.is_connected
        return arm_ok and hand_ok and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        status = self.calibration_status()
        if not status["arm_calibration_ok"]:
            return False
        return bool(status["hand_calibration_ok"])

    def calibration_status(self) -> dict[str, object]:
        required_lerobot = _required_lerobot_motors(self.config.mode, self.config.hand_calibration_source)
        missing_lerobot = [motor for motor in required_lerobot if motor not in self.calibration]
        missing_arm = [
            motor
            for motor in ARM_MOTORS
            if _needs_arm_calibration(self.config.mode) and motor not in self.calibration
        ]
        missing_hand = [
            motor
            for motor in HAND_MOTORS
            if _needs_hand_calibration(self.config.mode)
            and self.config.hand_calibration_source == "lerobot"
            and motor not in self.calibration
        ]
        missing_hand_features: list[str] = []
        hand_calibration_ok = True
        if _needs_hand_calibration(self.config.mode):
            if self.config.hand_calibration_source == "quest_yaml":
                if not self._quest_hand_calibration:
                    missing_hand_features = [str(self.config.hand_calibration_file)]
                else:
                    missing_hand_features = _missing_quest_hand_features(
                        self._quest_hand_calibration, side=self.side
                    )
                hand_calibration_ok = not missing_hand_features
            else:
                hand_calibration_ok = not missing_hand
        arm_calibration_ok = not missing_arm
        return {
            "arm_calibration_source": "lerobot",
            "arm_calibration_ok": arm_calibration_ok,
            "hand_calibration_source": self.config.hand_calibration_source,
            "hand_calibration_ok": hand_calibration_ok,
            "missing_arm_motors": missing_arm,
            "missing_hand_motors": missing_hand,
            "missing_lerobot_motors": missing_lerobot,
            "missing_hand_features": missing_hand_features,
            "robot_calibration_file": str(self.calibration_fpath),
            "hand_calibration_file": str(self.config.hand_calibration_file),
        }

    def _require_existing_calibration(self) -> None:
        if not self.config.require_calibration:
            return
        status = self.calibration_status()
        if not status["arm_calibration_ok"]:
            raise RuntimeError(
                f"{self.name} requires existing LeRobot calibration for SO-101 arm before real teleop. "
                f"Missing arm motors: {status['missing_arm_motors']}. Run lerobot-calibrate for the SO-101 arm."
            )
        if self.config.hand_calibration_source == "lerobot" and status["missing_hand_motors"]:
            raise RuntimeError(
                f"{self.name} requires existing LeRobot hand calibration before real teleop when "
                "--robot.hand-calibration-source=lerobot. "
                f"Missing hand motors: {status['missing_hand_motors']}. Run lerobot-calibrate first."
            )
        if (
            self.config.hand_calibration_source == "quest_yaml"
            and _needs_hand_calibration(self.config.mode)
            and status["missing_hand_features"]
        ):
            raise RuntimeError(
                f"{self.name} requires QuestHTS hand calibration YAML when "
                "--robot.hand-calibration-source=quest_yaml. "
                f"File: {self.config.hand_calibration_file}. "
                f"Missing hand features: {status['missing_hand_features']}."
            )

    @staticmethod
    def _clear_port_if_connected(handler) -> None:
        if handler is None:
            return
        if getattr(handler, "ser", object()) is None:
            return
        clear_port = getattr(handler, "clearPort", None)
        if clear_port is not None:
            clear_port()

    @staticmethod
    def _close_port_if_connected(handler) -> None:
        if handler is None:
            return
        if getattr(handler, "ser", object()) is None:
            return
        close_port = getattr(handler, "closePort", None)
        if close_port is not None:
            close_port()

    def _clear_after_hand_bus(self) -> None:
        handler = self._hand_port_handler if self._separate_hand_port else self._port_handler
        self._clear_port_if_connected(handler)

    def _clear_after_arm_bus(self) -> None:
        self._clear_port_if_connected(self._port_handler)

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        if calibrate:
            self._require_existing_calibration()

        opened_arm_port = False
        opened_hand_port = False
        try:
            if self.config.mode != "hand-only":
                if not self._port_handler.openPort():
                    raise ConnectionError(f"\nCould not open arm port '{self.config.port}'.")
                opened_arm_port = True
                _scs_def.SCS_END = 0
                self.arm_bus._connect(handshake=False)
                self.arm_bus.set_timeout()

            if self.config.mode != "arm-only":
                if self._separate_hand_port:
                    if not self._hand_port_handler.openPort():
                        raise ConnectionError(f"\nCould not open hand port '{self.config.hand_port}'.")
                    opened_hand_port = True
                elif not opened_arm_port:
                    if not self._port_handler.openPort():
                        raise ConnectionError(
                            f"\nCould not open hand port '{self.config.hand_port or self.config.port}'."
                        )
                    opened_arm_port = True
                with _big_endian():
                    self.hand_bus._connect(handshake=False)
                    self.hand_bus.set_timeout()
                if not self._separate_hand_port:
                    self._port_handler.clearPort()
                _scs_def.SCS_END = 0

            for cam in self.cameras.values():
                cam.connect()

            self.configure()
        except Exception:
            if opened_hand_port:
                self._hand_port_handler.closePort()
            if opened_arm_port:
                self._port_handler.closePort()
            raise
        logger.info("%s connected.", self)

    def configure(self) -> None:
        # Keep the existing robust configure path for arm+hand.  For single-side
        # modes, avoid touching disabled motors where possible.
        if self.config.mode == "arm-and-hand":
            super().configure()
            return
        if self.config.mode == "arm-only":
            self._clear_after_arm_bus()
            _scs_def.SCS_END = 0
            for motor in self.arm_bus.motors:
                self.arm_bus.write("Lock", motor, 0)
            with self.arm_bus.torque_disabled():
                self.arm_bus.configure_motors()
            self.arm_bus.enable_torque()
            return
        with _big_endian():
            self.hand_bus.sync_write("Torque_Enable", 0)
            self.hand_bus.sync_write("Torque_Enable", 1)
        self._clear_after_hand_bus()

    def calibrate(self) -> None:
        if self.config.mode == "arm-and-hand":
            super().calibrate()
            return

        mode_motors = _mode_motors(self.config.mode)
        existing_mode_calibration = {k: v for k, v in self.calibration.items() if k in mode_motors}
        if existing_mode_calibration and all(m in existing_mode_calibration for m in mode_motors):
            user_input = input(
                self._prompt(
                    f"Press ENTER to use existing calibration (id={self.id}, mode={self.config.mode}), "
                    "or type 'c' to run new calibration: "
                )
            )
            if user_input.strip().lower() != "c":
                if self.config.mode == "arm-only":
                    self.arm_bus.write_calibration(existing_mode_calibration)
                else:
                    with _big_endian():
                        self.hand_bus.disable_torque()
                        self.hand_bus.write_calibration(existing_mode_calibration)
                return

        if self.config.mode == "arm-only":
            self._calibrate_arm_only()
            return
        self._calibrate_hand_only()

    def _calibrate_arm_only(self) -> None:
        logger.info(self._prompt("Running arm-only SO-101 calibration ..."))

        self._clear_after_arm_bus()
        _scs_def.SCS_END = 0
        self.arm_bus.disable_torque()

        self._clear_after_arm_bus()
        _scs_def.SCS_END = 0

        input(self._prompt("Move SO-101 arm to the middle of its range and press ENTER ..."))

        self._clear_after_arm_bus()
        _scs_def.SCS_END = 0
        for motor, m in self.arm_bus.motors.items():
            max_res = self.arm_bus.model_resolution_table[m.model] - 1
            self.arm_bus.write("Homing_Offset", motor, 0, normalize=False)
            self.arm_bus.write("Min_Position_Limit", motor, 0, normalize=False)
            self.arm_bus.write("Max_Position_Limit", motor, max_res, normalize=False)
        self.arm_bus.calibration = {}

        time.sleep(0.1)
        self._clear_after_arm_bus()
        _scs_def.SCS_END = 0

        actual_positions: dict = {}
        for motor in self.arm_bus.motors:
            actual_positions[motor] = self.arm_bus.read("Present_Position", motor, normalize=False)
        homing_offsets = self.arm_bus._get_half_turn_homings(actual_positions)
        for motor, offset in homing_offsets.items():
            self.arm_bus.write("Homing_Offset", motor, offset, normalize=False)

        full_turn_motor = "wrist_roll"
        range_motors = [m for m in self.arm_bus.motors if m != full_turn_motor]
        print(
            self._prompt(
                "Move all arm joints (except wrist_roll) through full range. Press ENTER to stop ..."
            )
        )
        range_mins, range_maxes = self._record_arm_ranges(range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        arm_calibration = {}
        for motor, m in self.arm_bus.motors.items():
            arm_calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )
        self.arm_bus.write_calibration(arm_calibration)

        self.calibration = arm_calibration
        self._save_calibration()
        logger.info("Arm-only calibration saved -> %s", self.calibration_fpath)

    def _calibrate_hand_only(self) -> None:
        logger.info(self._prompt("Running hand-only AmazingHand calibration ..."))
        if self.config.auto_calibrate_hand:
            hand_calibration = self._auto_calibrate_hand()
        else:
            logger.info(self._prompt("Writing fixed position limits to AmazingHand ..."))
            hand_calibration = {}
            for motor, m in self.hand_bus.motors.items():
                hand_calibration[motor] = MotorCalibration(
                    id=m.id,
                    drive_mode=0,
                    homing_offset=0,
                    range_min=self.config.hand_range_min,
                    range_max=self.config.hand_range_max,
                )
            with _big_endian():
                self.hand_bus.disable_torque()
                self.hand_bus.write_calibration(hand_calibration)
        too_small = [
            motor
            for motor, calibration in hand_calibration.items()
            if abs(calibration.range_max - calibration.range_min) < _HAND_CAL_MIN_SPAN
        ]
        if too_small:
            raise ValueError(
                f"Some hand motors have a calibration span smaller than {_HAND_CAL_MIN_SPAN}: {too_small}"
            )

        self.calibration = hand_calibration
        self._save_calibration()
        logger.info("Hand-only calibration saved -> %s", self.calibration_fpath)

    @staticmethod
    def _is_fatal_hand_error(text: str) -> bool:
        return (
            "Overload error" in text or "Input voltage error" in text or "There is no status packet" in text
        )

    @staticmethod
    def _is_arm_overload_error(text: str) -> bool:
        return "Overload error" in text

    def _latch_hand_fault(self, motor: str, exc: Exception) -> None:
        """Record first fatal hand fault and disable further hand I/O."""
        text = str(exc)
        motors_dict = getattr(self.hand_bus, "motors", {}) if hasattr(self, "hand_bus") else {}
        motor_cfg = motors_dict.get(motor) if motor else None
        motor_id = getattr(motor_cfg, "id", None) if motor_cfg is not None else None
        if "Overload error" in text:
            fault_type = "Overload error"
        elif "Input voltage error" in text:
            fault_type = "Input voltage error"
        else:
            fault_type = text[:80]
        if self.hand_fault_first_timestamp is None:
            self.hand_fault_first_timestamp = time.monotonic()
            self.hand_fault_motor_id = motor_id
            self.hand_fault_key = _canonical_motor_key(motor, self.side)
            self.hand_fault_type = fault_type
        self.right_hand_disabled = True
        message = (
            f"hand fault ({fault_type}) on motor={motor!r} id={motor_id}; disabling hand I/O for this run"
        )
        self.last_hand_error = message
        self.hand_error_history.append(message)
        logger.warning(message)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: dict = {}
        if self.config.mode != "arm-only" and not self.right_hand_disabled:
            with _big_endian():
                for motor in self.hand_bus.motors:
                    if motor not in HAND_MOTORS:
                        continue
                    try:
                        obs[_canonical_motor_key(motor, self.side)] = self.hand_bus.read(
                            "Present_Position", motor
                        )
                    except Exception as exc:  # noqa: BLE001
                        text = str(exc)
                        if (
                            self._is_fatal_hand_error(text)
                            and self.config.hand_error_policy == "disable-hand"
                        ):
                            self._latch_hand_fault(motor, exc)
                        else:
                            logger.warning("Hand read [%s] failed: %s", motor, exc)
                        obs[_canonical_motor_key(motor, self.side)] = 0.0
                        if self.right_hand_disabled:
                            break
            self._clear_after_hand_bus()
        if self.config.mode != "hand-only":
            _scs_def.SCS_END = 0
            try:
                arm_pos = self.arm_bus.sync_read("Present_Position", num_retry=2)
            except Exception as sync_exc:  # noqa: BLE001
                if self._is_arm_overload_error(str(sync_exc)) and self.config.arm_error_policy == "abort":
                    raise
                arm_pos = {}
                for motor in self.arm_bus.motors:
                    try:
                        arm_pos[motor] = self.arm_bus.read("Present_Position", motor, num_retry=1)
                    except Exception as exc:  # noqa: BLE001
                        if self._is_arm_overload_error(str(exc)) and self.config.arm_error_policy == "abort":
                            raise
                        logger.warning("Arm read [%s] failed: %s", motor, exc)
                        arm_pos[motor] = 0.0
            obs.update({_canonical_motor_key(k, self.side): v for k, v in arm_pos.items() if k in ARM_MOTORS})
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        return {key: obs[key] for key in self.observation_features if key in obs}

    def should_abort_episode(self) -> bool:
        """Return True if a hand hardware fault occurred; hand_fault episodes must not be saved."""
        return self.hand_fault_type is not None

    def _handle_send_error(self, target: str, exc: Exception) -> None:
        text = str(exc)
        if target == "hand":
            if self._is_fatal_hand_error(text) and self.config.hand_error_policy == "disable-hand":
                motor_name = ""
                self._latch_hand_fault(motor_name, exc)
                return
            policy = self.config.hand_error_policy
            message = f"{self.side} hand send failed; policy={policy}: {text}"
            self.last_hand_error = message
            self.hand_error_history.append(message)
            if policy == "disable-hand":
                self.right_hand_disabled = True
                logger.warning(message)
                return
            if policy == "continue":
                logger.warning(message)
                return
        else:
            policy = self.config.arm_error_policy
            message = f"{self.side} arm send failed; policy={policy}: {text}"
            self.last_arm_error = message
            self.arm_error_history.append(message)
            if policy == "disable-arm":
                self.right_arm_disabled = True
                logger.warning(message)
                return
            if policy == "continue":
                logger.warning(message)
                return
        raise exc

    def _record_shutdown_warning(self, message: str) -> None:
        self.shutdown_warnings.append(message)
        logger.warning(message)
        print(f"WARNING: {message}", flush=True)

    def get_episode_metadata(self) -> dict[str, object]:
        status = self.calibration_status()
        return {
            **status,
            "human_side": self.side,
            "robot_side": self.side,
            "canonical_action_keys": list(self.action_features),
            "hand_disabled": self.right_hand_disabled,
            "arm_disabled": self.right_arm_disabled,
            "hand_disabled_due_to_fault": self.hand_fault_type is not None,
            "hand_fault_motor_id": self.hand_fault_motor_id,
            "hand_fault_key": self.hand_fault_key,
            "hand_fault_type": self.hand_fault_type,
            "hand_fault_first_timestamp": self.hand_fault_first_timestamp,
            "last_hand_error": self.last_hand_error,
            "last_arm_error": self.last_arm_error,
            "hand_error_history": list(self.hand_error_history),
            "arm_error_history": list(self.arm_error_history),
            "shutdown_warnings": list(self.shutdown_warnings),
        }

    def _best_effort_disable_arm_torque(self) -> None:
        if self.config.mode == "hand-only" or not getattr(self.arm_bus, "is_connected", False):
            return
        _scs_def.SCS_END = 0
        for motor, motor_cfg in self.arm_bus.motors.items():
            if motor not in ARM_MOTORS:
                continue
            motor_id = getattr(motor_cfg, "id", motor)
            try:
                self.arm_bus.write("Torque_Enable", motor, 0, normalize=False)
            except Exception as exc:  # noqa: BLE001
                self._record_shutdown_warning(
                    f"Failed to disable torque on arm motor ID {motor_id} during shutdown; "
                    "power off or inspect robot before next run."
                )
                logger.warning("Arm shutdown torque disable error detail for motor ID %s: %s", motor_id, exc)

    def _best_effort_disable_hand_torque(self) -> None:
        if self.config.mode == "arm-only" or not getattr(self.hand_bus, "is_connected", False):
            return
        with _big_endian():
            for motor, motor_cfg in self.hand_bus.motors.items():
                if motor not in HAND_MOTORS:
                    continue
                motor_id = getattr(motor_cfg, "id", motor)
                try:
                    self.hand_bus.write("Torque_Enable", motor, 0, normalize=False)
                except Exception as exc:  # noqa: BLE001
                    self._record_shutdown_warning(
                        f"Failed to disable torque on hand motor ID {motor_id} during shutdown; "
                        "power off or inspect robot before next run."
                    )
                    logger.warning(
                        "Hand shutdown torque disable error detail for motor ID %s: %s", motor_id, exc
                    )

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {}
        for key, value in action.items():
            motor = _motor_name_from_action_key(key, self.side)
            if motor is not None:
                goal_pos[motor] = value
        arm_goal = {k: v for k, v in goal_pos.items() if k in ARM_MOTORS and self.config.mode != "hand-only"}
        hand_goal = {k: v for k, v in goal_pos.items() if k in HAND_MOTORS and self.config.mode != "arm-only"}
        sent: dict[str, float] = {}

        # Arm first: a hand bus error must not hide a successful arm command.
        if arm_goal and not self.right_arm_disabled:
            try:
                _scs_def.SCS_END = 0
                self.arm_bus.sync_write("Goal_Position", arm_goal)
                sent.update({_canonical_motor_key(k, self.side): v for k, v in arm_goal.items()})
            except Exception as exc:  # noqa: BLE001
                self._handle_send_error("arm", exc)

        if hand_goal and not self.right_hand_disabled:
            try:
                with _big_endian():
                    self.hand_bus.sync_write("Goal_Position", hand_goal)
                self._clear_after_hand_bus()
                sent.update({_canonical_motor_key(k, self.side): v for k, v in hand_goal.items()})
            except Exception as exc:  # noqa: BLE001
                self._handle_send_error("hand", exc)

        return sent

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.config.disable_torque_on_disconnect:
            self._best_effort_disable_arm_torque()
            self._best_effort_disable_hand_torque()
        if self.config.mode != "hand-only" or not self._separate_hand_port:
            try:
                self._close_port_if_connected(self._port_handler)
            except Exception as exc:  # noqa: BLE001
                self._record_shutdown_warning(f"Failed to close arm port during shutdown: {exc}")
        if self.config.mode != "arm-only" and self._separate_hand_port:
            try:
                self._close_port_if_connected(self._hand_port_handler)
            except Exception as exc:  # noqa: BLE001
                self._record_shutdown_warning(f"Failed to close hand port during shutdown: {exc}")
        for cam_key, cam in self.cameras.items():
            try:
                cam.disconnect()
            except Exception as exc:  # noqa: BLE001
                self._record_shutdown_warning(f"Failed to disconnect camera {cam_key} during shutdown: {exc}")
        logger.info("%s disconnected.", self)
