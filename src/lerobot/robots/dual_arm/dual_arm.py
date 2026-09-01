#!/usr/bin/env python

"""Dual-Arm SO-101 + AmazingHand follower (26 DOF).

Right arm: r_shoulder_pan … r_finger4_motor2  (13 motors on port_r)
Left arm:  l_shoulder_pan … l_finger4_motor2  (13 motors on port_l)

Each side has five SO-101 arm motors and eight AmazingHand motors. Arm and hand buses may use
separate USB adapters; both arms can also share one arm bus when their ID ranges are disjoint.
"""

import logging
import os
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..so101_amazinghand.config_so101_amazinghand import SO101AmazingHandFollowerConfig
from ..so101_amazinghand.so101_amazinghand_follower import (
    ALL_MOTORS,
    SO101AmazingHandFollower,
    _make_shared_port_handler,
)
from .config_dual_arm import DualArmConfig

logger = logging.getLogger(__name__)

R_MOTORS = [f"r_{m}" for m in ALL_MOTORS]
L_MOTORS = [f"l_{m}" for m in ALL_MOTORS]
DUAL_ARM_MOTORS = R_MOTORS + L_MOTORS  # 26 total


class DualArm(Robot):
    """26-DOF dual-arm follower (right + left SO-101 + AmazingHand)."""

    config_class = DualArmConfig
    name = "dual_arm"

    def __init__(self, config: DualArmConfig):
        super().__init__(config)
        self.config = config

        sides = str(config.sides).lower()
        if sides not in ("both", "right", "left"):
            raise ValueError(f"sides must be 'both', 'right', or 'left' — got {config.sides!r}")
        self._sides = sides
        want_r = sides in ("both", "right")
        want_l = sides in ("both", "left")

        # Split calibration by prefix
        r_cal = {m.removeprefix("r_"): v for m, v in self.calibration.items() if m.startswith("r_")}
        l_cal = {m.removeprefix("l_"): v for m, v in self.calibration.items() if m.startswith("l_")}

        # SINGLE-PORT mode: if both arm ports resolve to the same physical device, the two arms
        # live on ONE RS485 bus. They must use non-colliding ID sets and share ONE PortHandler;
        # opening the same /dev twice causes
        # packet collisions ("Incorrect status packet"). The shared handler is owned here and closed
        # once in disconnect(), after BOTH sub-arms have disabled torque.
        def _resolve(p: str) -> str:
            try:
                return os.path.realpath(p)
            except OSError:
                return p

        # Arm motor IDs come from config regardless of bus topology.
        self._single_arm_port = want_r and want_l and _resolve(config.port_r) == _resolve(config.port_l)
        r_arm_ids, l_arm_ids = tuple(config.arm_ids_r), tuple(config.arm_ids_l)
        if self._single_arm_port:
            overlapping_ids = sorted(set(r_arm_ids) & set(l_arm_ids))
            if overlapping_ids:
                raise ValueError(
                    "Both arms resolve to the same serial port but their motor IDs overlap: "
                    f"{overlapping_ids}. Use separate arm adapters or configure non-overlapping "
                    "arm_ids_r and arm_ids_l."
                )
            self._shared_arm_ph = _make_shared_port_handler(config.port_r)
            logger.info(
                "Single-port mode: both arms on %s (right IDs %s, left IDs %s), shared PortHandler.",
                config.port_r,
                r_arm_ids,
                l_arm_ids,
            )
        else:
            self._shared_arm_ph = None
            if want_r and want_l:
                logger.info(
                    "Two-port mode: right arm on %s IDs %s, left arm on %s IDs %s.",
                    config.port_r,
                    r_arm_ids,
                    config.port_l,
                    l_arm_ids,
                )
            else:
                only = "right" if want_r else "left"
                only_port = config.port_r if want_r else config.port_l
                only_ids = r_arm_ids if want_r else l_arm_ids
                logger.info("Single-arm mode (%s only): arm on %s IDs %s.", only, only_port, only_ids)

        def _sub_cfg(port, hand_port, arm_ids, suffix):
            return SO101AmazingHandFollowerConfig(
                port=port,
                hand_port=hand_port,
                arm_motor_ids=arm_ids,
                # Per-side calibration files remain r.json/l.json regardless of the outer
                # Dual-Arm id (dual_arm.json stores the merged calibration).
                id=suffix,
                calibration_dir=config.calibration_dir,
                disable_torque_on_disconnect=config.disable_torque_on_disconnect,
                max_relative_target=config.max_relative_target,
                use_degrees=config.use_degrees,
                hand_range_min=config.hand_range_min,
                hand_range_max=config.hand_range_max,
                auto_calibrate_hand=config.auto_calibrate_hand,
                arm_p_coefficient=config.arm_p_coefficient,
                arm_p_coefficient_lift=config.arm_p_coefficient_lift,
                arm_d_coefficient=config.arm_d_coefficient,
                hand_protective_torque=config.hand_protective_torque,
                hand_protection_time=config.hand_protection_time,
                hand_p_coefficient=config.hand_p_coefficient,
                hand_max_torque_limit=config.hand_max_torque_limit,
            )

        # Build ONLY the active sub-arm(s). In single-port mode both share self._shared_arm_ph and
        # must NOT close it themselves (owns_arm_port=False); this follower closes it once in
        # disconnect(). In single-arm or two-port mode each arm owns its own port.
        self.r_arm = None
        self.l_arm = None
        if want_r:
            self.r_arm = SO101AmazingHandFollower(
                _sub_cfg(config.port_r, config.hand_port_r, r_arm_ids, "r"),
                arm_port_handler=self._shared_arm_ph,
                owns_arm_port=not self._single_arm_port,
            )
            # Override per-arm calibration only when the merged file provided data; otherwise keep
            # what the sub-arm loaded from its own JSON.
            if r_cal:
                self.r_arm.calibration = r_cal
        if want_l:
            self.l_arm = SO101AmazingHandFollower(
                _sub_cfg(config.port_l, config.hand_port_l, l_arm_ids, "l"),
                arm_port_handler=self._shared_arm_ph,
                owns_arm_port=not self._single_arm_port,
            )
            if l_cal:
                self.l_arm.calibration = l_cal

        self.cameras = make_cameras_from_configs(config.cameras)

    def _arms(self) -> list[tuple[str, "SO101AmazingHandFollower"]]:
        """Active (prefix, sub-arm) pairs in right-then-left order."""
        pairs = []
        if self.r_arm is not None:
            pairs.append(("r_", self.r_arm))
        if self.l_arm is not None:
            pairs.append(("l_", self.l_arm))
        return pairs

    @property
    def _active_motors(self) -> list[str]:
        motors: list[str] = []
        if self.r_arm is not None:
            motors += R_MOTORS
        if self.l_arm is not None:
            motors += L_MOTORS
        return motors

    # ── Features ──────────────────────────────────────────────────────────────

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        feats: dict = {f"{m}.pos": float for m in self._active_motors}
        for cam_key, cam_cfg in self.config.cameras.items():
            feats[cam_key] = (cam_cfg.height, cam_cfg.width, 3)
        return feats

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in self._active_motors}

    # ── Connection ────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return all(arm.is_connected for _, arm in self._arms()) and all(
            cam.is_connected for cam in self.cameras.values()
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        for _, arm in self._arms():
            arm.connect(calibrate=calibrate)

        # After connecting (and possibly calibrating) both arms, persist a merged
        # Dual-Arm-level calibration file so subsequent runs skip per-arm calibration entirely.
        # SINGLE-ARM mode: do NOT save — the merged file holds the OTHER side's arm + both hands'
        # calibration, and saving a one-side dict here would clobber it (the stale-merge footgun).
        if self._sides == "both":
            merged = {f"r_{k}": v for k, v in self.r_arm.calibration.items()}
            merged.update({f"l_{k}": v for k, v in self.l_arm.calibration.items()})
            if merged and merged != self.calibration:
                self.calibration = merged
                self._save_calibration()
                logger.info(f"Merged calibration saved → {self.calibration_fpath}")

        for cam in self.cameras.values():
            cam.connect()
        n_dof = 13 * len(self._arms())
        logger.info(f"{self} connected ({n_dof} DOF: {self._sides})")

    # ── Calibration ───────────────────────────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        return all(arm.is_calibrated for _, arm in self._arms())

    def configure(self) -> None:
        for _, arm in self._arms():
            arm.configure()

    def calibrate(self) -> None:
        if self.r_arm is not None:
            print("=== Calibrating RIGHT arm ===")
            self.r_arm.calibrate()
        if self.l_arm is not None:
            print("=== Calibrating LEFT arm ===")
            self.l_arm.calibrate()

        # Merge calibrations with prefix, save as single file. SINGLE-ARM mode: skip the merged save
        # so the unused side's (and the hands') calibration in the merged file is preserved.
        if self._sides == "both":
            merged = {}
            for k, v in self.r_arm.calibration.items():
                merged[f"r_{k}"] = v
            for k, v in self.l_arm.calibration.items():
                merged[f"l_{k}"] = v
            self.calibration = merged
            self._save_calibration()

    # ── Observation ───────────────────────────────────────────────────────────

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs = {}
        for prefix, arm in self._arms():
            for k, v in arm.get_observation().items():
                obs[f"{prefix}{k}"] = v

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    # ── Action ────────────────────────────────────────────────────────────────

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        sent = {}
        for prefix, arm in self._arms():
            sub_action = {k.removeprefix(prefix): v for k, v in action.items() if k.startswith(prefix)}
            if sub_action:
                for k, v in arm.send_action(sub_action).items():
                    sent[f"{prefix}{k}"] = v
        return sent

    # ── Disconnect ────────────────────────────────────────────────────────────

    @check_if_not_connected
    def disconnect(self) -> None:
        # Isolate each arm: if r_arm.disconnect() raises (e.g. hand-bus "Port is in use!" after a
        # Ctrl+C), l_arm MUST still disconnect — otherwise the left arm keeps its torque on (stiff).
        for prefix, arm in self._arms():
            try:
                arm.disconnect()
            except Exception as exc:
                logger.warning(f"{self}: {prefix}arm.disconnect() failed: {exc}")
        # Single-port mode: both sub-arms left the shared arm PortHandler open (owns_arm_port=False)
        # so each could disable its torque first. Now that BOTH are released, close it once.
        if self._single_arm_port and self._shared_arm_ph is not None:
            try:
                self._shared_arm_ph.closePort()
            except Exception as exc:
                logger.warning(f"{self}: shared arm port closePort failed: {exc}")
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception as exc:
                logger.warning(f"{self}: camera disconnect failed: {exc}")
        logger.info(f"{self} disconnected.")
