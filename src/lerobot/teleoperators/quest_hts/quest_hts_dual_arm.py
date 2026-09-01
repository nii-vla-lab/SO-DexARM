#!/usr/bin/env python
"""DualArm Quest HTS teleoperator.

Receives Quest HTS TCP data (left + right) and returns 26-DOF LeRobot actions:
  r_shoulder_pan.pos … r_finger4_motor2.pos   (13 joints, right)
  l_shoulder_pan.pos … l_finger4_motor2.pos   (13 joints, left)

Arm control: constrained planar IK (shoulder_lift + elbow_flex) anchored at
saved start position.  The remaining arm joints (shoulder_pan, wrist_flex,
wrist_roll) are held at their start values.

Hand control: piecewise-linear mapping from Quest landmarks to AmazingHand
motor targets, calibrated with fist / mid / open poses.
"""

from __future__ import annotations

import contextlib
import logging
import math
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.model.so101_kinematics import (
    SO101Kinematics,
    SO101LeRobotCalibration,
    TcpOffset,
    invert_transform,
    transform_from_xyz_quat,
)
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.utils import enter_pressed

from .config_quest_hts_dual_arm import QuestHTSDualArmTeleoperatorConfig

logger = logging.getLogger(__name__)

# Frame alignment: Quest world frame → Robot base frame.
# Verified with axis_map_diagnostic.py --capture:
# the Quest streams BOTH hands in ONE shared Unity (left-handed) world frame:
#   Quest +x = operator's RIGHT (lateral)   Quest +y = UP   Quest +z = FORWARD (toward robot)
# Mapping (robot = R @ quest):
#   Robot X (radial / forward) ← Quest +z
#   Robot Y (lateral)          ← Quest +x   (operator-right = +Y; flip planar_scale_z if the
#                                            gripper pans the wrong way — that's a motor-dir sign)
#   Robot Z (up)               ← Quest +y
# det = +1 (proper rotation). Single shared matrix: both hands use the SAME frame.
_R_QUEST_TO_ROBOT: np.ndarray = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=float,
)

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
HAND_MOTORS = tuple(f"finger{i}_motor{j}" for i in range(1, 5) for j in range(1, 3))
ARM_KEYS_R = tuple(f"r_{j}.pos" for j in ARM_JOINTS)
ARM_KEYS_L = tuple(f"l_{j}.pos" for j in ARM_JOINTS)
HAND_KEYS_R = tuple(f"r_{m}.pos" for m in HAND_MOTORS)
HAND_KEYS_L = tuple(f"l_{m}.pos" for m in HAND_MOTORS)
UNPREFIXED_ARM_KEYS = tuple(f"{j}.pos" for j in ARM_JOINTS)
UNPREFIXED_HAND_KEYS = tuple(f"{m}.pos" for m in HAND_MOTORS)


class _OneEuroFilter:
    """One-Euro filter (Casiez, Roussel, Vogel 2012) for an N-D vector signal.

    A low-pass filter whose cutoff frequency rises with the signal speed: it smooths hard when
    the input is slow/still (removing jitter) and barely at all when the input moves fast (so it
    stays responsive with almost no lag). Standard fix for jittery hand-tracking input.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x
        dt = t - self._t_prev
        if dt <= 0.0:
            dt = 1e-3
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


# ── Per-side runtime state ──────────────────────────────────────────────────


@dataclass
class _SideState:
    side: str  # "right" or "left"
    latest_wrist: tuple | None = None  # (x,y,z, qx,qy,qz,qw) Quest world frame
    latest_wrist_s: float | None = None  # monotonic timestamp
    latest_landmarks: tuple | None = None  # 63 floats (21 pts × xyz)
    latest_landmarks_s: float | None = None
    wrist_history: list = field(default_factory=list)  # recent wrist samples for baseline
    wrist_baseline_pose: Any = None  # 4×4 np.ndarray, captured at session start
    baseline_captured: bool = False
    start_action: dict[str, float] = field(default_factory=dict)  # unprefixed arm keys
    start_ee: dict | None = None  # {"x": m, "y": m} robot EE at start
    start_pan_deg: float = 0.0  # shoulder_pan at start position (degrees)
    start_tcp_pose: Any = None  # 4×4 np.ndarray flange pose at start (for full 3D IK)
    start_joint_degrees: dict[str, float] = field(default_factory=dict)  # all 5 arm joints at start (degrees)
    last_valid_ik_action: dict | None = None
    last_controlled_action: dict | None = None
    last_feedback: dict[str, float] = field(default_factory=dict)  # unprefixed
    arm_calibration: Any = None  # SO101LeRobotCalibration
    kinematics: Any = None  # SO101Kinematics
    hand_mapper: Any = None  # PiecewiseHandMapper (dual_arm_calibration)
    hand_start_action: dict[str, float] = field(default_factory=dict)  # prefixed hand keys at start
    hand_open_action: dict[str, float] = field(
        default_factory=dict
    )  # prefixed hand keys for "open"(パー) pose
    hand_fist_action: dict[str, float] = field(
        default_factory=dict
    )  # prefixed hand keys for "fist"(グー) pose — grip over-close reference
    last_hand_action: dict[str, float] = field(default_factory=dict)  # previous tick's hand cmd (EMA state)
    elbow_baseline_pos_quest: Any = None  # np.ndarray (3,) elbow pos at baseline, Quest world frame
    start_elbow_robot: Any = None  # np.ndarray (2,) elbow (forward, up) in robot sagittal at start
    wrist_filter: Any = None  # _OneEuroFilter for the wrist position (anti-jitter)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _wrist_to_transform(wrist: tuple) -> np.ndarray:
    return transform_from_xyz_quat(*wrist)


def _estimate_elbow_quest(
    wrist: tuple,
    forearm_axis: tuple,
    forearm_length_m: float,
) -> np.ndarray:
    """Estimate elbow position in Quest world frame from wrist pose.

    forearm_axis is the local axis of the Quest wrist frame that points FROM wrist TOWARD elbow.
    """
    from scipy.spatial.transform import Rotation

    pos = np.array(wrist[:3], dtype=float)
    quat = np.array(wrist[3:7], dtype=float)  # (qx, qy, qz, qw)
    axis = np.array(forearm_axis, dtype=float)
    elbow_dir = Rotation.from_quat(quat).apply(axis)
    return pos + elbow_dir * forearm_length_m


def _project_to_reach(
    x: float, y: float, l1: float, l2: float, max_frac: float = 0.95
) -> tuple[float, float]:
    r = (x * x + y * y) ** 0.5
    # Cap below full extension (l1+l2) so the elbow never locks straight (singularity).
    max_r = max_frac * (l1 + l2)
    if r > max_r and r > 1e-9:
        s = max_r / r
        x, y = x * s, y * s
    min_r = abs(l1 - l2) + 0.001
    r2 = (x * x + y * y) ** 0.5
    if r2 < min_r:
        s = min_r / r2 if r2 > 1e-9 else 1.0
        x, y = x * s, y * s
    return x, y


def _scale_rotation(rotation: np.ndarray, scale: float) -> np.ndarray:
    """Interpolate a rotation from identity by factor `scale` via angle-axis SLERP.

    scale=0 → identity (no orientation tracking)
    scale=1 → R unchanged (full tracking)
    scale=-1 → inverted rotation (flip direction if wrist_roll is reversed on your setup)
    """
    if abs(scale) < 1e-9:
        return np.eye(3, dtype=float)
    if abs(scale - 1.0) < 1e-9:
        return rotation.copy()
    cos_a = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) / 2.0))
    angle = math.acos(cos_a)
    if angle < 1e-9:
        return np.eye(3, dtype=float)
    sin_a = math.sin(angle)
    ax = float(rotation[2, 1] - rotation[1, 2]) / (2.0 * sin_a)
    ay = float(rotation[0, 2] - rotation[2, 0]) / (2.0 * sin_a)
    az = float(rotation[1, 0] - rotation[0, 1]) / (2.0 * sin_a)
    sa = angle * scale
    c, s, t = math.cos(sa), math.sin(sa), 1.0 - math.cos(sa)
    return np.array(
        [
            [t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay],
            [t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax],
            [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c],
        ],
        dtype=float,
    )


def _strip_prefix(key: str) -> str:
    if key.startswith(("r_", "l_")):
        return key[2:]
    return key


# ── Main teleoperator ───────────────────────────────────────────────────────


class QuestHTSDualArmTeleoperator(Teleoperator):
    """26-DOF dual_arm teleoperator driven by Meta Quest hand tracking (TCP)."""

    config_class = QuestHTSDualArmTeleoperatorConfig
    name = "quest_hts_dual_arm"

    def __init__(self, config: QuestHTSDualArmTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._server_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._connected = False
        # When paused, get_action() is not called by the record loop (is_control_ready()=False),
        # so the robot HOLDS its last pose — arm AND hands freeze. Used between episodes so moving
        # your real hands to reset the environment doesn't drive the robot.
        self._paused = False

        self._right = _SideState(side="right")
        self._left = _SideState(side="left")

        self._init_side(self._right)
        self._init_side(self._left)

    # ── Initialisation ───────────────────────────────────────────────────────

    def _init_side(self, state: _SideState) -> None:
        side = state.side
        calib_file = self.config.right_arm_calib_file if side == "right" else self.config.left_arm_calib_file
        try:
            state.arm_calibration = SO101LeRobotCalibration.from_file(
                calib_file, action_units="range_m100_100"
            )
            logger.info("Loaded %s arm calibration from %s", side, calib_file)
        except Exception as exc:
            logger.warning("Cannot load %s arm calibration from %s: %s", side, calib_file, exc)
            state.arm_calibration = None

        state.kinematics = SO101Kinematics()
        state.wrist_filter = _OneEuroFilter(
            min_cutoff=self.config.wrist_filter_min_cutoff,
            beta=self.config.wrist_filter_beta,
        )

        state.start_action = self._load_start_action(side)
        if state.arm_calibration and state.start_action:
            try:
                deg = state.arm_calibration.action_dict_to_degrees(state.start_action)
                # Anchor the control frame at the AmazingHand FINGERTIP (TCP), not the wrist flange,
                # so hand deltas map onto fingertip motion (see tcp_offset_m). With offset 0 this is
                # exactly the flange planar position (forward_kinematics).
                tcp_off = TcpOffset(x=float(self.config.tcp_offset_m))
                start_tcp = state.kinematics.forward_tcp_pose(deg, tcp_off)
                x = math.hypot(float(start_tcp[0, 3]), float(start_tcp[1, 3]))  # fingertip radial
                y = float(start_tcp[2, 3])  # fingertip vertical
                state.start_ee = {"x": float(x), "y": float(y)}
                state.start_pan_deg = float(deg.get("shoulder_pan", 0.0))
                state.start_tcp_pose = start_tcp
                state.start_joint_degrees = {k: float(v) for k, v in deg.items()}
                logger.info("%s start EE: x=%.4f m  y=%.4f m  pan=%.2f°", side, x, y, state.start_pan_deg)
            except Exception as exc:
                logger.warning("Cannot compute %s start EE: %s", side, exc)
                state.start_ee = None

        state.hand_mapper = self._load_hand_mapper(side)
        state.hand_open_action = self._load_hand_open_action(side)
        state.hand_fist_action = self._load_hand_pose_action(side, "fist")  # grip over-close reference

    def _load_hand_pose_action(self, side: str, pose: str) -> dict[str, float]:
        """Robot motor positions for a calibrated hand pose ('open' パー / 'fist' グー / 'mid').

        Returns prefixed motor keys (r_/l_) → calibrated positions. The poses are NOT uniform
        (odd/even motors invert), so the per-motor calibrated values must be used.
        """
        try:
            from lerobot.teleoperators.quest_hts.dual_arm_calibration import read_yaml

            data = read_yaml(self.config.hand_calib_file)
            positions = data.get(side, {}).get("robot", {}).get(pose, {}).get("positions", {})
            return {str(k): float(v) for k, v in positions.items() if str(k).endswith(".pos")}
        except Exception as exc:
            logger.warning(
                "Cannot load %s hand %s pose from %s: %s", side, pose, self.config.hand_calib_file, exc
            )
            return {}

    def _load_hand_open_action(self, side: str) -> dict[str, float]:
        """Robot 'open' (パー) hand pose for this side, from the hand calibration file."""
        return self._load_hand_pose_action(side, "open")

    def _load_start_action(self, side: str) -> dict[str, float]:
        try:
            from lerobot.teleoperators.quest_hts.dual_arm_calibration import load_start_position

            payload = load_start_position(self.config.startup_file, side)
            arm_pos = payload.get("arm_positions", {})
            # Strip side prefix if present (stored with or without prefix)
            return {_strip_prefix(k): float(v) for k, v in arm_pos.items() if k.endswith(".pos")}
        except Exception as exc:
            logger.warning("Cannot load %s start action from %s: %s", side, self.config.startup_file, exc)
            return {}

    def _load_hand_mapper(self, side: str):
        try:
            from lerobot.teleoperators.quest_hts.dual_arm_calibration import (
                build_hand_mapping,
                mapper_from_mapping_payload,
            )

            mapping = build_hand_mapping(side=side, hand_calib_file=self.config.hand_calib_file)
            mapper = mapper_from_mapping_payload(mapping, side)
            logger.info("Loaded %s hand mapper from %s", side, self.config.hand_calib_file)
            return mapper
        except Exception as exc:
            logger.warning("Cannot load %s hand mapper from %s: %s", side, self.config.hand_calib_file, exc)
            return None

    # ── Teleoperator interface ───────────────────────────────────────────────

    # ── Active-side selection (single-side support) ───────────────────────────
    def _active_side_names(self) -> tuple[str, ...]:
        s = str(getattr(self.config, "active_sides", "both")).lower()
        if s == "right":
            return ("right",)
        if s == "left":
            return ("left",)
        return ("right", "left")

    def _active_states(self) -> list[tuple[_SideState, str]]:
        names = self._active_side_names()
        return [
            (state, prefix)
            for state, prefix in ((self._right, "r_"), (self._left, "l_"))
            if state.side in names
        ]

    @property
    def action_features(self) -> dict[str, type]:
        keys: dict[str, type] = {}
        names = self._active_side_names()
        arm_keys: tuple[str, ...] = ()
        hand_keys: tuple[str, ...] = ()
        if "right" in names:
            arm_keys += ARM_KEYS_R
            hand_keys += HAND_KEYS_R
        if "left" in names:
            arm_keys += ARM_KEYS_L
            hand_keys += HAND_KEYS_L
        if self.config.mode != "hand-only":
            keys.update(dict.fromkeys(arm_keys, float))
        if self.config.mode != "arm-only":
            keys.update(dict.fromkeys(hand_keys, float))
        return keys

    @property
    def feedback_features(self) -> dict[str, type]:
        return dict.fromkeys(ARM_KEYS_R + ARM_KEYS_L + HAND_KEYS_R + HAND_KEYS_L, float)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        self._init_side(self._right)
        self._init_side(self._left)

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self._stop_event.clear()
        if self.config.start_receiver:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.config.host, self.config.port))
            server.listen(5)  # Both Hands mode opens 2 simultaneous TCP connections
            server.settimeout(0.5)
            self._server_sock = server
            self._thread = threading.Thread(
                target=self._tcp_server_loop,
                args=(server,),
                daemon=True,
                name="quest_hts_dual_arm_tcp",
            )
            self._thread.start()
            print(
                f"[QuestHTSDualArm] TCP server listening on {self.config.host}:{self.config.port}",
                flush=True,
            )
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

    def send_feedback(self, feedback: dict) -> None:
        with self._lock:
            for state, prefix in self._active_states():
                state.last_feedback.update(
                    {
                        _strip_prefix(k): float(v)
                        for k, v in feedback.items()
                        if k.startswith(prefix) and k.endswith(".pos") and isinstance(v, int | float)
                    }
                )

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        action: dict[str, float] = {}
        with self._lock:
            for state, prefix in self._active_states():
                if self.config.mode != "hand-only":
                    arm = self._build_arm_action_locked(state)
                    for k, v in arm.items():
                        action[f"{prefix}{k}" if not k.startswith(prefix) else k] = float(v)
                if self.config.mode != "arm-only":
                    hand = self._build_hand_action_locked(state)
                    action.update({k: float(v) for k, v in hand.items()})
        return action

    # ── Session setup ────────────────────────────────────────────────────────

    def _await_baseline_capture(self, sides_hint: str) -> None:
        """Wait for baseline capture: countdown timer or manual ENTER.

        If baseline_countdown_s > 0: show a live countdown; ENTER skips remaining wait.
        If baseline_countdown_s == 0: block on ENTER (original behaviour).
        """
        countdown = self.config.baseline_countdown_s
        if countdown <= 0:
            print(
                f"\n[QuestHTSDualArm] Hold {sides_hint} at the robot start pose,"
                " then press ENTER to capture baselines.",
                flush=True,
            )
            input()
            return

        print(
            f"\n[QuestHTSDualArm] Hold {sides_hint} at the robot start pose.",
            flush=True,
        )
        print(
            f"  Auto-capturing baseline in {countdown:.0f}s  (press ENTER to capture now)",
            flush=True,
        )
        deadline = time.monotonic() + countdown
        last_shown: int = -1
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print("\r  Capturing now...                              ", flush=True)
                break
            secs = int(math.ceil(remaining))
            if secs != last_shown:
                print(
                    f"\r  Auto-capturing in {secs:2d}s  (press ENTER to capture now) ",
                    end="",
                    flush=True,
                )
                last_shown = secs
            if enter_pressed():
                print("\r  ENTER pressed — capturing now.                ", flush=True)
                break
            time.sleep(0.05)

    def begin_episode(self, robot=None, events=None) -> None:
        """Re-arm before each record episode: move the robot back to the saved start pose,
        open both AmazingHands (パー), and re-capture the wrist baseline so teleop resumes
        cleanly from start.

        Called by lerobot_record at the start of every episode after the first (the first is
        already prepared by prepare_control_session() right after connect()).

        This is intentionally LIGHTWEIGHT — it does NOT re-run prepare_control_session (which
        waits for the stream and runs the interactive 10s baseline countdown). Between episodes
        we just (1) move to start, then (2) auto re-baseline against the hand's CURRENT pose so
        'current hand = start pose' and the arm holds at start until the operator moves. Without
        the re-baseline the first get_action() would diff the current hand against the OLD
        baseline and jerk the arm off the start pose.
        """
        print("[QuestHTSDualArm] begin_episode: returning to start pose (+ open hands)...", flush=True)
        # Invalidate baselines first so a stray get_action() can't drive the arm mid-move.
        with self._lock:
            self._right.baseline_captured = False
            self._left.baseline_captured = False
            # Reset finger-smoothing state so we don't blend across the open-hand reset.
            self._right.last_hand_action = {}
            self._left.last_hand_action = {}

        if robot is not None:
            # Refresh feedback so move-to-start interpolates from the true current pose.
            obs = robot.get_observation()
            self.send_feedback({k: v for k, v in obs.items() if isinstance(v, int | float)})
            if self.config.move_to_start:
                self._move_to_start(robot)
                # Refresh feedback so the post-move (start + open-hand) pose is the hand baseline.
                obs = robot.get_observation()
                self.send_feedback({k: v for k, v in obs.items() if isinstance(v, int | float)})
                print("[QuestHTSDualArm] begin_episode: start pose reached.", flush=True)
            else:
                print("[QuestHTSDualArm] begin_episode: move_to_start disabled — skipping move.", flush=True)
        else:
            print("[QuestHTSDualArm] begin_episode: no robot handle — cannot move to start!", flush=True)

        # Alignment gate: stay PAUSED (robot frozen at start) until the operator presses → AGAIN.
        # We reuse the → (right-arrow) key the record workflow already uses — NOT ENTER, because
        # both hands are busy being tracked and can't reach the keyboard reliably. The second →
        # triggers the hand re-baseline below so tracking restarts without a jump.
        self._await_alignment(events)

        # Auto re-baseline immediately (no interactive countdown between episodes).
        with self._lock:
            for state in (self._right, self._left):
                if state.latest_wrist is not None:
                    state.wrist_baseline_pose = self._baseline_pose_from_history(state)
                    state.baseline_captured = True
                    state._stale_warned = False
                    if self.config.use_elbow_ik:
                        state.elbow_baseline_pos_quest = _estimate_elbow_quest(
                            state.latest_wrist, self.config.forearm_axis, self.config.forearm_length_m
                        )
                    prefix = "r_" if state.side == "right" else "l_"
                    state.hand_start_action = {
                        f"{prefix}{k}": float(v)
                        for k, v in state.last_feedback.items()
                        if k.startswith("finger") and k.endswith(".pos")
                    }
        ready = [s for s, st in (("right", self._right), ("left", self._left)) if st.baseline_captured]
        # Resume tracking now that we're at start, aligned, and re-baselined.
        self._paused = False
        print(
            f"[QuestHTSDualArm] begin_episode: aligned + baseline re-captured for {ready} — RESUMED.",
            flush=True,
        )

    # ── Pause / freeze control (between episodes) ─────────────────────────────
    def is_control_ready(self) -> bool:
        """record_loop skips get_action() (robot HOLDS — arm + hands freeze) when this is False."""
        return not self._paused

    def is_paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def end_episode(self, robot=None) -> None:
        """Freeze tracking at the END of an episode (→ exit-early / ← re-record / timeout) so the
        operator can move their real hands to reset the environment WITHOUT the robot mimicking.
        Stays frozen through the reset window; begin_episode() resumes it after the alignment gate.
        """
        self._paused = True
        print(
            "[QuestHTSDualArm] end_episode: tracking PAUSED — robot frozen (arm + hands). "
            "Reset the environment freely.",
            flush=True,
        )

    def _await_alignment(self, events=None) -> None:
        """Block (robot frozen at start) until the operator presses → (right arrow) AGAIN.

        Resume trigger is → (events['exit_early'] set by the record keyboard listener), NOT ENTER —
        both hands are occupied by hand-tracking and can't reach the keyboard. When events is not
        provided (teleop used outside record), fall back to ENTER. Ctrl+C still aborts.
        """
        print("[QuestHTSDualArm] ALIGN: robot is at START and FROZEN.", flush=True)
        print("  → Reset the environment and bring both hands to match the robot's start pose.", flush=True)
        print(
            "  → Press the → (right arrow) AGAIN to re-calibrate the hands and start the next episode.",
            flush=True,
        )
        while not self._stop_event.is_set():
            if events is not None:
                if events.get("stop_recording"):
                    # ESC: leave the flags set so the record loop exits cleanly — don't resume.
                    print("  ESC — stopping (no resume).", flush=True)
                    break
                if events.get("exit_early"):
                    events["exit_early"] = False  # consume the → so it can't end the next episode
                    print("  → received — re-calibrating hands and resuming.", flush=True)
                    break
            elif enter_pressed():
                print("  ENTER — resuming.", flush=True)
                break
            time.sleep(0.05)

    def prepare_control_session(self, robot=None) -> None:
        """Called by LeRobot before the main teleoperation loop.

        Steps:
          1. Wait for Quest HTS stream on both sides.
          2. Move robot to saved start positions.
          3. Capture wrist baselines (interactive ENTER prompt).
        """
        if robot is not None:
            obs = robot.get_observation()
            self.send_feedback({k: v for k, v in obs.items() if isinstance(v, int | float)})

        print("[QuestHTSDualArm] Waiting for Quest HTS stream (left + right)...", flush=True)
        print(
            f"  Start Meta Quest Hand Tracking Streamer → send to {self.config.host}:{self.config.port}",
            flush=True,
        )
        if not self._wait_for_both_sides(timeout_s=120.0):
            raise RuntimeError(
                "Timed out waiting for Quest HTS stream. "
                f"Ensure the Quest streamer is sending to port {self.config.port}."
            )
        print("[QuestHTSDualArm] Quest HTS stream connected (left + right).", flush=True)

        # One-time banner so the active IK mode + lateral signs are visible in the terminal.
        # full_ik tracks wrist ORIENTATION (quaternion), which is mirror-handed between the Quest
        # left/right hands → one side can feel reversed / "won't go up". Planar mode is position-only
        # (no handedness) and is the recommended default for symmetric dual_arm control.
        if self.config.use_full_ik:
            mode_str = "FULL_IK (5-joint, wrist ORIENTATION tracked — handedness-sensitive!)"
        elif self.config.use_elbow_ik:
            mode_str = "ELBOW_IK"
        else:
            mode_str = "PLANAR (position-only: shoulder_lift+elbow_flex(+pan), wrist held — recommended)"
        print(
            f"[QuestHTSDualArm] IK mode = {mode_str}\n"
            f"  planar_scale_z  right={self.config.planar_scale_z_right:+g}  left={self.config.planar_scale_z_left:+g}  (lateral→shoulder_pan sign)\n"
            f"  lift_feedforward_deg={self.config.lift_feedforward_deg:+g}  (gravity sag compensation)",
            flush=True,
        )

        # Move robot to start positions
        if self.config.move_to_start and robot is not None:
            print("[QuestHTSDualArm] Moving robot to start positions...", flush=True)
            self._move_to_start(robot)
            print("[QuestHTSDualArm] Start positions reached.", flush=True)
            obs = robot.get_observation()
            self.send_feedback({k: v for k, v in obs.items() if isinstance(v, int | float)})

        # Capture wrist baselines
        sides_hint = "BOTH hands" if self.config.require_both_sides else "your hand(s)"
        self._await_baseline_capture(sides_hint)
        with self._lock:
            for state in (self._right, self._left):
                if state.latest_wrist is not None:
                    state.wrist_baseline_pose = self._baseline_pose_from_history(state)
                    state.baseline_captured = True
                    if self.config.use_elbow_ik:
                        state.elbow_baseline_pos_quest = _estimate_elbow_quest(
                            state.latest_wrist, self.config.forearm_axis, self.config.forearm_length_m
                        )
                        if state.start_joint_degrees:
                            kin = state.kinematics
                            theta1 = (
                                math.radians(state.start_joint_degrees.get("shoulder_lift", 0.0))
                                - kin.theta1_offset
                            )
                            state.start_elbow_robot = np.array(
                                [kin.l1 * math.cos(theta1), kin.l1 * math.sin(theta1)]
                            )
                    # Snapshot hand start action from feedback
                    prefix = "r_" if state.side == "right" else "l_"
                    state.hand_start_action = {
                        f"{prefix}{k}": float(v)
                        for k, v in state.last_feedback.items()
                        if k.startswith("finger") and k.endswith(".pos")
                    }

        active = self._active_side_names()
        missing = [
            s
            for s, st in (("right", self._right), ("left", self._left))
            if s in active and not st.baseline_captured
        ]
        if missing:
            # In single-side mode that one side is mandatory; otherwise honour require_both_sides.
            require_all = self.config.require_both_sides or len(active) == 1
            if require_all:
                raise RuntimeError(
                    f"Could not capture wrist baseline for: {missing}. "
                    "Ensure Quest is tracking the required hand(s) and the HTS stream is running."
                )
            # require_both_sides=False: only fail if NO active side was captured at all
            captured = [
                s
                for s, st in (("right", self._right), ("left", self._left))
                if s in active and st.baseline_captured
            ]
            if not captured:
                raise RuntimeError(
                    "Could not capture wrist baseline for any hand. "
                    "Ensure Quest is tracking at least one hand and HTS stream is running."
                )
            print(
                f"[QuestHTSDualArm] Warning: baseline missing for {missing} — continuing with {captured} only.",
                flush=True,
            )
        print("[QuestHTSDualArm] Wrist baselines captured. Teleoperation started.", flush=True)

    def _move_to_start(self, robot) -> None:
        obs = robot.get_observation()
        current = {k: float(v) for k, v in obs.items() if k.endswith(".pos") and isinstance(v, int | float)}

        goal: dict[str, float] = {}
        for state, prefix in self._active_states():
            for k, v in state.start_action.items():
                full_key = f"{prefix}{k}"
                goal[full_key] = float(v)
            # Open the AmazingHand (パー) together with the arm move, unless arm-only.
            # hand_open_action keys are already prefixed (r_/l_).
            if self.config.mode != "arm-only":
                for k, v in state.hand_open_action.items():
                    goal[k] = float(v)

        if not goal:
            print("[QuestHTSDualArm] No start positions saved; skipping move-to-start.", flush=True)
            return

        steps = self.config.move_to_start_steps
        for i in range(1, steps + 1):
            t = i / steps
            waypoint = {k: current.get(k, v) + t * (v - current.get(k, v)) for k, v in goal.items()}
            robot.send_action(waypoint)
            time.sleep(self.config.move_to_start_delay_s)

    def _wait_for_both_sides(self, *, timeout_s: float = 120.0) -> bool:
        """Wait until Quest sends wrist data for the required sides.

        If require_both_sides=False, proceeds as soon as at least one side is ready.
        Prints per-side frame counts every 5s for diagnosis.
        """
        active = self._active_side_names()
        r_needed = "right" in active
        l_needed = "left" in active
        deadline = time.monotonic() + timeout_s
        last_print_s = time.monotonic() - 10.0
        while time.monotonic() < deadline:
            with self._lock:
                r_ok = self._right.latest_wrist is not None
                l_ok = self._left.latest_wrist is not None
                r_frames = len(self._right.wrist_history)
                l_frames = len(self._left.wrist_history)

            # A side that is not active never gates progress (treat as satisfied).
            r_satisfied = (not r_needed) or r_ok
            l_satisfied = (not l_needed) or l_ok

            # Proceed if required sides are ready
            if self.config.require_both_sides:
                if r_satisfied and l_satisfied:
                    return True
            else:
                if (r_needed and r_ok) or (l_needed and l_ok):
                    if r_needed and not r_ok:
                        print(
                            "  [INFO] right=waiting but require_both_sides=False → proceeding with left only.",
                            flush=True,
                        )
                    elif not l_ok:
                        print(
                            "  [INFO] left=waiting but require_both_sides=False → proceeding with right only.",
                            flush=True,
                        )
                    return True

            now = time.monotonic()
            if now - last_print_s >= 5.0:
                status = (
                    f"right={'OK' if r_ok else f'waiting(frames={r_frames})'} "
                    f"left={'OK' if l_ok else f'waiting(frames={l_frames})'}"
                )
                print(f"  [{status}]", flush=True)
                if not r_ok and r_frames == 0 and l_frames > 10:
                    print(
                        "  HINT: Quest is not sending right hand data.\n"
                        "  → Make sure your RIGHT hand is clearly visible to the Quest cameras.\n"
                        "  → Or set --teleop.require_both_sides false for left-only control.",
                        flush=True,
                    )
                last_print_s = now
            time.sleep(0.1)
        return False

    # ── Action computation ───────────────────────────────────────────────────

    def _baseline_pose_from_history(self, state: _SideState):
        """Baseline wrist pose with the POSITION averaged over the last `baseline_samples` (already
        One-Euro-filtered) samples to reject residual jitter; orientation from the latest sample.

        A single-sample baseline anchors the whole control frame to one noisy Quest reading, so the
        arm drifts off target. Averaging the recent (filtered) positions gives a steady origin.
        """
        base = _wrist_to_transform(state.latest_wrist)
        n = max(1, int(self.config.baseline_samples))
        hist = state.wrist_history[-n:] if state.wrist_history else []
        if len(hist) >= 2:
            pos = np.mean(np.asarray([h[:3] for h in hist], dtype=float), axis=0)
            base = base.copy()
            base[:3, 3] = pos
        return base

    def _apply_pitch_hold(
        self, state: _SideState, ik_joints: dict, shoulder_lift_deg: float, elbow_flex_deg: float
    ) -> None:
        """Hold the gripper PITCH at its start value by driving wrist_flex.

        The planar / elbow IK controls the wrist flange, but the operator controls the AmazingHand
        FINGERTIP (~10 cm past the flange). Keeping pitch (= wrist_flex + shoulder + elbow) constant
        makes the hand offset a constant vector, so the fingertip tracks the hand 1:1 (verified
        ~20 mm error → 0 mm). No-op unless hold_gripper_pitch is set and the start pose is known.
        """
        if not self.config.hold_gripper_pitch or not state.start_joint_degrees:
            return
        start_pitch = (
            state.start_joint_degrees.get("wrist_flex", 0.0)
            + state.start_joint_degrees.get("shoulder_lift", 0.0)
            + state.start_joint_degrees.get("elbow_flex", 0.0)
        )
        ik_joints["wrist_flex"] = start_pitch - shoulder_lift_deg - elbow_flex_deg

    def _build_arm_action_locked(self, state: _SideState) -> dict[str, float]:
        """IK for arm joints.

        use_full_ik=False (default): planar IK controlling shoulder_lift + elbow_flex.
          shoulder_pan optionally tracks Quest Z-axis. wrist_flex / wrist_roll held at start.
        use_full_ik=True: full 3D IK via solve_tcp_ik() controlling all 5 joints.
          Quest position delta → robot TCP position (same mapping as planar mode).
          Quest orientation delta → robot TCP orientation (wrist_roll + wrist_flex).
          Frame alignment: Quest X→Robot X, Quest Y→Robot Z, Quest Z→Robot -Y.
          Tune wrist_orientation_scale_right/left (sign / magnitude) to match your setup.
        """
        fallback = (
            dict(state.last_valid_ik_action) if state.last_valid_ik_action else dict(state.start_action)
        )

        if not state.baseline_captured or state.wrist_baseline_pose is None:
            return dict(state.start_action) if state.start_action else {}

        if state.latest_wrist is None:
            return fallback

        # Stale check
        if (
            state.latest_wrist_s is not None
            and time.monotonic() - state.latest_wrist_s > self.config.stale_timeout_s
        ):
            age = time.monotonic() - state.latest_wrist_s
            if not getattr(state, "_stale_warned", False):
                print(
                    f"[QuestHTSDualArm] {state.side} wrist data stale ({age:.1f}s) — "
                    "arm holding last position. Check Quest hand tracking.",
                    flush=True,
                )
                state._stale_warned = True
            return fallback
        state._stale_warned = False

        # Human wrist pose relative to baseline.
        # human_delta (baseline-local SE3) is still used for full-IK orientation tracking.
        human_pose = _wrist_to_transform(state.latest_wrist)
        human_delta = invert_transform(state.wrist_baseline_pose) @ human_pose

        # Position delta is taken in the Quest WORLD frame (not wrist-local), then rotated
        # into the robot frame. This makes hand translation map to robot motion regardless
        # of how the hand is tilted — the previous wrist-local delta meant "forward" drifted
        # with wrist rotation, so pulling the hand back / moving sideways didn't track.
        #   robot x = forward (radial), robot y = lateral, robot z = up
        dw_quest = np.asarray(state.latest_wrist[:3], dtype=float) - state.wrist_baseline_pose[:3, 3]
        dw_robot = _R_QUEST_TO_ROBOT @ dw_quest
        d_forward = float(dw_robot[0])  # depth toward/away from robot → radial
        d_lateral = float(dw_robot[1])  # left/right → shoulder_pan
        d_up = float(dw_robot[2])  # up/down → vertical

        if state.side == "right":
            scale_x = self.config.planar_scale_x_right
            scale_y = self.config.planar_scale_y_right
            scale_z = self.config.planar_scale_z_right
        else:
            scale_x = self.config.planar_scale_x_left
            scale_y = self.config.planar_scale_y_left
            scale_z = self.config.planar_scale_z_left

        if state.start_ee is None or state.arm_calibration is None or state.kinematics is None:
            return dict(state.start_action)

        # Quest position delta (world frame) → robot FINGERTIP target. start_ee is the fingertip
        # (TCP) anchor, so forward/lateral/up are the desired fingertip radial/lateral/vertical.
        forward = state.start_ee["x"] + scale_x * d_forward  # fingertip radial (arm forward/back)
        lateral = scale_z * d_lateral  # fingertip lateral (→ shoulder_pan)

        l1 = state.kinematics.l1
        l2 = state.kinematics.l2

        # Decompose the (pitch-held) AmazingHand offset into radial + vertical components so the
        # planar IK can pull the FLANGE back to where it must be for the FINGERTIP to hit the target.
        # The gripper pitch is held constant at the start value, so this projection is constant.
        start_pitch_rad = 0.0
        if state.start_joint_degrees:
            start_pitch_rad = math.radians(
                state.start_joint_degrees.get("wrist_flex", 0.0)
                + state.start_joint_degrees.get("shoulder_lift", 0.0)
                + state.start_joint_degrees.get("elbow_flex", 0.0)
            )
        off = float(self.config.tcp_offset_m)
        off_radial = off * math.cos(start_pitch_rad)
        off_vert = off * math.sin(start_pitch_rad)

        if self.config.use_full_ik and state.start_tcp_pose is not None:
            # ── Full 3D IK: all 5 joints (shoulder_pan, lift, elbow, wrist_flex, wrist_roll) ──
            target_radial = math.hypot(forward, lateral)
            target_vertical = state.start_ee["y"] + scale_y * d_up
            target_radial, target_vertical = _project_to_reach(
                target_radial, target_vertical, l1, l2, self.config.max_reach_fraction
            )

            pan_deg = state.start_pan_deg
            if scale_z != 0.0:
                pan_delta_deg = math.degrees(math.atan2(lateral, max(forward, 1e-4)))
                pan_deg = state.start_pan_deg + pan_delta_deg
            pan_rad = math.radians(pan_deg)

            # Quest orientation delta → robot frame → scale → apply to start TCP orientation
            ori_scale = (
                self.config.wrist_orientation_scale_right
                if state.side == "right"
                else self.config.wrist_orientation_scale_left
            )
            rotation_delta_robot = _R_QUEST_TO_ROBOT @ human_delta[:3, :3] @ _R_QUEST_TO_ROBOT.T
            rotation_delta_scaled = _scale_rotation(rotation_delta_robot, ori_scale)
            target_rotation = state.start_tcp_pose[:3, :3] @ rotation_delta_scaled

            target_tcp = np.eye(4, dtype=float)
            target_tcp[:3, :3] = target_rotation
            target_tcp[0, 3] = target_radial * math.cos(pan_rad)
            target_tcp[1, 3] = target_radial * math.sin(pan_rad)
            target_tcp[2, 3] = target_vertical

            # target_tcp is the FINGERTIP pose (start_ee/start_tcp_pose are TCP-anchored), so pass the
            # offset and let solve_tcp_ik back-compute the flange — consistent with planar mode.
            result = state.kinematics.solve_tcp_ik(
                target_tcp, tcp_offset=TcpOffset(x=float(self.config.tcp_offset_m)), clamp_to_workspace=True
            )
            if not result.success and not result.projected:
                return fallback
            ik_joints: dict[str, float] = dict(result.joint_degrees)
            if self.config.fix_wrist_flex:
                ik_joints.pop("wrist_flex", None)

        elif (
            self.config.use_elbow_ik
            and state.elbow_baseline_pos_quest is not None
            and state.start_elbow_robot is not None
        ):
            # ── Elbow-retargeting IK ─────────────────────────────────────────
            # shoulder_lift ← elbow position (wrist quat back-projected to elbow)
            # elbow_flex    ← forearm direction (wrist_pos − elbow_pos angle)
            kin = state.kinematics

            # Estimate current elbow in Quest world frame
            elbow_current_quest = _estimate_elbow_quest(
                state.latest_wrist, self.config.forearm_axis, self.config.forearm_length_m
            )
            wrist_current_quest = np.array(state.latest_wrist[:3], dtype=float)

            # Deltas from baseline (Quest world frame)
            de_quest = elbow_current_quest - state.elbow_baseline_pos_quest
            dw_quest = wrist_current_quest - state.wrist_baseline_pose[:3, 3]

            # Transform deltas to robot frame (x=forward, y=lateral, z=up)
            de_robot = _R_QUEST_TO_ROBOT @ de_quest
            dw_robot = _R_QUEST_TO_ROBOT @ dw_quest

            # Target elbow in robot 2D sagittal plane (forward, up)
            ex = state.start_elbow_robot[0] + scale_x * de_robot[0]
            ey = state.start_elbow_robot[1] + scale_y * de_robot[2]  # robot Z = up

            # Clamp elbow to single-link reach (l1)
            elbow_r = math.hypot(ex, ey)
            if elbow_r > kin.l1 - 0.001 and elbow_r > 1e-9:
                s = (kin.l1 - 0.001) / elbow_r
                ex, ey = ex * s, ey * s

            # shoulder_lift from elbow position (single-link IK)
            theta1 = math.atan2(ey, ex)
            shoulder_lift_deg = math.degrees(theta1 + kin.theta1_offset)

            # Target wrist in robot sagittal plane
            wx = state.start_ee["x"] + scale_x * dw_robot[0]
            wy = state.start_ee["y"] + scale_y * dw_robot[2]

            # elbow_flex from forearm direction:  θ2 = forearm_angle − θ1 + π
            forearm_angle = math.atan2(wy - ey, wx - ex)
            theta2 = forearm_angle - theta1 + math.pi
            theta2 = max(-0.2, min(math.pi, theta2))
            elbow_flex_deg = math.degrees(theta2 + kin.theta2_offset)

            ik_joints = {
                "shoulder_lift": shoulder_lift_deg,
                "elbow_flex": elbow_flex_deg,
            }
            if scale_z != 0.0:
                # Use elbow lateral delta for shoulder_pan
                pan_delta_deg = math.degrees(math.atan2(scale_z * de_robot[1], max(ex, 1e-4)))
                ik_joints["shoulder_pan"] = state.start_pan_deg + pan_delta_deg

            self._apply_pitch_hold(state, ik_joints, shoulder_lift_deg, elbow_flex_deg)

        else:
            # ── Planar IK: shoulder_lift + elbow_flex (+ optional shoulder_pan) ──
            # forward/lateral/up describe the FINGERTIP target; pull the flange back along the
            # gripper axis so the IK places the flange where the fingertip lands on target.
            fingertip_radial = math.hypot(forward, lateral)
            fingertip_z = state.start_ee["y"] + scale_y * d_up
            target_x = fingertip_radial - off_radial  # flange radial
            target_y = fingertip_z + off_vert  # flange vertical
            target_x, target_y = _project_to_reach(target_x, target_y, l1, l2, self.config.max_reach_fraction)

            try:
                shoulder_lift_deg, elbow_flex_deg = state.kinematics.inverse_kinematics(
                    target_x, target_y, l1=l1, l2=l2, clamp_to_workspace=True
                )
            except (ValueError, Exception):
                return fallback

            ik_joints = {
                "shoulder_lift": shoulder_lift_deg,
                "elbow_flex": elbow_flex_deg,
            }
            if scale_z != 0.0:
                pan_delta_deg = math.degrees(math.atan2(lateral, max(forward, 1e-4)))
                ik_joints["shoulder_pan"] = state.start_pan_deg + pan_delta_deg

            # Hold the gripper pitch constant so the AmazingHand FINGERTIP (not the flange) tracks
            # the hand 1:1 — see hold_gripper_pitch in the config. wrist_flex absorbs the lift+elbow
            # change so pitch (= wrist_flex + shoulder + elbow) stays at its start value.
            self._apply_pitch_hold(state, ik_joints, shoulder_lift_deg, elbow_flex_deg)

        # Gravity feed-forward: bias the commanded lift up so the gravity-sagged actual pose
        # matches intent without raising P.
        # +deg = arm higher. Applies in every IK mode that controls shoulder_lift.
        ff = self.config.lift_feedforward_deg
        if ff != 0.0 and "shoulder_lift" in ik_joints:
            ik_joints["shoulder_lift"] = ik_joints["shoulder_lift"] + ff

        controlled_action = state.arm_calibration.degrees_dict_to_action(ik_joints)

        # Merge with start action (joints not in ik_joints hold their start values)
        arm_action = dict(state.start_action)
        arm_action.update(controlled_action)

        # Reference = previous tick's commanded value (or start on the first tick).
        reference = state.last_controlled_action or {
            k: float(state.start_action.get(k, arm_action[k])) for k in controlled_action
        }

        # EMA smoothing: low-pass the noisy IK target toward the previous command.
        #   cmd = alpha * target + (1 - alpha) * prev_cmd
        # This is the primary anti-judder filter; the step clamp below is just a safety cap.
        alpha = self.config.arm_smoothing_alpha
        if alpha < 1.0:
            for key in list(controlled_action.keys()):
                if key in reference:
                    arm_action[key] = alpha * arm_action[key] + (1.0 - alpha) * reference[key]

        # Step clamping: hard safety cap on change per tick (bounds a tracking glitch).
        max_step = self.config.max_controlled_joint_step
        for key in list(controlled_action.keys()):
            if key in reference:
                delta = arm_action[key] - reference[key]
                if abs(delta) > max_step:
                    arm_action[key] = reference[key] + max_step * (1.0 if delta > 0 else -1.0)

        # Clamp to calibration limits
        with contextlib.suppress(Exception):
            arm_action, _ = state.arm_calibration.clamp_action(arm_action)

        state.last_valid_ik_action = dict(arm_action)
        state.last_controlled_action = {k: arm_action[k] for k in controlled_action}
        return arm_action

    def _build_hand_action_locked(self, state: _SideState) -> dict[str, float]:
        """Piecewise-linear hand mapping from Quest landmarks to AmazingHand motor targets."""
        hand_keys = HAND_KEYS_R if state.side == "right" else HAND_KEYS_L

        # Fallback: hold feedback or start
        def _fallback() -> dict[str, float]:
            result = {}
            for key in hand_keys:
                bare = _strip_prefix(key)
                result[key] = float(state.last_feedback.get(bare, 50.0))
            return result

        if state.hand_mapper is None or state.latest_landmarks is None:
            return _fallback()

        # Check stale
        if (
            state.latest_landmarks_s is not None
            and time.monotonic() - state.latest_landmarks_s > self.config.stale_timeout_s
        ):
            return _fallback()

        try:
            from lerobot.teleoperators.quest_hts.dual_arm_calibration import (
                hand_feature_vector_from_landmarks,
            )

            # The mapper carries the feature mode its anchors were captured with; compute the
            # live feature the same way so curl-calibrated hands use the curl feature and legacy
            # distance-calibrated hands keep using distance. (No mixing → grip stays consistent.)
            feature_mode = getattr(state.hand_mapper, "feature_mode", "distance")
            features = hand_feature_vector_from_landmarks(state.latest_landmarks, mode=feature_mode)
            targets = state.hand_mapper.map_features(features)
        except Exception as exc:
            logger.debug("Hand mapping error for %s: %s", state.side, exc)
            return _fallback()

        # Left-hand finger remap: the left AmazingHand is a mirror build, so its physical finger
        # order is reversed vs the motor-ID numbering. Re-route the computed values so each
        # physical finger pair receives the value of the human finger that should drive it.
        if state.side == "left":
            remap = tuple(self.config.left_hand_finger_remap)
            if remap != (1, 2, 3, 4):
                remapped = {}
                for dest in range(1, 5):
                    src = remap[dest - 1]
                    for j in (1, 2):
                        dst_key = f"l_finger{dest}_motor{j}.pos"
                        src_key = f"l_finger{src}_motor{j}.pos"
                        if src_key in targets:
                            remapped[dst_key] = targets[src_key]
                if remapped:
                    targets = remapped

        # Thumb (finger1 = motor IDs 1-2) closing-gain. The thumb's tip-to-MCP feature has a small
        # dynamic range, so the mapped thumb under-bends vs the other fingers. Amplify finger1's
        # mapped closedness (stays WITHIN the calibrated [open, fist] range — the separate
        # over-close below handles squeezing past fist). No recalibration needed.
        gain = self.config.thumb_close_gain
        if gain != 1.0 and state.hand_open_action and state.hand_fist_action:
            for key in list(targets.keys()):
                if "finger1_" not in key:
                    continue
                open_val = state.hand_open_action.get(key)
                fist_val = state.hand_fist_action.get(key)
                if open_val is None or fist_val is None:
                    continue
                span = fist_val - open_val
                if span == 0.0:
                    continue
                closedness = max(0.0, (targets[key] - open_val) / span)  # 0=open … 1=fist
                closedness = min(1.0, closedness * gain)
                targets[key] = open_val + closedness * span

        # Grip "squeeze" over-close: push the closing fingers PAST the calibrated fist pose so a
        # blocking object meets a firmer grip. The SCS0009 torque registers are already maxed
        # (P=32, Max_Torque=1000), so a larger commanded close — i.e. a larger position error
        # against the object — is the only remaining way to squeeze harder. The boost scales with
        # how much of a fist the human is making, so an open hand is untouched (acts "especially
        # when making a fist", as requested). Result is clamped to the calibrated [0,100] range.
        boost = self.config.grip_close_boost
        if boost > 0.0 and state.hand_open_action and state.hand_fist_action:
            for key in list(targets.keys()):
                open_val = state.hand_open_action.get(key)
                fist_val = state.hand_fist_action.get(key)
                if open_val is None or fist_val is None:
                    continue
                span = fist_val - open_val  # signed: encodes this motor's closing direction
                if span == 0.0:
                    continue
                closedness = (targets[key] - open_val) / span  # 0=open … 1=fist
                closedness = max(0.0, min(1.0, closedness))
                targets[key] = max(0.0, min(100.0, targets[key] + boost * closedness * span))

        # Delta-from-start clamping
        max_delta = self.config.max_hand_joint_delta_from_start
        if state.hand_start_action:
            for key in list(targets.keys()):
                start_val = state.hand_start_action.get(key)
                if start_val is not None:
                    delta = targets[key] - start_val
                    if abs(delta) > max_delta:
                        targets[key] = start_val + max_delta * (1.0 if delta > 0 else -1.0)

        # EMA smoothing: the landmark→motor map has no other low-pass, so this is the primary
        # anti-jitter filter for the fingers.  cmd = a*target + (1-a)*prev_cmd.
        alpha = self.config.hand_smoothing_alpha
        if alpha < 1.0 and state.last_hand_action:
            for key in list(targets.keys()):
                prev = state.last_hand_action.get(key)
                if prev is not None:
                    targets[key] = alpha * targets[key] + (1.0 - alpha) * prev

        out = {k: float(v) for k, v in targets.items()}
        state.last_hand_action = dict(out)
        return out

    # ── HTS line handling ────────────────────────────────────────────────────

    def handle_hts_line(self, line: str) -> bool:
        try:
            from lerobot.teleoperators.quest_hts.hts_protocol import parse_hts_line
        except ImportError:
            return False
        parsed = parse_hts_line(line)
        if parsed is None:
            return False
        side, kind, values = parsed
        if side not in ("right", "left"):
            return False
        # Optionally swap which logical side this Quest hand drives. Swapping HERE moves the whole
        # side together (arm + hand + calib + startup), keeping arm and hand on the same physical
        # side — unlike swapping only the arm ports, which splits them.
        if self.config.swap_sides:
            side = "left" if side == "right" else "right"
        now = time.monotonic()
        with self._lock:
            state = self._right if side == "right" else self._left
            if kind == "wrist":
                wrist = tuple(float(v) for v in values)  # (x,y,z, qx,qy,qz,qw)
                # Anti-jitter: One-Euro low-pass the POSITION at ingest so both the live IK target
                # and the captured baseline use the cleaned signal. Orientation passes through.
                if self.config.wrist_filter_enabled and state.wrist_filter is not None and len(wrist) >= 3:
                    pos = state.wrist_filter(wrist[:3], now)
                    wrist = (float(pos[0]), float(pos[1]), float(pos[2])) + wrist[3:]
                state.latest_wrist = wrist
                state.latest_wrist_s = now
                state.wrist_history.append(state.latest_wrist)
                if len(state.wrist_history) > 200:
                    del state.wrist_history[:-200]
                # Auto-baseline: capture (or re-capture) baseline when tracking (re)starts.
                # - First arrival after ENTER: activates arm without requiring restart.
                # - Recovery after stale: resets baseline to current wrist so arm doesn't jump.
                was_stale = getattr(state, "_stale_warned", False)
                needs_baseline = not state.baseline_captured and state.start_action
                if needs_baseline or was_stale:
                    state.wrist_baseline_pose = self._baseline_pose_from_history(state)
                    state.baseline_captured = True
                    state._stale_warned = False
                    prefix = "r_" if state.side == "right" else "l_"
                    state.hand_start_action = {
                        f"{prefix}{k}": float(v)
                        for k, v in state.last_feedback.items()
                        if k.startswith("finger") and k.endswith(".pos")
                    }
                    if needs_baseline:
                        print(
                            f"[QuestHTSDualArm] Late baseline captured for {side} — arm now active.",
                            flush=True,
                        )
                    else:
                        print(
                            f"[QuestHTSDualArm] {side} tracking recovered — baseline reset, resuming.",
                            flush=True,
                        )
            elif kind == "landmarks":
                state.latest_landmarks = tuple(float(v) for v in values)
                state.latest_landmarks_s = now
        return True

    # ── TCP server ───────────────────────────────────────────────────────────

    def _tcp_server_loop(self, server: socket.socket) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    conn, addr = server.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                print(f"[QuestHTSDualArm] Quest HTS client connected from {addr}", flush=True)
                threading.Thread(
                    target=self._handle_connection,
                    args=(conn,),
                    daemon=True,
                    name=f"quest_hts_conn_{addr[0]}_{addr[1]}",
                ).start()
        finally:
            with contextlib.suppress(OSError):
                server.close()
            if self._server_sock is server:
                self._server_sock = None

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(0.5)
            buf = ""
            while not self._stop_event.is_set():
                try:
                    data = conn.recv(8192)
                except TimeoutError:
                    continue
                except OSError as exc:
                    logger.warning("HTS connection error: %s", exc)
                    break
                if not data:
                    print("[QuestHTSDualArm] Quest HTS client disconnected.", flush=True)
                    break
                try:
                    buf += data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        self.handle_hts_line(line)
