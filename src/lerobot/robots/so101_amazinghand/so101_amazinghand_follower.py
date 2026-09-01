#!/usr/bin/env python

"""SO-101 + AmazingHand 13-DOF follower robot.

Action / observation space
--------------------------
Arm  (5 DOF)  : shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
Hand (8 DOF)  : finger1_motor1..2, finger2_motor1..2, finger3_motor1..2, finger4_motor1..2

All arm values are in [-100, 100] (RANGE_M100_100 normalised).
All hand values are in [0, 100]  (RANGE_0_100 normalised, 0=closed, 100=open).

Endian note
-----------
STS3215 (arm)  uses Little Endian  → scservo_def.SCS_END = 0  (SDK default)
SCS0009 (hand) uses Big Endian     → scservo_def.SCS_END = 1

By default both motor types share one physical RS485 port (one USB-RS485 adapter).
A single PortHandler is opened once and shared between arm_bus and hand_bus.

If config.hand_port is set to a different device path than config.port, arm and hand
will use separate PortHandlers (two USB adapters), which avoids power-sharing issues.
Operations are always serialised (never concurrent) so there is no race on the bus.
"""

import logging
import os
import time
from contextlib import contextmanager
from functools import cached_property

import scservo_sdk as _scs
import scservo_sdk.scservo_def as _scs_def

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors.feetech.feetech import patch_setPacketTimeout
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so101_amazinghand import SO101AmazingHandFollowerConfig

logger = logging.getLogger(__name__)

# ── AmazingHand auto-calibration constants ────────────────────────────────────
# Sweep each SCS0009 motor to raw 0 then raw 1023; record where it stops.
# SCS0009 default EEPROM limits are 0–1023, so no EEPROM pre-write is needed.
_HAND_CAL_LO = 0  # raw Goal_Position for finding physical minimum
_HAND_CAL_HI = 1023  # raw Goal_Position for finding physical maximum
_HAND_CAL_SETTLE = 2.5  # seconds to wait for motor to reach physical stop
_HAND_CAL_MARGIN = 10  # raw ticks safety margin inward from measured stop
_HAND_CAL_MIN_SPAN = 50  # reject sweep if range span < this (motor didn't move)

ARM_MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
HAND_MOTORS = [
    "finger1_motor1",
    "finger1_motor2",
    "finger2_motor1",
    "finger2_motor2",
    "finger3_motor1",
    "finger3_motor2",
    "finger4_motor1",
    "finger4_motor2",
]
ALL_MOTORS = ARM_MOTORS + HAND_MOTORS  # 13 total


@contextmanager
def _big_endian():
    """Set SCS_END=1 (Big Endian) for SCS0009 hand bus operations, then restore."""
    _scs_def.SCS_END = 1
    try:
        yield
    finally:
        _scs_def.SCS_END = 0


def _make_shared_port_handler(port: str):
    """Create (but do NOT open) a PortHandler for the given port.
    Opening is deferred to connect() so that is_connected stays False until then.
    """
    ph = _scs.PortHandler(port)
    ph.setPacketTimeout = patch_setPacketTimeout.__get__(ph, _scs.PortHandler)
    return ph


class SO101AmazingHandFollower(Robot):
    """13-DOF follower: SO-101 arm (IDs 1-5) + AmazingHand (IDs 1-8).

    Both buses share ONE physical serial port (shared PortHandler).
    """

    config_class = SO101AmazingHandFollowerConfig
    name = "so101_amazinghand_follower"

    def __init__(
        self,
        config: SO101AmazingHandFollowerConfig,
        *,
        arm_port_handler=None,
        owns_arm_port: bool = True,
    ):
        # arm_port_handler: when set, the arm bus uses THIS externally-owned PortHandler instead of
        #   creating its own — lets two follower instances put two arms (different IDs) on ONE
        #   physical RS485 bus without opening the same /dev twice (that double-open = packet
        #   collisions). owns_arm_port=False then means this instance must NOT close that shared
        #   port on disconnect (the owner — the dual_arm follower — closes it once, after BOTH
        #   arms have had their torque disabled).
        super().__init__(config)
        self.config = config
        self._owns_arm_port = owns_arm_port
        norm_arm = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self._last_hand_obs: dict[str, float] = {}  # last-known hand positions for read-error fallback
        self._hand_overload_count: dict[str, int] = {}  # consecutive overload error counts per motor
        self._last_arm_obs: dict[str, float] = {}  # last-known arm positions for read-error fallback
        self._arm_overload_count: dict[str, int] = {}  # consecutive overload error counts per arm motor

        arm_cal = {k: v for k, v in self.calibration.items() if k in ARM_MOTORS} or None
        hand_cal = {k: v for k, v in self.calibration.items() if k in HAND_MOTORS} or None

        # Arm port handler — externally injected (shared across two arms on one bus) or own.
        self._port_handler = (
            arm_port_handler if arm_port_handler is not None else _make_shared_port_handler(config.port)
        )

        # Hand port handler — separate device when hand_port is set, shared otherwise.
        # Compare RESOLVED device paths, not the raw strings: passing the arm as a udev
        # symlink (e.g. /dev/ttyso101_..._arm) and the hand as a different string that resolves
        # to the SAME physical device must NOT open two handlers on one serial port — that
        # causes packet collisions and juddery motion.
        def _resolve(p: str) -> str:
            try:
                return os.path.realpath(p)
            except OSError:
                return p

        _hand_port = config.hand_port or config.port
        self._separate_hand_port: bool = bool(
            config.hand_port and _resolve(config.hand_port) != _resolve(config.port)
        )
        self._hand_port_handler = (
            _make_shared_port_handler(_hand_port) if self._separate_hand_port else self._port_handler
        )

        arm_ids = tuple(config.arm_motor_ids)
        if len(arm_ids) != len(ARM_MOTORS):
            raise ValueError(
                f"arm_motor_ids must have {len(ARM_MOTORS)} ids (one per {ARM_MOTORS}), got {arm_ids}"
            )
        self.arm_bus = FeetechMotorsBus(
            port=config.port,
            motors={name: Motor(arm_ids[i], "sts3215", norm_arm) for i, name in enumerate(ARM_MOTORS)},
            calibration=arm_cal,
            protocol_version=0,
            shared_port_handler=self._port_handler,
        )

        self.hand_bus = FeetechMotorsBus(
            port=_hand_port,
            motors={
                # Physical ID assignment: finger1 (index side) = ID1-2,
                # finger4 (thumb side) = ID7-8.
                "finger1_motor1": Motor(1, "scs0009", MotorNormMode.RANGE_0_100),
                "finger1_motor2": Motor(2, "scs0009", MotorNormMode.RANGE_0_100),
                "finger2_motor1": Motor(3, "scs0009", MotorNormMode.RANGE_0_100),
                "finger2_motor2": Motor(4, "scs0009", MotorNormMode.RANGE_0_100),
                "finger3_motor1": Motor(5, "scs0009", MotorNormMode.RANGE_0_100),
                "finger3_motor2": Motor(6, "scs0009", MotorNormMode.RANGE_0_100),
                "finger4_motor1": Motor(7, "scs0009", MotorNormMode.RANGE_0_100),
                "finger4_motor2": Motor(8, "scs0009", MotorNormMode.RANGE_0_100),
            },
            calibration=hand_cal,
            protocol_version=1,
            shared_port_handler=self._hand_port_handler,
        )

        self.cameras = make_cameras_from_configs(config.cameras)

        # PacketHandler(1) may set SCS_END=1 globally. Reset to 0 for arm ops.
        _scs_def.SCS_END = 0

    # ── Features ──────────────────────────────────────────────────────────────

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        feats: dict = {f"{m}.pos": float for m in ALL_MOTORS}
        for cam_key, cam_cfg in self.config.cameras.items():
            feats[cam_key] = (cam_cfg.height, cam_cfg.width, 3)
        return feats

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in ALL_MOTORS}

    # ── Connection ────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return (
            self.arm_bus.is_connected
            and self.hand_bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @staticmethod
    def _hand_bus_motors_spec() -> dict:
        """Return the canonical hand motor name → Motor mapping (for tests and introspection)."""
        return {
            "finger1_motor1": Motor(1, "scs0009", MotorNormMode.RANGE_0_100),
            "finger1_motor2": Motor(2, "scs0009", MotorNormMode.RANGE_0_100),
            "finger2_motor1": Motor(3, "scs0009", MotorNormMode.RANGE_0_100),
            "finger2_motor2": Motor(4, "scs0009", MotorNormMode.RANGE_0_100),
            "finger3_motor1": Motor(5, "scs0009", MotorNormMode.RANGE_0_100),
            "finger3_motor2": Motor(6, "scs0009", MotorNormMode.RANGE_0_100),
            "finger4_motor1": Motor(7, "scs0009", MotorNormMode.RANGE_0_100),
            "finger4_motor2": Motor(8, "scs0009", MotorNormMode.RANGE_0_100),
        }

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        # Open arm port.
        if not self._port_handler.openPort():
            raise ConnectionError(f"\nCould not open port '{self.config.port}'. Check the port symlink.")

        # Open hand port (only needed when it's a separate physical device).
        if self._separate_hand_port:
            _hand_port = self.config.hand_port
            if not self._hand_port_handler.openPort():
                self._port_handler.closePort()
                raise ConnectionError(f"\nCould not open hand port '{_hand_port}'. Check the port path.")
            logger.info(f"Hand using separate port: {_hand_port}")

        # Arm: bypass bus.connect() (shared port already open); call _connect() directly.
        # handshake=False because SCS0009 motors (IDs 1-8) may share the same RS485 bus and
        # their protocol-1 responses interfere with protocol-0 ping reads for IDs 1-5.
        _scs_def.SCS_END = 0
        self.arm_bus._connect(handshake=False)
        self.arm_bus.set_timeout()

        # Hand: same bypass with Big Endian. Also skip handshake for same reason.
        with _big_endian():
            self.hand_bus._connect(handshake=False)
            self.hand_bus.set_timeout()

        # When sharing one port: clear residual bytes left by the Big Endian hand-bus
        # init before switching back to protocol-0 arm operations.
        if not self._separate_hand_port:
            self._port_handler.clearPort()
        _scs_def.SCS_END = 0

        if calibrate and not self.is_calibrated:
            logger.info("Calibration missing — running calibration")
            self.calibrate()
        elif self.calibration:
            # Calibration file was loaded at init; write it to the motor buses so
            # normalized reads (Present_Position) and normalized writes (Goal_Position)
            # work correctly without going through the interactive calibrate() prompt.
            self._apply_calibration_to_buses()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected (13 DOF: 5 arm + 8 hand)")

    # ── Calibration ───────────────────────────────────────────────────────────

    def _apply_calibration_to_buses(self) -> None:
        """Write the loaded calibration dict to the arm and hand bus objects.

        Required so that normalized reads (Present_Position) and writes
        (Goal_Position) work correctly.  Called automatically in connect() when
        a calibration file already exists — avoids the interactive calibrate()
        prompt just to register calibration in the bus.
        """
        arm_cal = {k: v for k, v in self.calibration.items() if k in ARM_MOTORS}
        hand_cal = {k: v for k, v in self.calibration.items() if k in HAND_MOTORS}
        if arm_cal:
            self.arm_bus.write_calibration(arm_cal)
            logger.info("Arm calibration written to bus (%d motors)", len(arm_cal))
        if hand_cal:
            with _big_endian():
                self.hand_bus.write_calibration(hand_cal)
            logger.info("Hand calibration written to bus (%d motors)", len(hand_cal))

    @property
    def is_calibrated(self) -> bool:
        """Check calibration using local dict (avoids hardware reads on shared RS485 bus)."""
        return all(m in self.calibration for m in ALL_MOTORS)

    def _record_arm_ranges(self, motors: list[str]) -> tuple[dict, dict]:
        """Record min/max ranges using individual reads (no sync_read).

        Replaces arm_bus.record_ranges_of_motion() during calibration because
        sync_read on STS3215 is unreliable immediately after EEPROM writes.
        """
        # Read initial positions individually.
        positions = {}
        for motor in motors:
            positions[motor] = self.arm_bus.read("Present_Position", motor, normalize=False)
        mins = positions.copy()
        maxes = positions.copy()

        user_pressed_enter = False
        while not user_pressed_enter:
            for motor in motors:
                try:  # noqa: SIM105 - retain the previous sample after a transient bus error
                    positions[motor] = self.arm_bus.read("Present_Position", motor, normalize=False)
                except Exception:
                    pass  # keep last known value on transient read error
            mins = {m: min(positions[m], mins[m]) for m in motors}
            maxes = {m: max(positions[m], maxes[m]) for m in motors}

            print("\n-------------------------------------------")
            print(f"{'NAME':<20} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
            for motor in motors:
                print(f"{motor:<20} | {mins[motor]:>6} | {positions[motor]:>6} | {maxes[motor]:>6}")

            if enter_pressed():
                user_pressed_enter = True

            if not user_pressed_enter:
                move_cursor_up(len(motors) + 3)

        same_min_max = [m for m in motors if mins[m] == maxes[m]]
        if same_min_max:
            raise ValueError(f"Some motors have the same min and max: {same_min_max}")

        return mins, maxes

    @property
    def _arm_label(self) -> str:
        """Human-readable side label for calibration prompts.

        Returns 'RIGHT' / 'LEFT' when the arm id ends with '_r' / '_l' (dual-arm mode),
        or an empty string in single-arm mode so prompts stay unchanged.
        """
        if self.id.endswith("_r"):
            return "RIGHT"
        if self.id.endswith("_l"):
            return "LEFT"
        return ""

    def _prompt(self, msg: str) -> str:
        """Prefix *msg* with the arm side label (dual-arm) or return *msg* unchanged (single)."""
        label = self._arm_label
        return f"[{label}] {msg}" if label else msg

    def _auto_calibrate_hand(self) -> dict:
        """Sweep all AmazingHand motors to physical limits and record actual range.

        Sends raw Goal_Position = 0 to every motor, waits for settling, then sweeps
        to 1023.  The positions where each motor stops become range_min / range_max
        (with a small inward safety margin).  Falls back to config defaults for any
        motor whose measured span is suspiciously small.
        """
        print(self._prompt("Auto-calibrating AmazingHand — ensure all fingers are free to move fully."))
        input(self._prompt("Press ENTER to begin the sweep …"))

        lo_pos: dict[str, int] = {}
        hi_pos: dict[str, int] = {}

        # Enable torque so motors actively move to the commanded targets.
        with _big_endian():
            self.hand_bus.sync_write("Torque_Enable", 1)

        # ── Sweep to raw 0 ────────────────────────────────────────────────
        print(self._prompt("  Sweeping all motors → raw 0 (one physical extreme) …"))
        with _big_endian():
            for motor in self.hand_bus.motors:
                self.hand_bus.write("Goal_Position", motor, _HAND_CAL_LO, normalize=False)
        time.sleep(_HAND_CAL_SETTLE)
        with _big_endian():
            for motor in self.hand_bus.motors:
                lo_pos[motor] = int(self.hand_bus.read("Present_Position", motor, normalize=False))

        # ── Sweep to raw 1023 ─────────────────────────────────────────────
        print(self._prompt("  Sweeping all motors → raw 1023 (other physical extreme) …"))
        with _big_endian():
            for motor in self.hand_bus.motors:
                self.hand_bus.write("Goal_Position", motor, _HAND_CAL_HI, normalize=False)
        time.sleep(_HAND_CAL_SETTLE)
        with _big_endian():
            for motor in self.hand_bus.motors:
                hi_pos[motor] = int(self.hand_bus.read("Present_Position", motor, normalize=False))

        # ── Build calibration ─────────────────────────────────────────────
        hand_calibration: dict[str, MotorCalibration] = {}
        print(self._prompt("  Measured ranges:"))
        for motor, m in self.hand_bus.motors.items():
            raw_lo = min(lo_pos[motor], hi_pos[motor])
            raw_hi = max(lo_pos[motor], hi_pos[motor])
            span = raw_hi - raw_lo
            if span < _HAND_CAL_MIN_SPAN:
                logger.warning(
                    f"{motor}: span too small (lo={raw_lo}, hi={raw_hi}); "
                    f"using config defaults ({self.config.hand_range_min}, {self.config.hand_range_max})"
                )
                raw_lo = self.config.hand_range_min
                raw_hi = self.config.hand_range_max
            else:
                raw_lo = raw_lo + _HAND_CAL_MARGIN
                raw_hi = raw_hi - _HAND_CAL_MARGIN
            hand_calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=0,
                range_min=raw_lo,
                range_max=raw_hi,
            )
            print(self._prompt(f"    {motor:<22}: range [{raw_lo:>4}, {raw_hi:>4}]"))

        with _big_endian():
            self.hand_bus.disable_torque()
            self.hand_bus.write_calibration(hand_calibration)

        return hand_calibration

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                self._prompt(
                    f"Press ENTER to use existing calibration (id={self.id}), "
                    "or type 'c' to run new calibration: "
                )
            )
            if user_input.strip().lower() != "c":
                arm_cal = {k: v for k, v in self.calibration.items() if k in ARM_MOTORS}
                hand_cal = {k: v for k, v in self.calibration.items() if k in HAND_MOTORS}
                if arm_cal:
                    self.arm_bus.write_calibration(arm_cal)
                if hand_cal:
                    with _big_endian():
                        self.hand_bus.disable_torque()
                        self.hand_bus.write_calibration(hand_cal)
                return

        # ── Arm calibration (interactive) ─────────────────────────────────
        logger.info(self._prompt("Running arm calibration …"))

        # disable_torque() writes Lock=0 per motor (see FeetechMotorsBus.disable_torque).
        self._port_handler.clearPort()
        _scs_def.SCS_END = 0
        self.arm_bus.disable_torque()

        self._port_handler.clearPort()
        _scs_def.SCS_END = 0

        input(self._prompt("Move SO-101 arm to the middle of its range and press ENTER …"))

        # ── Homing offsets (avoid sync_read immediately after EEPROM writes) ───
        # reset_calibration() writes Homing_Offset / Min_Position_Limit / Max_Position_Limit
        # (all EEPROM) to 5 motors = 15 writeTxRx calls.  Sending sync_read right after
        # leaves the STS3215 motors in mid-EEPROM-write — responses are garbled.
        # Fix: inline reset + 100 ms settle + individual Present_Position reads.
        self._port_handler.clearPort()
        _scs_def.SCS_END = 0
        for motor, m in self.arm_bus.motors.items():
            max_res = self.arm_bus.model_resolution_table[m.model] - 1
            self.arm_bus.write("Homing_Offset", motor, 0, normalize=False)
            self.arm_bus.write("Min_Position_Limit", motor, 0, normalize=False)
            self.arm_bus.write("Max_Position_Limit", motor, max_res, normalize=False)
        self.arm_bus.calibration = {}

        time.sleep(0.1)  # let EEPROM writes settle before reading
        self._port_handler.clearPort()
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
            self._prompt("Move all arm joints (except wrist_roll) through full range. Press ENTER to stop …")
        )
        # Use individual reads instead of sync_read: STS3215 sync_read is unreliable
        # immediately after EEPROM (Homing_Offset) writes → COMM_RX_CORRUPT.
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

        # ── Hand calibration ──────────────────────────────────────────────
        if self.config.auto_calibrate_hand:
            hand_calibration = self._auto_calibrate_hand()
        else:
            logger.info(self._prompt("Writing fixed position limits to AmazingHand …"))
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

        self.calibration = {**arm_calibration, **hand_calibration}
        self._save_calibration()
        logger.info(f"Calibration saved → {self.calibration_fpath}")

    # ── Configure ─────────────────────────────────────────────────────────────

    def configure(self) -> None:
        # Flush any residual bus bytes before arm (protocol-0) operations.
        self._port_handler.clearPort()
        _scs_def.SCS_END = 0

        # STS3215: Lock=1 (default after power-on) blocks EEPROM writes (addr 0-39).
        # Must set Lock=0 before writing Operating_Mode, P/I/D, etc.
        for motor in self.arm_bus.motors:
            self.arm_bus.write("Lock", motor, 0)

        lift_joints = {"shoulder_lift", "elbow_flex"}  # gravity-loaded → stiffer hold
        with self.arm_bus.torque_disabled():
            self.arm_bus.configure_motors()
            for motor in self.arm_bus.motors:
                p_coef = (
                    self.config.arm_p_coefficient_lift
                    if motor in lift_joints
                    else self.config.arm_p_coefficient
                )
                self.arm_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.arm_bus.write("P_Coefficient", motor, p_coef)
                self.arm_bus.write("I_Coefficient", motor, 0)
                self.arm_bus.write("D_Coefficient", motor, self.config.arm_d_coefficient)
                self.arm_bus.write("Acceleration", motor, 50)
                # Goal_Velocity=0: non-zero value triggers wheel/velocity mode on STS3215,
                # which ignores Goal_Position commands and produces no holding torque.
                self.arm_bus.write("Goal_Velocity", motor, 0, normalize=False)

        # ── Latch Goal_Position = Present_Position before enabling torque ────────
        # Without this, Goal_Position SRAM defaults to 0 → motor fights to reach
        # tick 0 (extreme end), appearing "no torque" in the opposite direction.
        self._port_handler.clearPort()
        _scs_def.SCS_END = 0
        present: dict | None = None
        try:
            present = self.arm_bus.sync_read("Present_Position", normalize=False)
        except Exception:
            # sync_read unreliable — fall back to individual reads with more retries.
            present = {}
            for motor in self.arm_bus.motors:
                try:
                    present[motor] = self.arm_bus.read(
                        "Present_Position", motor, num_retry=3, normalize=False
                    )
                except Exception:
                    present = None
                    break
        if present:
            for motor, pos in present.items():
                self.arm_bus.write("Goal_Position", motor, int(pos), normalize=False)
            logger.info(f"Arm Goal_Position latched to present: {present}")
        else:
            raise RuntimeError(
                f"{self}: Could not read arm Present_Position — refusing to enable torque "
                "to avoid motor fighting to position 0. "
                "Check the RS-485 cable/adapter on the arm port."
            )

        self._port_handler.clearPort()
        with _big_endian():
            # AmazingHand (SCS0009) grip tuning. These are EEPROM registers (addr < 40), so unlock
            # with Lock=0 before writing. The overload-protection pair lets
            # a stalled (gripping) finger hold harder (Protective_Torque) and longer
            # (Protection_Time) before the firmware derates it. Wrapped in try/except so a flaky
            # SCS0009 write never aborts configure() (which would break recording entirely).
            for motor in self.hand_bus.motors:
                try:
                    self.hand_bus.write("Lock", motor, 0)
                    self.hand_bus.write("P_Coefficient", motor, self.config.hand_p_coefficient)
                    self.hand_bus.write("Max_Torque_Limit", motor, self.config.hand_max_torque_limit)
                    self.hand_bus.write("Protective_Torque", motor, self.config.hand_protective_torque)
                    self.hand_bus.write("Protection_Time", motor, self.config.hand_protection_time)
                except Exception as exc:
                    logger.warning(f"Hand grip-tuning write failed for {motor}: {exc}")
            self.hand_bus.sync_write("Torque_Enable", 0)
            self.hand_bus.sync_write("Torque_Enable", 1)

        # Re-enable arm torque; write Lock=0 again in case hand-bus ops reset it.
        self._port_handler.clearPort()
        _scs_def.SCS_END = 0
        for motor in self.arm_bus.motors:
            self.arm_bus.write("Lock", motor, 0)
        self.arm_bus.enable_torque()
        logger.info("Arm torque enabled. configure() complete.")

    # ── Observation ───────────────────────────────────────────────────────────

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: dict = {}

        # ── Read hand motors first (Big Endian, protocol 1) ───────────────────
        # Hand reads first so any bus residue from SCS0009 is cleared before arm reads.
        with _big_endian():
            for motor in self.hand_bus.motors:
                key = f"{motor}.pos"
                try:
                    val = self.hand_bus.read("Present_Position", motor)
                    obs[key] = val
                    self._last_hand_obs[key] = val
                    self._hand_overload_count[motor] = 0
                except Exception as exc:
                    exc_str = str(exc)
                    logger.warning(f"Hand read [{motor}] failed: {exc}")
                    obs[key] = self._last_hand_obs.get(key, 50.0)
                    if "Overload error" in exc_str:
                        count = self._hand_overload_count.get(motor, 0) + 1
                        self._hand_overload_count[motor] = count
                        if count == 1:
                            # First overload: cycle torque to clear protection state
                            try:
                                self.hand_bus.write("Torque_Enable", motor, 0)
                                self.hand_bus.write("Torque_Enable", motor, 1)
                            except Exception:
                                pass

        # Clear any stale bytes left on the shared RS485 bus before switching protocol.
        self._port_handler.clearPort()

        # ── Read arm motors (Little Endian, protocol 0) ───────────────────────
        _scs_def.SCS_END = 0
        try:
            arm_pos = self.arm_bus.sync_read("Present_Position", num_retry=2)
            for k, v in arm_pos.items():
                self._last_arm_obs[k] = v
                self._arm_overload_count[k] = 0
        except Exception:
            # Bus still unstable — fall back to individual reads per motor.
            arm_pos = {}
            for motor in self.arm_bus.motors:
                try:
                    val = self.arm_bus.read("Present_Position", motor, num_retry=1)
                    arm_pos[motor] = val
                    self._last_arm_obs[motor] = val
                    self._arm_overload_count[motor] = 0
                except Exception as exc:
                    exc_str = str(exc)
                    logger.warning(f"Arm read [{motor}] failed: {exc}")
                    # Hold last-known value instead of 0.0 (0.0 pollutes recorded data and
                    # would command the joint to its calibration centre).
                    arm_pos[motor] = self._last_arm_obs.get(motor, 0.0)
                    # Overload protection latches the STS3215 (no torque, no response).
                    # Cycle torque once to clear it — same recovery the hand bus uses.
                    # NOTE: this clears the latch but the joint re-overloads if it is still
                    # commanded into a stall (e.g. wrist_roll driven past its mechanical limit
                    # by full-IK wrist tracking). Reduce wrist_orientation_scale to stop that.
                    if "Overload error" in exc_str or "voltage" in exc_str.lower():
                        count = self._arm_overload_count.get(motor, 0) + 1
                        self._arm_overload_count[motor] = count
                        if count == 1:
                            try:
                                self.arm_bus.write("Torque_Enable", motor, 0)
                                self.arm_bus.write("Torque_Enable", motor, 1)
                            except Exception:
                                pass
        obs.update({f"{k}.pos": v for k, v in arm_pos.items()})

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    # ── Action ────────────────────────────────────────────────────────────────

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {k.removesuffix(".pos"): v for k, v in action.items() if k.endswith(".pos")}

        arm_goal = {k: v for k, v in goal_pos.items() if k in ARM_MOTORS}
        hand_goal = {k: v for k, v in goal_pos.items() if k in HAND_MOTORS}

        if self.config.max_relative_target is not None and arm_goal:
            present = self.arm_bus.sync_read("Present_Position")
            arm_goal = ensure_safe_goal_position(
                {k: (arm_goal[k], present[k]) for k in arm_goal if k in present},
                self.config.max_relative_target,
            )

        if hand_goal:
            with _big_endian():
                self.hand_bus.sync_write("Goal_Position", hand_goal)
            self._port_handler.clearPort()

        if arm_goal:
            _scs_def.SCS_END = 0
            self.arm_bus.sync_write("Goal_Position", arm_goal)

        return {f"{k}.pos": v for k, v in {**arm_goal, **hand_goal}.items()}

    # ── Disconnect ────────────────────────────────────────────────────────────

    @check_if_not_connected
    def disconnect(self) -> None:
        # Best-effort, fully isolated: each step is wrapped so ONE failure (e.g. a hand-bus
        # "Port is in use!" when Ctrl+C interrupts a transaction mid-record) can NEVER leave the
        # ARM stiff (torque on). Releasing arm torque is the priority on a hard stop.
        if self.config.disable_torque_on_disconnect:
            try:
                _scs_def.SCS_END = 0
                self.arm_bus.disable_torque()
            except Exception as exc:
                logger.warning(f"{self}: arm torque-off on disconnect failed: {exc}")
            try:
                # clearPort() resets the port's is_using flag in case Ctrl+C interrupted a
                # read/write (that stuck flag is what raises "Port is in use!").
                self._hand_port_handler.clearPort()
                with _big_endian():
                    self.hand_bus.sync_write("Torque_Enable", 0)
            except Exception as exc:
                logger.warning(f"{self}: hand torque-off on disconnect failed: {exc}")
        # Only close the arm port if WE own it. When two arms share one bus (single-port dual_arm),
        # the shared handler is owned/closed by the dual_arm follower AFTER both arms disable torque
        # — closing it here would slam it shut before the other arm can release its torque.
        if self._owns_arm_port:
            try:
                self._port_handler.closePort()
            except Exception as exc:
                logger.warning(f"{self}: closePort failed: {exc}")
        if self._separate_hand_port:
            try:
                self._hand_port_handler.closePort()
            except Exception as exc:
                logger.warning(f"{self}: hand closePort failed: {exc}")
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception as exc:
                logger.warning(f"{self}: camera disconnect failed: {exc}")
        logger.info(f"{self} disconnected.")
