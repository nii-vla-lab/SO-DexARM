#!/usr/bin/env python
"""SO-DexARM pipeline CLI: calibrate / teleop / record / train / eval.

Originally scripts/quest_hts/phase32_dual_arm_calibrate_teleop.py, integrated into the
LeRobot package as the single entry point for the Quest-teleoperated Dual-Arm manipulator.

Subcommands
-----------
calibrate-arm   -- Run LeRobot arm calibration for one side (interactive)
calibrate-hand  -- Capture AmazingHand fist / mid / open poses (robot + human)
capture-startup -- Save current arm joint positions as the start position
setup-motors    -- Assign the SO-101 motor IDs for one side (interactive)
teleop          -- Start Dual-Arm teleoperation (Meta Quest → robot)
record          -- Start LeRobot dataset recording
train           -- Print/run LeRobot train command
eval            -- Print/run LeRobot eval command

Typical workflow
----------------
# 1. Calibrate both SO-101 arms (run once, takes ~3 min each)
lerobot-so-dexarm calibrate-arm --side right
lerobot-so-dexarm calibrate-arm --side left

# 2. Capture AmazingHand calibration poses (3 poses × 2 sides), captured in the order
#    open(パー) → mid(中間) → fist(グー). All three are anchors of the piecewise-linear
#    open/close mapping.
lerobot-so-dexarm calibrate-hand --side right --all-poses --from-hardware
lerobot-so-dexarm calibrate-hand --side left  --all-poses --from-hardware

# 3. Capture start positions (move arms to desired start pose, then read the hardware)
lerobot-so-dexarm capture-startup --side both --from-hardware

# 4. Teleoperate
lerobot-so-dexarm teleop

# 5. Record a dataset
lerobot-so-dexarm record --repo-id myuser/my-task --num-episodes 20
"""

from __future__ import annotations

import argparse
import contextlib
import json
import socket as _socket
import subprocess
import sys
import threading as _threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Constants / defaults ────────────────────────────────────────────────────

# Canonical convention: right names always refer to the physical right side and left names to the
# physical left side. The udev template creates these stable, installation-independent names.
DEFAULT_RIGHT_ARM_PORT = "/dev/ttyso101_amazinghand_r_arm"
DEFAULT_LEFT_ARM_PORT = "/dev/ttyso101_amazinghand_l_arm"
DEFAULT_RIGHT_ARM_IDS = (1, 2, 3, 4, 5)
DEFAULT_LEFT_ARM_IDS = (1, 2, 3, 4, 5)
DEFAULT_RIGHT_HAND_PORT = "/dev/ttyso101_amazinghand_r_hand"
DEFAULT_LEFT_HAND_PORT = "/dev/ttyso101_amazinghand_l_hand"

DEFAULT_RIGHT_ARM_CALIB_DIR = Path(".cache/calibration/robots/so101_amazinghand_right")
DEFAULT_LEFT_ARM_CALIB_DIR = Path(".cache/calibration/robots/so101_amazinghand_left")

# Camera identifiers are installation-specific. Users must provide a draccus camera dictionary or
# explicitly choose `none` for a state-only run.
DEFAULT_CAMERAS = None

# Unified calibration dir for DualArm (r.json + l.json)
DEFAULT_DUAL_ARM_CALIB_DIR = Path(".cache/calibration/robots/dual_arm")

DEFAULT_RUNTIME_CONFIG_DIR = Path(".cache/so_dexarm")
DEFAULT_HAND_CALIB_FILE = DEFAULT_RUNTIME_CONFIG_DIR / "quest_hts_dual_arm_hand_calibration.yaml"
DEFAULT_STARTUP_FILE = DEFAULT_RUNTIME_CONFIG_DIR / "quest_hts_dual_arm_startup.yaml"
DEFAULT_HTS_HOST = "0.0.0.0"
DEFAULT_HTS_PORT = 8000

# Capture order: パー(open) → 中間(mid) → グー(fist). All three are anchors of the
# piecewise-linear open/close mapping.
POSES = ("open", "mid", "fist")


# ── Helpers ─────────────────────────────────────────────────────────────────


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: list[str], *, dry_run: bool = False) -> int:
    import shlex

    print("$ " + " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def _print_cmd(cmd: list[str]) -> None:
    import shlex

    print(" ".join(shlex.quote(str(c)) for c in cmd), flush=True)


# ── Hardware adapters ────────────────────────────────────────────────────────


class _ArmAdapter:
    """Minimal read/write adapter for one SO-101 arm (arm-only mode)."""

    def __init__(self, *, port: str, calib_dir: Path, arm_ids: tuple[int, ...], side: str = "right"):
        self._port = port
        self._calib_dir = calib_dir
        self._arm_ids = arm_ids
        self._side = side
        self._robot = None

    def connect(self) -> None:
        from lerobot.robots.so101_amazinghand_right import SO101AmazingHandRightConfig
        from lerobot.robots.utils import make_robot_from_config

        self._robot = make_robot_from_config(
            SO101AmazingHandRightConfig(
                id=self._side,
                port=self._port,
                arm_motor_ids=self._arm_ids,
                mode="arm-only",
                calibration_dir=self._calib_dir,
                require_calibration=True,
            )
        )
        self._robot.connect(calibrate=False)

    def read_action(self) -> dict[str, float]:
        """Return arm joint positions with unprefixed keys (shoulder_pan.pos, etc.)."""
        if self._robot is None:
            return {}
        obs = self._robot.get_observation()
        result = {}
        for k, v in obs.items():
            if k.endswith(".pos") and isinstance(v, int | float):
                # Strip any r_/l_ prefix so keys are canonical (shoulder_pan.pos etc.)
                bare = k[2:] if k.startswith(("r_", "l_")) else k
                result[bare] = float(v)
        return result

    def send_action(self, action: dict[str, float]) -> None:
        if self._robot is not None:
            self._robot.send_action(action)

    def disconnect(self) -> None:
        if self._robot is not None and self._robot.is_connected:
            self._robot.disconnect()
        self._robot = None


class _HandAdapter:
    """Read adapter for one AmazingHand (SCS0009 IDs 1-8, protocol 1).

    Uses per-motor individual connections to avoid sync_read (not supported by SCS0009).
    """

    def __init__(self, *, port: str, side: str):
        self._port = port
        self._side = side

    # SCS0009 raw tick range (from DualArmConfig defaults)
    _RANGE_MIN = 200
    _RANGE_MAX = 800

    def read_action(self) -> dict[str, float]:
        """Read all 8 motor positions one-by-one, return normalized [0-100] values.

        Uses normalize=False (raw ticks) to avoid requiring a calibration file,
        then normalizes manually using the default SCS0009 range [200, 800].
        """
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        prefix = "r_" if self._side == "right" else "l_"
        result = {}
        for i in range(1, 5):
            for j in range(1, 3):
                motor_name = f"finger{i}_motor{j}"
                motor_id = (i - 1) * 2 + j
                key = f"{prefix}{motor_name}.pos"
                # Use RANGE_M100_100 norm (any norm is fine since we read normalize=False)
                bus = FeetechMotorsBus(
                    port=self._port,
                    motors={motor_name: Motor(motor_id, "scs0009", MotorNormMode.RANGE_M100_100)},
                    protocol_version=1,
                )
                try:
                    bus.connect(handshake=False)
                    raw = float(bus.read("Present_Position", motor_name, normalize=False))
                    span = self._RANGE_MAX - self._RANGE_MIN
                    normalized = (raw - self._RANGE_MIN) / span * 100.0 if span > 0 else 50.0
                    normalized = max(0.0, min(100.0, normalized))
                    result[key] = normalized
                    print(f"    {key}: raw={int(raw)}  norm={normalized:.1f}", flush=True)
                except Exception as exc:
                    print(f"    WARNING: {key} read failed: {exc}", flush=True)
                finally:
                    with contextlib.suppress(Exception):
                        bus.disconnect(disable_torque=False)
        return result


# ── Persistent Quest HTS session ─────────────────────────────────────────────


class _QuestHTSSession:
    """TCP server that accepts one Quest HTS connection and buffers incoming data.

    Keeps the connection alive across multiple captures (e.g. all 3 hand poses).
    Usage:
        with _QuestHTSSession(host=..., port=...) as session:
            session.wait_connected(timeout_s=60)
            session.clear()
            input("Make FIST pose, then ENTER")
            landmarks = session.get_latest_landmarks(side)
    """

    def __init__(self, *, host: str, port: int):
        self._host = host
        self._port = port
        self._server: _socket.socket | None = None
        self._conn: _socket.socket | None = None
        self._lock = _threading.Lock()
        self._stop = _threading.Event()
        self._thread: _threading.Thread | None = None
        self._latest: dict[str, dict] = {}  # {side: {"wrist": tuple, "landmarks": tuple}}
        self._connected = False

    def __enter__(self):
        self._server = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        self._server.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        self._server.bind((self._host, self._port))
        self._server.listen(1)
        self._server.settimeout(0.5)
        self._stop.clear()
        self._thread = _threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._conn:
            with contextlib.suppress(OSError):
                self._conn.close()
        if self._server:
            with contextlib.suppress(OSError):
                self._server.close()
        if self._thread:
            self._thread.join(timeout=2.0)

    def wait_connected(self, *, timeout_s: float = 120.0) -> bool:
        """Wait until TCP connection is established (any side)."""
        deadline = time.monotonic() + timeout_s
        last_print = time.monotonic() - 10
        while time.monotonic() < deadline:
            with self._lock:
                if self._connected:
                    return True
            now = time.monotonic()
            if now - last_print >= 5.0:
                print(f"  Waiting for Quest HTS on {self._host}:{self._port}...", flush=True)
                last_print = now
            time.sleep(0.1)
        return False

    def wait_for_any_data(self, *, timeout_s: float = 30.0) -> dict[str, list[str]]:
        """Wait until some data arrives. Returns dict of {side: [kinds]} received."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                received = {side: list(data.keys()) for side, data in self._latest.items() if data}
            if received:
                return received
            time.sleep(0.1)
        return {}

    def clear(self) -> None:
        """Reset latest data so next get_ call returns only fresh frames."""
        with self._lock:
            self._latest.clear()

    def get_latest_landmarks(self, side: str, *, wait_s: float = 5.0) -> tuple | None:
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            with self._lock:
                val = self._latest.get(side, {}).get("landmarks")
            if val is not None:
                return val
            time.sleep(0.05)
        return None

    def get_latest_wrist(self, side: str) -> tuple | None:
        with self._lock:
            return self._latest.get(side, {}).get("wrist")

    def _loop(self) -> None:
        from lerobot.teleoperators.quest_hts.hts_protocol import parse_hts_line

        while not self._stop.is_set():
            try:
                conn, addr = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            print(f"  Quest HTS connected from {addr}", flush=True)
            with self._lock:
                self._connected = True
                first_frame_logged = False
            self._conn = conn
            buf = ""
            conn.settimeout(0.5)
            line_count = 0
            parsed_count = 0
            try:
                while not self._stop.is_set():
                    try:
                        data = conn.recv(4096)
                    except TimeoutError:
                        continue
                    except OSError:
                        break
                    if not data:
                        break
                    buf += data.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        line_count += 1
                        parsed = parse_hts_line(line)
                        if parsed is None:
                            if line_count <= 3:
                                print(f"  [HTS] unparsed line: {line[:80]!r}", flush=True)
                            continue
                        parsed_count += 1
                        side, kind, values = parsed
                        with self._lock:
                            entry = self._latest.setdefault(side, {})
                            entry[kind] = tuple(float(v) for v in values)
                            if not first_frame_logged and kind == "landmarks":
                                first_frame_logged = True
                                print(
                                    f"  [HTS] first {side} landmarks received ({len(values)} values)",
                                    flush=True,
                                )
            finally:
                with contextlib.suppress(OSError):
                    conn.close()
                with self._lock:
                    self._connected = False
                self._conn = None
            print(f"  Quest HTS disconnected. (lines={line_count} parsed={parsed_count})", flush=True)


# ── Subcommand: setup-motors ─────────────────────────────────────────────────


def cmd_setup_motors(args: argparse.Namespace) -> int:
    """Assign the five SO-101 arm motor IDs, one physically connected motor at a time."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    side = args.side
    port = args.right_arm_port if side == "right" else args.left_arm_port
    ids = args.right_arm_ids if side == "right" else args.left_arm_ids
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    motors = {
        name: Motor(motor_id, "sts3215", MotorNormMode.RANGE_M100_100)
        for name, motor_id in zip(names, ids, strict=True)
    }

    print(f"\n=== SO-DexARM motor setup: {side.upper()} arm ===", flush=True)
    print(f"Port: {port}", flush=True)
    for name, motor_id in zip(names, ids, strict=True):
        print(f"  {name:<14} -> ID {motor_id}", flush=True)
    print(
        "\nConnect exactly ONE motor to the controller for each prompt. "
        "Multiple connected motors can cause the wrong servo ID to be overwritten.\n",
        flush=True,
    )

    bus = FeetechMotorsBus(port=port, motors=motors, protocol_version=0)
    try:
        for name in names:
            if not args.yes:
                input(f"Connect only '{name}', then press ENTER…")
            try:
                bus.setup_motor(name)
            except Exception as exc:
                print(f"ERROR: failed to configure {name}: {exc}", flush=True)
                return 1
            print(f"  Configured {name} as ID {motors[name].id}.", flush=True)
    finally:
        if bus.is_connected:
            bus.disconnect()

    print("Motor setup complete.", flush=True)
    return 0


# ── Subcommand: calibrate-arm ────────────────────────────────────────────────


def cmd_calibrate_arm(args: argparse.Namespace) -> int:
    """Run `lerobot calibrate` for one SO-101 arm.

    The calibration file is saved as:
      right: .cache/calibration/robots/so101_amazinghand_right/right.json
      left:  .cache/calibration/robots/so101_amazinghand_left/left.json
    """
    side = args.side
    port = args.right_arm_port if side == "right" else args.left_arm_port
    calib_dir = args.right_calib_dir if side == "right" else args.left_calib_dir
    robot_id = side  # "right" or "left" → becomes the calibration file basename
    arm_ids = args.right_arm_ids if side == "right" else args.left_arm_ids
    ids_arg = "[" + ",".join(str(i) for i in arm_ids) + "]"

    print(f"\n=== Calibrating {side.upper()} arm ===", flush=True)
    print(f"  port={port}  ids={arm_ids}  calib_dir={calib_dir}", flush=True)
    print("  Follow the on-screen instructions to move each joint.", flush=True)

    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_calibrate",
        "--robot.type",
        "so101_amazinghand_right",
        "--robot.id",
        robot_id,
        "--robot.port",
        port,
        f"--robot.arm_motor_ids={ids_arg}",
        "--robot.mode",
        "arm-only",
        "--robot.calibration_dir",
        str(calib_dir),
    ]
    rc = _run(cmd)
    if rc == 0:
        _sync_dual_arm_calib_files(args)
    return rc


# ── Subcommand: calibrate-hand ───────────────────────────────────────────────


def cmd_calibrate_hand(args: argparse.Namespace) -> int:
    """Capture AmazingHand calibration poses (open パー / mid ピンチ / fist グー) for one side.

    Captures robot motor positions AND human hand landmarks SIMULTANEOUSLY.
    Quest HTS must be running and streaming to this PC before starting.

    The flow for each pose:
      1. Connect Quest HTS (once for all poses in this session).
      2. Verify Quest is tracking the specified hand (--quest-side).
      3. Prompt: "Move AmazingHand to [pose] AND show same pose → ENTER"
      4. Read robot hand positions from hardware.
      5. Read latest Quest landmarks for the quest_side.
      6. Save both to calibration YAML.

    --quest-side controls which Quest hand stream to use for landmarks.
    Defaults to the same as --side, but can be overridden (e.g. if Quest labels
    your left hand as "right" in your physical setup).
    """
    side = args.side
    quest_side = getattr(args, "quest_side", side) or side
    hand_port = args.right_hand_port if side == "right" else args.left_hand_port
    hand_calib_file: Path = args.hand_calib_file
    from_hardware = getattr(args, "from_hardware", False)
    all_poses = getattr(args, "all_poses", False)

    poses_to_capture = list(POSES) if all_poses else [args.pose]

    print(
        f"\n=== Calibrate hand: robot_side={side}  quest_side={quest_side}  poses={poses_to_capture} ===",
        flush=True,
    )
    print(f"  hand_port={hand_port}  calib_file={hand_calib_file}", flush=True)
    if side != quest_side:
        print(f"  NOTE: Using Quest '{quest_side}' stream to calibrate robot '{side}' hand.", flush=True)
    print("\n  !! Quest HTS Streamer must be running on Meta Quest.", flush=True)
    print(f"  !! Configure it to send to this PC's IP on port {args.hts_port}.", flush=True)

    from lerobot.teleoperators.quest_hts.dual_arm_calibration import (
        save_human_hand_pose,
        save_robot_hand_pose,
    )

    with _QuestHTSSession(host=args.hts_host, port=args.hts_port) as session:
        print(f"\n  Waiting for Quest HTS connection (port {args.hts_port})...", flush=True)
        if not session.wait_connected(timeout_s=120.0):
            print("  ERROR: Quest HTS timed out. Check streamer settings.", flush=True)
            return 1

        # Show what Quest is actually sending (diagnostic)
        print("  Quest HTS connected! Checking which hands are tracked...", flush=True)
        received = session.wait_for_any_data(timeout_s=5.0)
        if received:
            for s, kinds in sorted(received.items()):
                print(f"    Quest is sending: {s} hand ({', '.join(kinds)})", flush=True)
        else:
            print("  WARNING: Quest connected but no data received yet.", flush=True)

        if quest_side not in received:
            print(
                f"\n  NOTICE: Quest is not sending '{quest_side}' hand data yet.\n"
                f"  Sending so far: {list(received.keys()) or 'none'}\n"
                f"  If Quest labels your physical '{side}' hand as the other side,\n"
                f"  re-run with --quest-side {'left' if quest_side == 'right' else 'right'}",
                flush=True,
            )
        print("", flush=True)

        for pose in poses_to_capture:
            print(f"\n--- Pose: {pose.upper()} ({_pose_hint(pose)}) ---", flush=True)

            # Step A: Verify Quest is tracking the correct hand BEFORE prompting
            print(f"  Waiting for Quest to track your '{quest_side}' hand...", flush=True)
            pre_landmarks = session.get_latest_landmarks(quest_side, wait_s=15.0)
            if pre_landmarks is None:
                # Show what IS being received to help user
                with session._lock:
                    available = {s: list(d.keys()) for s, d in session._latest.items() if d}
                print(
                    f"  ERROR: Quest is not tracking the '{quest_side}' hand after 15s.\n"
                    f"  Quest is currently tracking: {available or 'nothing'}\n"
                    f"  Options:\n"
                    f"    1. Show your physical {'right' if quest_side == 'right' else 'left'} hand to Quest cameras\n"
                    f"    2. If Quest labels it as the other side, re-run with --quest-side {'left' if quest_side == 'right' else 'right'}",
                    flush=True,
                )
                return 1
            print(f"  Quest is tracking your '{quest_side}' hand!", flush=True)

            # Step B: Ask user to hold the pose
            if from_hardware:
                print(
                    f"  Move {side} AmazingHand to '{pose}' AND hold the same pose with your {side} human hand.",
                    flush=True,
                )
            else:
                print(f"  Make the '{pose}' pose with your {side} human hand.", flush=True)
            print("  Hold the pose steady, then press ENTER to capture.", flush=True)

            # Clear old data just before Enter so we get only the held pose
            session.clear()
            input()

            # Step C: Read robot positions from hardware
            robot_positions: dict[str, float] = {}
            if from_hardware:
                print(f"  Reading {side} hand positions from {hand_port}...", flush=True)
                adapter = _HandAdapter(port=hand_port, side=side)
                robot_positions = adapter.read_action()
                if not robot_positions:
                    print(f"  WARNING: No robot hand positions read. Check {hand_port}.", flush=True)

            # Step D: Get fresh Quest landmarks using quest_side stream
            print(f"  Capturing Quest landmarks for '{quest_side}' hand ({pose})...", flush=True)
            landmarks = session.get_latest_landmarks(quest_side, wait_s=5.0)
            if landmarks is None:
                # Quest might have moved; try getting any recent data
                print("  Retrying...", flush=True)
                session.clear()
                landmarks = session.get_latest_landmarks(quest_side, wait_s=5.0)
            if landmarks is None:
                print(
                    f"  ERROR: No Quest landmarks received for '{quest_side}' hand.\n"
                    "  The hand may have moved out of Quest view. Please retry this pose.",
                    flush=True,
                )
                return 1
            print(f"  Captured {len(landmarks)} landmark values.", flush=True)

            # Save to YAML
            save_human_hand_pose(
                path=hand_calib_file,
                side=side,
                pose=pose,
                landmarks=landmarks,
                feature_mode=getattr(args, "hand_feature_mode", "curl"),
            )
            if robot_positions:
                save_robot_hand_pose(path=hand_calib_file, side=side, pose=pose, positions=robot_positions)

            print(f"  Saved {side} {pose} pose to {hand_calib_file}", flush=True)
            if pose != poses_to_capture[-1]:
                print("", flush=True)

    print(f"\n[calibrate-hand] Done: side={side} poses={poses_to_capture}", flush=True)
    return 0


def _pose_hint(pose: str) -> str:
    return {
        "open": "all fingers open / パー",
        "mid": "fingers half-closed / 中間",
        "fist": "all fingers closed / グー",
    }.get(pose, pose)


# ── Subcommand: capture-startup ──────────────────────────────────────────────


def cmd_capture_startup(args: argparse.Namespace) -> int:
    """Read current arm positions from hardware and save as the start position."""
    sides: tuple[str, ...]
    sides = ("right", "left") if args.side == "both" else (args.side,)

    startup_file: Path = args.startup_file
    errors = []

    for side in sides:
        port = args.right_arm_port if side == "right" else args.left_arm_port
        calib_dir = args.right_calib_dir if side == "right" else args.left_calib_dir

        print(f"\n=== Capture startup: side={side} ===", flush=True)
        print(f"  Move {side} arm to the desired start pose, then press ENTER.", flush=True)
        input()

        if not getattr(args, "from_hardware", False):
            print(f"  --from-hardware not set; saving zero positions for {side}.", flush=True)
            arm_positions: dict[str, float] = {}
        else:
            print(f"  Reading {side} arm positions from {port}...", flush=True)
            arm_ids = args.right_arm_ids if side == "right" else args.left_arm_ids
            adapter = _ArmAdapter(port=port, calib_dir=calib_dir, arm_ids=arm_ids, side=side)
            try:
                adapter.connect()
                arm_positions = adapter.read_action()
                print(f"  {side} arm positions: {arm_positions}", flush=True)
            except Exception as exc:
                print(f"  ERROR: {exc}", flush=True)
                errors.append(f"{side}: {exc}")
                continue
            finally:
                adapter.disconnect()

        from lerobot.teleoperators.quest_hts.dual_arm_calibration import save_start_position

        save_start_position(
            path=startup_file,
            side=side,
            arm_positions=arm_positions,
        )

    if errors:
        print(f"[capture-startup] Errors: {errors}", flush=True)
        return 1

    # Sync arm calibration files to dual_arm format
    # (r.json / l.json in one unified directory)
    _sync_dual_arm_calib_files(args)

    print("[capture-startup] Done.", flush=True)
    return 0


def _sync_dual_arm_calib_files(args: argparse.Namespace) -> None:
    """Refresh the Dual-Arm-level calibration after an arm recalibration: per-arm r.json/l.json AND the
    merged Dual-Arm-level file, keeping the merged file's ARM half FRESH and its HAND half intact.

    Why this is delicate (DualArm behaviour):
      - It loads the merged calibration_fpath ('{id}.json'; id defaults to None → None.json) and
        SPLITS it to OVERRIDE each sub-arm's own r.json/l.json (`if r_cal: self.r_arm.calibration
        = r_cal`). So a STALE merged file silently reverts a freshly-recalibrated arm to the OLD
        calibration, which can make the arm move toward an invalid start target.
      - BUT the 8-per-side HAND (SCS0009) calibration lives ONLY in that merged file — r.json/l.json
        hold the 5 ARM motors only. So simply DELETING the merged file drops the hand calibration
        and connect() then fails with "FeetechMotorsBus … has no calibration registered" on the
        hand (teleop runs calibrate=False, so auto-hand-calibration never runs).
    Therefore: rebuild the merged file = FRESH arm (from r.json/l.json) + PRESERVED hand (the
    r_finger*/l_finger* entries from the existing merged file, or a backup that still has them).
    """
    import json
    import shutil

    dual_arm_dir = DEFAULT_DUAL_ARM_CALIB_DIR
    dual_arm_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        args.right_calib_dir / "right.json": dual_arm_dir / "r.json",
        args.left_calib_dir / "left.json": dual_arm_dir / "l.json",
    }
    for src, dst in mapping.items():
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Synced {src} → {dst}", flush=True)
        else:
            print(f"  WARNING: {src} not found (arm not yet calibrated?)", flush=True)

    # Rebuild the merged Dual-Arm-level file with FRESH arm + PRESERVED hand.
    merged: dict = {}
    for side_file, prefix in ((dual_arm_dir / "r.json", "r_"), (dual_arm_dir / "l.json", "l_")):
        if side_file.exists():
            with side_file.open() as input_file:
                side_calibration = json.load(input_file)
            for k, v in side_calibration.items():
                merged[f"{prefix}{k}"] = v
    # Pull hand (finger) entries from the first source that still has them. Newest merged first,
    # then legacy twin, then any stale-bak backup (covers a merged file already clobbered to arm-only).
    hand_kept = 0
    candidates = ["dual_arm.json", "None.json", "bi_amazinghand_follower.json"]
    candidates += sorted((p.name for p in dual_arm_dir.glob("*.stale-bak-*")), reverse=True)
    for cand in candidates:
        p = dual_arm_dir / cand
        if not p.exists():
            continue
        try:
            with p.open() as input_file:
                existing = json.load(input_file)
        except Exception:
            continue
        added = False
        for k, v in existing.items():
            if "finger" in k and k not in merged:
                merged[k] = v
                hand_kept += 1
                added = True
        if added:
            break  # got the hand half from this source
    if hand_kept == 0:
        print(
            "  WARNING: no HAND calibration found to preserve — run calibrate-hand "
            "(the merged file will lack finger calibration and teleop will fail on the hand).",
            flush=True,
        )

    # Write the canonical file. The older names above are read-only migration fallbacks.
    with open(dual_arm_dir / "dual_arm.json", "w") as output_file:
        json.dump(merged, output_file, indent=4)
    print(
        f"  Rebuilt merged calibration ({len([k for k in merged if 'finger' not in k])} arm + "
        f"{hand_kept} hand) → {dual_arm_dir / 'dual_arm.json'}",
        flush=True,
    )


# ── Subcommand: teleop ───────────────────────────────────────────────────────


def _teleop_args(args: argparse.Namespace) -> list[str]:
    """Shared robot + teleop CLI args for both teleop and record commands.

    LeRobot uses snake_case (underscores) for all config field args.

    DualArm internally creates SO101AmazingHandFollower with
    id="r" and id="l", so it looks for r.json / l.json in calibration_dir.
    We copy right.json → r.json and left.json → l.json into DEFAULT_DUAL_ARM_CALIB_DIR.
    """
    dual_arm_calib_dir = getattr(args, "dual_arm_calib_dir", DEFAULT_DUAL_ARM_CALIB_DIR)
    out = [
        "--robot.type",
        "dual_arm",
        "--robot.id",
        "dual_arm",
        "--robot.port_r",
        args.right_arm_port,
        "--robot.port_l",
        args.left_arm_port,
        "--robot.hand_port_r",
        args.right_hand_port,
        "--robot.hand_port_l",
        args.left_hand_port,
        "--robot.arm_ids_r",
        _motor_ids_arg(args.right_arm_ids),
        "--robot.arm_ids_l",
        _motor_ids_arg(args.left_arm_ids),
        "--robot.calibration_dir",
        str(dual_arm_calib_dir),
        "--teleop.type",
        "quest_hts_dual_arm",
        "--teleop.host",
        args.hts_host,
        "--teleop.port",
        str(args.hts_port),
        "--teleop.startup_file",
        str(args.startup_file),
        "--teleop.hand_calib_file",
        str(args.hand_calib_file),
        "--teleop.right_arm_calib_file",
        str(args.right_calib_dir / "right.json"),
        "--teleop.left_arm_calib_file",
        str(args.left_calib_dir / "left.json"),
        "--teleop.mode",
        getattr(args, "teleop_mode", "arm-and-hand"),
    ]
    # Single-side mode (e.g. only the LEFT arm+hand is physically connected). "both" = full
    # dual_arm (default). Passing left/right makes BOTH the follower and the teleoperator expose +
    # drive only that side (13 DOF), and that one Quest hand becomes mandatory.
    sides = getattr(args, "sides", "both")
    out += ["--robot.sides", sides, "--teleop.active_sides", sides]
    if sides != "both":
        out += ["--teleop.require_both_sides", "false"]
    if getattr(args, "swap_sides", False):
        out += ["--teleop.swap_sides", "true"]
    if getattr(args, "tcp_offset_m", None) is not None:
        out += ["--teleop.tcp_offset_m", str(args.tcp_offset_m)]
    return out


def _build_teleop_cmd(args: argparse.Namespace, *, extra: list[str] | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_teleoperate"] + _teleop_args(args)
    cmd += ["--fps", str(getattr(args, "fps", 30))]
    if extra:
        cmd.extend(extra)
    return cmd


def cmd_teleop(args: argparse.Namespace) -> int:
    print("\n=== SO-DexARM Teleop (Dual-Arm) ===", flush=True)
    cmd = _build_teleop_cmd(args, extra=list(getattr(args, "extra", []) or []))
    if getattr(args, "print_only", False):
        _print_cmd(cmd)
        return 0
    return _run(cmd)


# ── Subcommand: record ───────────────────────────────────────────────────────


def cmd_record(args: argparse.Namespace) -> int:
    if not args.repo_id:
        print("ERROR: --repo-id is required for record.", flush=True)
        return 1

    print(f"\n=== SO-DexARM Record (Dual-Arm): {args.repo_id} ===", flush=True)
    cmd = (
        [sys.executable, "-m", "lerobot.scripts.lerobot_record"]
        + _teleop_args(args)
        + [
            "--dataset.repo_id",
            args.repo_id,
            "--dataset.single_task",
            getattr(args, "single_task", "SO-DexARM task") or "SO-DexARM task",
            "--dataset.num_episodes",
            str(args.num_episodes),
            "--dataset.episode_time_s",
            str(args.episode_time_s),
            "--dataset.reset_time_s",
            str(args.reset_time_s),
            "--dataset.fps",
            str(getattr(args, "fps", 30)),
            "--dataset.push_to_hub",
            "true" if args.push_to_hub else "false",
        ]
    )
    # Cameras (record only): without this the dataset has no images. _teleop_args is shared with
    # teleop, so the camera flag is appended here rather than there (teleop must not open cameras).
    cams = (getattr(args, "cameras", "") or "").strip()
    if not cams:
        print(
            "ERROR: --cameras must be provided for recording. Pass an installation-specific "
            "camera dictionary, or pass --cameras none explicitly for a state-only dataset.",
            flush=True,
        )
        return 1
    if "<" in cams or ">" in cams:
        print("ERROR: --cameras still contains an unreplaced <...> placeholder.", flush=True)
        return 1
    if cams and cams.lower() != "none":
        cmd += ["--robot.cameras", cams]
        print(f"  Cameras: {cams}", flush=True)
    else:
        print("  Cameras: NONE (state-only dataset — no images recorded)", flush=True)
    if getattr(args, "display", False):
        cmd += ["--display_data", "true"]
        print("  Display: ON (Rerun viewer — shows the RealSense feed + state)", flush=True)
    cmd += list(getattr(args, "extra", []) or [])
    if getattr(args, "print_only", False):
        _print_cmd(cmd)
        return 0
    return _run(cmd)


# ── Subcommand: train ────────────────────────────────────────────────────────


def cmd_train(args: argparse.Namespace) -> int:
    if not args.repo_id:
        print("ERROR: --repo-id is required for train.", flush=True)
        return 1

    policy = getattr(args, "policy", "act")
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        "--dataset.repo_id",
        args.repo_id,
        "--policy.type",
        policy,
        # Required for local training: --policy.device (NOT --device, which lerobot_train rejects)
        # and --policy.push_to_hub=false (otherwise the trainer errors trying to push to the Hub).
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
    ]
    if getattr(args, "steps", None) is not None:
        cmd += ["--steps", str(args.steps)]
    if getattr(args, "batch_size", None) is not None:
        cmd += ["--batch_size", str(args.batch_size)]
    if getattr(args, "save_freq", None) is not None:
        cmd += ["--save_freq", str(args.save_freq)]
    if getattr(args, "output_dir", None):
        cmd += ["--output_dir", args.output_dir]
    if getattr(args, "job_name", None):
        cmd += ["--job_name", args.job_name]
    cmd += list(getattr(args, "extra", []) or [])
    print(f"\n=== Train ({policy}): {args.repo_id} ===", flush=True)
    if getattr(args, "print_only", False):
        _print_cmd(cmd)
        return 0
    return _run(cmd)


# ── Subcommand: eval ─────────────────────────────────────────────────────────


def cmd_eval(args: argparse.Namespace) -> int:
    if not args.pretrained_policy_path:
        print("ERROR: --pretrained-policy-path is required for eval.", flush=True)
        return 1

    # Real-robot eval uses lerobot_record with a policy (no teleop). The policy generates the
    # actions, so the Quest IK teleop (and tcp_offset_m etc.) is NOT in the loop here.
    # --policy.path MUST be a single '=' -joined token: lerobot's parser picks it up via
    # parse_arg("--policy.path=…") BEFORE argparse; space-separated it falls through to argparse
    # and errors as "unrecognized arguments".
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_record",
        "--robot.type",
        "dual_arm",
        "--robot.id",
        "dual_arm",
        "--robot.port_r",
        args.right_arm_port,
        "--robot.port_l",
        args.left_arm_port,
        "--robot.hand_port_r",
        args.right_hand_port,
        "--robot.hand_port_l",
        args.left_hand_port,
        "--robot.arm_ids_r",
        _motor_ids_arg(args.right_arm_ids),
        "--robot.arm_ids_l",
        _motor_ids_arg(args.left_arm_ids),
        # Reuse the SAME calibration the data was recorded with (r.json / l.json). Without this the
        # follower finds no calibration and forces a fresh arm calibration at startup.
        "--robot.calibration_dir",
        str(getattr(args, "dual_arm_calib_dir", DEFAULT_DUAL_ARM_CALIB_DIR)),
        f"--policy.path={args.pretrained_policy_path}",
        # LeRobot REQUIRES the dataset name (after '/') to start with 'eval_' when a policy is given.
        "--dataset.repo_id",
        getattr(args, "repo_id", None) or "local/eval_so-dexarm",
        "--dataset.single_task",
        getattr(args, "task", None) or "eval",
        "--dataset.num_episodes",
        str(getattr(args, "num_episodes", 5)),
        "--dataset.episode_time_s",
        str(getattr(args, "episode_time_s", 30)),
        "--dataset.reset_time_s",
        str(getattr(args, "reset_time_s", 10)),
        "--dataset.fps",
        str(getattr(args, "fps", 30)),
        "--dataset.push_to_hub",
        "false",
    ]
    # Cameras are REQUIRED for a visuomotor policy: it was trained with observation.images.cam_front,
    # so without the camera the observation is missing that input and inference fails. Use the same
    # default as record (the RealSense). --cameras none only works for a state-only policy.
    cams = (getattr(args, "cameras", "") or "").strip()
    if not cams:
        print(
            "ERROR: --cameras must match the policy's training inputs. Pass the camera dictionary, "
            "or pass --cameras none explicitly for a state-only policy.",
            flush=True,
        )
        return 1
    if "<" in cams or ">" in cams:
        print("ERROR: --cameras still contains an unreplaced <...> placeholder.", flush=True)
        return 1
    if cams and cams.lower() != "none":
        cmd += ["--robot.cameras", cams]
        print(f"  Cameras: {cams}", flush=True)
    else:
        print("  Cameras: NONE — only valid if the policy was trained state-only.", flush=True)
    # Anti-lunge: cap the per-step arm goal delta. Without a cap, the move-to-start and every
    # inference stall can make the arms jump at full servo speed.
    if getattr(args, "max_relative_target", None) is not None:
        cmd += [f"--robot.max_relative_target={args.max_relative_target}"]
    # Diffusion-policy latency: the stored default (num_inference_steps=None → num_train_timesteps
    # =100 denoise steps) can introduce a long pause between action chunks. NOTE:
    # non-diffusion policies (SmolVLA/ACT) REJECT
    # these two flags ("unrecognized arguments") — only use them when evaluating a diffusion policy.
    if getattr(args, "num_inference_steps", None) is not None:
        cmd += [f"--policy.num_inference_steps={args.num_inference_steps}"]
    if getattr(args, "noise_scheduler", None):
        cmd += [f"--policy.noise_scheduler_type={args.noise_scheduler}"]
    cmd += list(getattr(args, "extra", []) or [])
    if getattr(args, "display", False):
        cmd += ["--display_data", "true"]
    print(f"\n=== Eval: {args.pretrained_policy_path} ===", flush=True)
    if getattr(args, "print_only", False):
        _print_cmd(cmd)
        return 0
    return _run(cmd)


# ── Subcommand: map-hand (diagnostic) ────────────────────────────────────────


def cmd_map_hand(args: argparse.Namespace) -> int:
    from lerobot.teleoperators.quest_hts.dual_arm_calibration import build_hand_mapping

    sides = ("right", "left") if args.side == "both" else (args.side,)
    for side in sides:
        mapping = build_hand_mapping(side=side, hand_calib_file=args.hand_calib_file)
        print(json.dumps({side: mapping}, indent=2), flush=True)
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_motor_ids(value: str) -> tuple[int, ...]:
    cleaned = value.strip().removeprefix("[").removesuffix("]")
    try:
        ids = tuple(int(item.strip()) for item in cleaned.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("motor IDs must be comma-separated integers") from exc
    if len(ids) != 5 or len(set(ids)) != 5 or any(motor_id < 0 or motor_id > 253 for motor_id in ids):
        raise argparse.ArgumentTypeError("motor IDs must contain five unique integers from 0 to 253")
    return ids


def _motor_ids_arg(ids: tuple[int, ...]) -> str:
    return "[" + ",".join(str(motor_id) for motor_id in ids) + "]"


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--right-arm-port", default=DEFAULT_RIGHT_ARM_PORT)
    parser.add_argument("--left-arm-port", default=DEFAULT_LEFT_ARM_PORT)
    parser.add_argument("--right-hand-port", default=DEFAULT_RIGHT_HAND_PORT)
    parser.add_argument("--left-hand-port", default=DEFAULT_LEFT_HAND_PORT)
    parser.add_argument("--right-arm-ids", type=_parse_motor_ids, default=DEFAULT_RIGHT_ARM_IDS)
    parser.add_argument("--left-arm-ids", type=_parse_motor_ids, default=DEFAULT_LEFT_ARM_IDS)
    parser.add_argument("--right-calib-dir", type=Path, default=DEFAULT_RIGHT_ARM_CALIB_DIR)
    parser.add_argument("--left-calib-dir", type=Path, default=DEFAULT_LEFT_ARM_CALIB_DIR)
    parser.add_argument("--hts-host", default=DEFAULT_HTS_HOST)
    parser.add_argument("--hts-port", type=int, default=DEFAULT_HTS_PORT)
    parser.add_argument("--hand-calib-file", type=Path, default=DEFAULT_HAND_CALIB_FILE)
    parser.add_argument("--startup-file", type=Path, default=DEFAULT_STARTUP_FILE)
    parser.add_argument(
        "--tcp-offset-m",
        type=float,
        default=None,
        help="AmazingHand fingertip offset (m) from the wrist flange, for fingertip-"
        "accurate IK. Measure flange→fingertip and set it (config default 0.10). "
        "Makes lateral 1:1 too. Omit to use the config default; 0.0 = flange only.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SO-DexARM: an end-to-end system for controlling an AmazingHand + SO-ARM101 "
            "Dual-Arm manipulator with Meta Quest hand tracking, recording datasets, and "
            "running learned policies."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- setup-motors --
    p = sub.add_parser("setup-motors", help="Assign SO-101 arm motor IDs for one side.")
    _add_common_args(p)
    p.add_argument("--side", choices=("right", "left"), required=True)
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the one-motor-at-a-time confirmation prompts (unsafe unless wiring is verified).",
    )

    # -- calibrate-arm --
    p = sub.add_parser("calibrate-arm", help="Run LeRobot arm calibration for one side.")
    _add_common_args(p)
    p.add_argument("--side", choices=("right", "left"), required=True)

    # -- calibrate-hand --
    p = sub.add_parser(
        "calibrate-hand", help="Capture hand poses (open=パー, mid=中間, fist=グー) for robot + human."
    )
    _add_common_args(p)
    p.add_argument("--side", choices=("right", "left"), required=True, help="Which robot hand to calibrate.")
    p.add_argument(
        "--quest-side",
        choices=("right", "left"),
        default=None,
        help="Which Quest hand stream to use for landmarks (default = same as --side). "
        "Override if Quest labels your physical right hand as 'left' or vice versa.",
    )
    p.add_argument(
        "--pose",
        choices=POSES,
        default="open",
        help="Single pose to capture (ignored if --all-poses). open=パー, mid=中間, fist=グー.",
    )
    p.add_argument(
        "--all-poses",
        action="store_true",
        help="Capture all 3 poses (open → mid → fist = パー → 中間 → グー) in one Quest session.",
    )
    p.add_argument(
        "--from-hardware", action="store_true", help="Read current robot hand positions from hardware."
    )
    p.add_argument(
        "--hand-feature-mode",
        choices=("curl", "distance"),
        default="curl",
        help="Human-hand open/close feature. 'curl' (default) = sum of finger joint "
        "bend angles → uniform grip resolution; 'distance' = legacy tip-to-MCP. "
        "The chosen mode is stamped into the calib file and reused at runtime, so "
        "re-capture ALL poses for a side after changing this.",
    )

    # -- capture-startup --
    p = sub.add_parser("capture-startup", help="Save current arm positions as the start position.")
    _add_common_args(p)
    p.add_argument("--side", choices=("right", "left", "both"), default="both")
    p.add_argument(
        "--from-hardware",
        action="store_true",
        help="Read current arm positions from hardware (default: save zeros).",
    )

    # -- teleop --
    p = sub.add_parser("teleop", help="Start SO-DexARM teleoperation with the Dual-Arm manipulator.")
    _add_common_args(p)
    p.add_argument(
        "--sides",
        choices=("both", "right", "left"),
        default="both",
        help="Which side(s) to drive. 'both' (default) = full dual_arm. 'left'/'right' = "
        "single-arm: only that side's arm+hand is connected and driven (use when only "
        "one side is physically plugged in).",
    )
    p.add_argument("--teleop-mode", default="arm-and-hand", choices=("arm-only", "hand-only", "arm-and-hand"))
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Teleop control-loop target rate (Hz). Default 30 (matches the record fps and "
        "the 30fps camera). Passed to lerobot_teleoperate --fps.",
    )
    p.add_argument(
        "--swap-sides",
        action="store_true",
        help="Swap which Quest hand drives which logical side (arm+hand+calib together, "
        "coherently). Use to flip left/right without splitting arm from hand.",
    )
    p.add_argument("--print-only", action="store_true", help="Print the command without running it.")
    p.add_argument(
        "extra",
        nargs="*",
        default=[],
        help="Extra args appended verbatim to lerobot_teleoperate; put them after '--'.",
    )

    # -- record --
    p = sub.add_parser("record", help="Start LeRobot dataset recording.")
    _add_common_args(p)
    p.add_argument("--repo-id", required=True)
    p.add_argument(
        "--single-task",
        default="SO-DexARM task",
        help="Short task description for the dataset (e.g. 'pick and place cube').",
    )
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--episode-time-s", type=int, default=30)
    p.add_argument("--reset-time-s", type=int, default=5)
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording/control rate (Hz). Default 30 (matches the 30fps camera). If the "
        "hardware can't sustain it (the hand reads 8 SCS0009 motors one-by-one), the "
        "loop runs slower than this.",
    )
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument(
        "--cameras",
        default=DEFAULT_CAMERAS,
        help="Installation-specific draccus dict-string passed to --robot.cameras, for example "
        "'{cam_front: {type: intelrealsense, serial_number_or_name: <CAMERA_SERIAL>, "
        "width: 640, height: 480, fps: 30}}'. Use 'none' for state-only recording.",
    )
    p.add_argument(
        "--display",
        action="store_true",
        help="Show the Rerun viewer (camera feed + state) while recording (--display_data=true).",
    )
    p.add_argument(
        "--swap-sides",
        action="store_true",
        help="Swap which Quest hand drives which logical side (arm+hand+calib together, "
        "coherently). Use to flip left/right without splitting arm from hand.",
    )
    p.add_argument(
        "--sides",
        choices=("both", "right", "left"),
        default="both",
        help="Which side(s) to record. 'both' (default) = full dual_arm. 'left'/'right' = "
        "single-arm: only that side's arm+hand is connected and recorded (13 DOF).",
    )
    p.add_argument("--teleop-mode", default="arm-and-hand", choices=("arm-only", "hand-only", "arm-and-hand"))
    p.add_argument("--print-only", action="store_true", help="Print the command without running it.")
    p.add_argument(
        "extra",
        nargs="*",
        default=[],
        help="Extra args appended verbatim to lerobot_record; put them after '--'.",
    )

    # -- train --
    p = sub.add_parser("train", help="Run LeRobot training.")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--policy", default="act", choices=("act", "diffusion", "pi0", "smolvla", "tdmpc"))
    p.add_argument("--device", default="cuda", help="Training device passed to --policy.device.")
    p.add_argument("--steps", type=int, default=None, help="Training steps (lerobot default if omitted).")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--save-freq", type=int, default=None, help="Checkpoint save interval in steps.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--job-name", default=None)
    p.add_argument("--print-only", action="store_true")
    p.add_argument(
        "extra",
        nargs="*",
        default=[],
        help="Extra args appended verbatim to lerobot_train; put them after '--', "
        "e.g. -- --policy.optimizer_lr=1e-5 --wandb.enable=true",
    )

    # -- eval --
    p = sub.add_parser("eval", help="Run LeRobot policy evaluation on hardware.")
    _add_common_args(p)
    p.add_argument("--pretrained-policy-path", required=True)
    p.add_argument(
        "--repo-id",
        default=None,
        help="Output dataset repo_id. The name after '/' MUST start with 'eval_' (LeRobot "
        "rule when a policy is set). Default local/eval_so-dexarm.",
    )
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument("--episode-time-s", type=int, default=30)
    p.add_argument("--reset-time-s", type=int, default=10)
    p.add_argument(
        "--task",
        default="eval",
        help="Language instruction fed to the policy. IGNORED by ACT (not language-"
        "conditioned) but REQUIRED-CORRECT for SmolVLA: pass the SAME string the "
        "dataset was trained with, e.g. 'Place the fruit on a plate.'",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Must match the camera fps (RealSense default 30) and the training data fps.",
    )
    p.add_argument(
        "--cameras",
        default=DEFAULT_CAMERAS,
        help="Installation-specific camera dictionary. It must match the camera keys used during "
        "training. Pass 'none' only for a state-only policy.",
    )
    p.add_argument(
        "--display", action="store_true", help="Show the Rerun viewer (camera feed + state) during eval."
    )
    p.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="Cap on the per-step arm goal delta, passed to --robot.max_relative_target. "
        "2.0 = smooth start + smooth inference stalls; 20.0 = same clamp as record. "
        "Omit = NO cap (the arm lunges to the start pose at full speed).",
    )
    p.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Diffusion policy ONLY: denoise steps at inference. 10 recommended (the "
        "stored default of 100 takes ~1.6 s per chunk → the robot freezes). "
        "SmolVLA/ACT reject this flag.",
    )
    p.add_argument(
        "--noise-scheduler",
        choices=("DDPM", "DDIM"),
        default=None,
        help="Diffusion policy ONLY: scheduler override; DDIM recommended with few "
        "inference steps. SmolVLA/ACT reject this flag.",
    )
    p.add_argument("--print-only", action="store_true")
    p.add_argument(
        "extra",
        nargs="*",
        default=[],
        help="Extra args appended verbatim to lerobot_record; put them after '--', "
        "e.g. -- --policy.device=cuda",
    )

    # -- map-hand (diagnostic) --
    p = sub.add_parser("map-hand", help="Print computed hand mapping from calibration file.")
    _add_common_args(p)
    p.add_argument("--side", choices=("right", "left", "both"), default="both")

    return parser


def run(args: argparse.Namespace) -> int:
    cmd = args.command
    if cmd == "setup-motors":
        return cmd_setup_motors(args)
    if cmd == "calibrate-arm":
        return cmd_calibrate_arm(args)
    if cmd == "calibrate-hand":
        return cmd_calibrate_hand(args)
    if cmd == "capture-startup":
        return cmd_capture_startup(args)
    if cmd == "teleop":
        return cmd_teleop(args)
    if cmd == "record":
        return cmd_record(args)
    if cmd == "train":
        return cmd_train(args)
    if cmd == "eval":
        return cmd_eval(args)
    if cmd == "map-hand":
        return cmd_map_hand(args)
    print(f"Unknown command: {cmd}", flush=True)
    return 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
