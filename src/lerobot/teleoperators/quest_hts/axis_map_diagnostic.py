#!/usr/bin/env python
"""Phase 33 — Quest wrist AXIS-MAP diagnostic (no robot hardware).

Purpose
-------
We need to know, FOR EACH HAND SEPARATELY, which Quest wrist axis (x/y/z) and sign
corresponds to a real-world "up", "operator-right" and "forward (away from you)" motion.
The teleoperator currently assumes ONE global Quest→robot rotation for both hands
(_R_QUEST_TO_ROBOT). The symptom "left hand up raises the arm, right hand up makes the
RIGHT arm reach FORWARD instead" means the Quest is streaming the two hands in DIFFERENT
frames — so we must measure each side and build a PER-SIDE rotation.

How it works
------------
Binds the same TCP port the teleop uses (8000) and reads the HTS "Right/Left wrist" lines.
It does NOT touch the robot. For each hand it keeps a short rolling window and reports the
wrist DISPLACEMENT over the last ~0.6 s, flagging the dominant axis + sign. Hold a hand
still → deltas ≈ 0. Move it slowly ~20-30 cm in ONE direction → the dominant axis lights up.

Procedure (stop the teleop first — this binds port 8000)
--------------------------------------------------------
1. Start your Quest Hand Tracking Streamer as usual (so it sends to this PC:8000).
   If wired: adb reverse tcp:8000 tcp:8000
2. Run:  python -m lerobot.teleoperators.quest_hts.axis_map_diagnostic
3. For the RIGHT hand only, do each move slowly and read the "DOMINANT" line:
     - move hand UP                      → note axis+sign  (e.g. "+y")
     - move hand to YOUR RIGHT           → note axis+sign
     - move hand FORWARD (away from you) → note axis+sign
4. Repeat for the LEFT hand.
5. Paste the 6 results back. From those I build the correct per-side mapping.
"""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import threading
import time
from collections import deque

from lerobot.teleoperators.quest_hts.hts_protocol import parse_hts_line

LOG = logging.getLogger("axis_map_diagnostic")

WINDOW_S = 0.6  # displacement is measured over this trailing window
MOVE_THRESH_M = 0.02  # below this total displacement we treat the hand as "still"
AXIS_NAMES = ("x", "y", "z")


class SideTracker:
    def __init__(self, side: str) -> None:
        self.side = side
        self.samples: deque[tuple[float, float, float, float]] = deque()  # (t, x, y, z)

    def add(self, t: float, x: float, y: float, z: float) -> None:
        self.samples.append((t, x, y, z))
        cutoff = t - WINDOW_S * 4
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def displacement(self, now: float) -> tuple[float, float, float] | None:
        """Current position minus the oldest sample within WINDOW_S."""
        if not self.samples:
            return None
        cur = self.samples[-1]
        old = None
        for s in self.samples:
            if now - s[0] <= WINDOW_S:
                old = s
                break
        if old is None:
            old = self.samples[0]
        return (cur[1] - old[1], cur[2] - old[2], cur[3] - old[3])

    def avg_recent(self, now: float, win: float = 0.5) -> tuple[float, float, float] | None:
        """Mean wrist position over the last `win` seconds (denoise for capture)."""
        pts = [(x, y, z) for (t, x, y, z) in self.samples if now - t <= win]
        if not pts:
            return None
        n = len(pts)
        return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n, sum(p[2] for p in pts) / n)


def _dominant(d: tuple[float, float, float]) -> tuple[int, int, str]:
    """Return (axis_index, sign, 'name') of the largest-magnitude component."""
    i = max(range(3), key=lambda k: abs(d[k]))
    sign = 1 if d[i] >= 0 else -1
    return i, sign, f"{'+' if sign > 0 else '-'}{AXIS_NAMES[i]}"


def _fmt(side: str, d: tuple[float, float, float] | None) -> str:
    if d is None:
        return f"  {side:5s}: (no data)"
    dx, dy, dz = d
    mag = (dx * dx + dy * dy + dz * dz) ** 0.5
    if mag < MOVE_THRESH_M:
        return f"  {side:5s}: still         Δ=({dx:+.3f},{dy:+.3f},{dz:+.3f})"
    # dominant axis
    comps = (dx, dy, dz)
    i = max(range(3), key=lambda k: abs(comps[k]))
    sign = "+" if comps[i] >= 0 else "-"
    dom = f"{sign}{AXIS_NAMES[i]}"
    return f"  {side:5s}: DOMINANT {dom}   Δ=({dx:+.3f},{dy:+.3f},{dz:+.3f})  |Δ|={mag:.3f}m"


class Shared:
    """Receiver state shared between the accept loop, reader threads, and the main mode."""

    def __init__(self) -> None:
        self.trackers = {"right": SideTracker("right"), "left": SideTracker("left")}
        self.counts = {"total": 0, "wrist": 0, "landmarks": 0, "none": 0}
        self.last_raw: deque[str] = deque(maxlen=3)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.conn_count = 0


def _serve_accept_loop(host: str, port: int, sh: Shared) -> None:
    """Bind + accept; one reader thread per connection (silent/stale conns can't block new ones)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(8)
    server.settimeout(0.5)
    LOG.info("Listening on %s:%d (wired Quest: adb reverse tcp:%d tcp:%d)", host, port, port, port)

    def reader(conn: socket.socket, addr) -> None:
        with conn:
            conn.settimeout(0.5)
            buf = ""
            while not sh.stop.is_set():
                try:
                    data = conn.recv(8192)
                except TimeoutError:
                    continue
                except OSError as exc:
                    LOG.warning("conn %s error: %s", addr, exc)
                    break
                if not data:
                    LOG.info("HTS %s disconnected", addr)
                    break
                buf += data.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    parsed = parse_hts_line(line)
                    with sh.lock:
                        sh.counts["total"] += 1
                        sh.last_raw.append(line)
                        if parsed is None:
                            sh.counts["none"] += 1
                            continue
                        side, kind, values = parsed
                        sh.counts[kind] = sh.counts.get(kind, 0) + 1
                        if kind == "wrist" and side in sh.trackers and len(values) >= 3:
                            sh.trackers[side].add(time.monotonic(), values[0], values[1], values[2])

    try:
        while not sh.stop.is_set():
            try:
                conn, addr = server.accept()
            except TimeoutError:
                continue
            with sh.lock:
                sh.conn_count += 1
                n = sh.conn_count
            LOG.info("HTS connected from %s (conn #%d)", addr, n)
            threading.Thread(target=reader, args=(conn, addr), daemon=True).start()
    finally:
        server.close()


def streaming_readout(sh: Shared, period_s: float) -> None:
    """Free-form live readout: rolling-window displacement + dominant axis."""
    LOG.info("Move ONE hand slowly in ONE direction; read its DOMINANT axis. (Ctrl+C to stop)")
    try:
        while not sh.stop.is_set():
            time.sleep(period_s)
            now = time.monotonic()
            with sh.lock:
                c = dict(sh.counts)
                raws = list(sh.last_raw)
                dr = sh.trackers["right"].displacement(now)
                dl = sh.trackers["left"].displacement(now)
            print(
                f"[{now:8.1f}] lines total={c['total']} wrist={c['wrist']} "
                f"landmarks={c['landmarks']} unparsed={c['none']}"
            )
            if c["wrist"] == 0:
                for raw in raws:
                    print(f"    raw: {raw[:90]}")
                if c["total"] == 0:
                    print(
                        "    >>> connection made but 0 bytes — restart the Quest streamer "
                        "(toggle streaming OFF/ON, or re-enter host:port)."
                    )
                else:
                    print("    >>> data arriving but NO 'wrist' lines — paste the raw line(s) above.")
            print(_fmt("right", dr))
            print(_fmt("left", dl))
    except KeyboardInterrupt:
        sh.stop.set()


# Guided capture: (hand, motion, robot-axis it should map to)
_CAPTURE_PLAN = [
    ("right", "UP (straight up)", "up"),
    ("right", "to YOUR RIGHT", "right"),
    ("right", "FORWARD (away from you, toward the robot)", "forward"),
    ("left", "UP (straight up)", "up"),
    ("left", "to YOUR LEFT", "left"),
    ("left", "FORWARD (away from you, toward the robot)", "forward"),
]


def _capture_avg(sh: Shared, side: str, win: float = 0.5) -> tuple[float, float, float] | None:
    with sh.lock:
        return sh.trackers[side].avg_recent(time.monotonic(), win)


def capture_wizard(sh: Shared) -> None:
    """Guided, ENTER-gated capture of each hand's up/right(left)/forward axis."""
    print("\n=== Phase 33 GUIDED axis capture ===")
    print("For each prompt: move ONLY the named hand in the named direction ~25 cm, HOLD it there,")
    print("then press ENTER. Keep the OTHER hand still. Press Ctrl+C anytime to abort.\n")

    # wait until both hands stream
    print("Waiting for wrist data from both hands...", flush=True)
    while not sh.stop.is_set():
        if _capture_avg(sh, "right") is not None and _capture_avg(sh, "left") is not None:
            break
        time.sleep(0.2)
    print("Both hands detected.\n")

    results: dict[tuple[str, str], tuple[float, float, float]] = {}
    try:
        for side in ("right", "left"):
            input(f"[{side.upper()}] Put the {side} hand at a comfortable CENTER, hold still, press ENTER...")
            base = _capture_avg(sh, side)
            while base is None:
                input("  (no data yet — hold still and press ENTER again)")
                base = _capture_avg(sh, side)
            for s, motion, robot_axis in _CAPTURE_PLAN:
                if s != side:
                    continue
                input(f"  [{side.upper()}] Move the {side} hand {motion}, HOLD, press ENTER...")
                end = _capture_avg(sh, side)
                if end is None:
                    print("    (no data — skipped)")
                    continue
                delta = (end[0] - base[0], end[1] - base[1], end[2] - base[2])
                _, _, dom = _dominant(delta)
                mag = (delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2) ** 0.5
                print(
                    f"    → Δ=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f})  "
                    f"dominant {dom}  |Δ|={mag:.3f}m" + ("   ⚠ very small move, redo" if mag < 0.05 else "")
                )
                results[(side, robot_axis)] = delta
                print(f"    (return the {side} hand to CENTER)")
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        sh.stop.set()
        return

    _report_capture(results)
    sh.stop.set()


def _report_capture(results: dict[tuple[str, str], tuple[float, float, float]]) -> None:
    print("\n================ MEASURED AXIS MAP ================")
    print(f"{'hand':5s} {'motion':9s} {'Δx':>8s} {'Δy':>8s} {'Δz':>8s}  dominant")
    order = {"right": ["up", "right", "forward"], "left": ["up", "left", "forward"]}
    for side in ("right", "left"):
        for axis in order[side]:
            d = results.get((side, axis))
            if d is None:
                print(f"{side:5s} {axis:9s}  (missing)")
                continue
            _, _, dom = _dominant(d)
            print(f"{side:5s} {axis:9s} {d[0]:+8.3f} {d[1]:+8.3f} {d[2]:+8.3f}  {dom}")

    # Build a per-side quest→robot rotation so that:
    #   robot X(forward) ← the hand's FORWARD motion
    #   robot Y(lateral) ← the hand's RIGHT(right hand)/LEFT(left hand) motion  [operator-outward]
    #   robot Z(up)      ← the hand's UP motion
    for side in ("right", "left"):
        lateral_key = "right" if side == "right" else "left"
        need = {
            "forward": results.get((side, "forward")),
            "lateral": results.get((side, lateral_key)),
            "up": results.get((side, "up")),
        }
        if any(v is None for v in need.values()):
            print(f"\n[{side}] incomplete — cannot derive matrix.")
            continue
        rows = {}
        used = {}
        ok = True
        for robot_axis, d in (("X", need["forward"]), ("Y", need["lateral"]), ("Z", need["up"])):
            i, sign, _ = _dominant(d)
            if i in used:
                ok = False
            used[i] = robot_axis
            row = [0, 0, 0]
            row[i] = sign
            rows[robot_axis] = row
        print(f"\n[{side}] suggested _R_QUEST_TO_ROBOT_{side.upper()} (robot = R @ quest):")
        print(f"    X(forward) = {rows['X']}")
        print(f"    Y(lateral) = {rows['Y']}")
        print(f"    Z(up)      = {rows['Z']}")
        if not ok:
            print("    ⚠ two motions shared a dominant axis — moves were sloppy; please redo cleanly.")
    print("\n>>> Paste this whole MEASURED AXIS MAP block back to me; I'll wire the matrices in.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--print-period-s", type=float, default=0.3)
    p.add_argument(
        "--capture",
        action="store_true",
        help="Guided ENTER-gated capture of the per-hand axis map (recommended).",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sh = Shared()
    signal.signal(signal.SIGTERM, lambda *_: sh.stop.set())
    LOG.info("Phase 33 axis-map diagnostic — NO robot hardware is touched.")
    threading.Thread(target=_serve_accept_loop, args=(args.host, args.port, sh), daemon=True).start()
    try:
        if args.capture:
            capture_wizard(sh)
        else:
            streaming_readout(sh, max(args.print_period_s, 0.05))
    except KeyboardInterrupt:
        sh.stop.set()
    LOG.info("stopped")


if __name__ == "__main__":
    main()
