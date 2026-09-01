#!/usr/bin/env python

"""Phase 0 TCP receiver for Hand Tracking Streamer (HTS).

This script does not control any robot hardware. It accepts a TCP connection
from HTS, parses wrist and landmark CSV lines, and prints a compact summary of
the latest right/left hand data.

HTS line format:
    Right wrist:, x, y, z, qx, qy, qz, qw
    Right landmarks:, 63 floats = 21 * (x, y, z)
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import socket
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

WRIST_VALUE_COUNT = 7
LANDMARK_VALUE_COUNT = 63
LANDMARK_COUNT = 21


@dataclass
class HandSnapshot:
    side: str
    wrist: tuple[float, ...] | None = None
    landmarks: tuple[tuple[float, float, float], ...] | None = None
    wrist_time_s: float = 0.0
    landmarks_time_s: float = 0.0
    packets: int = 0
    parse_errors: int = 0

    def update_wrist(self, values: tuple[float, ...]) -> None:
        self.wrist = values
        self.wrist_time_s = time.monotonic()
        self.packets += 1

    def update_landmarks(self, values: tuple[float, ...]) -> None:
        points = []
        for idx in range(0, LANDMARK_VALUE_COUNT, 3):
            points.append((values[idx], values[idx + 1], values[idx + 2]))
        self.landmarks = tuple(points)
        self.landmarks_time_s = time.monotonic()
        self.packets += 1


@dataclass
class HTSState:
    hands: dict[str, HandSnapshot] = field(
        default_factory=lambda: {
            "right": HandSnapshot("right"),
            "left": HandSnapshot("left"),
        }
    )
    malformed_lines: int = 0
    total_lines: int = 0


def _parse_floats(parts: Iterable[str]) -> tuple[float, ...] | None:
    values: list[float] = []
    for part in parts:
        text = part.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    return tuple(values)


def parse_hts_line(line: str) -> tuple[str, str, tuple[float, ...]] | None:
    parts = line.strip().split(",")
    if not parts:
        return None

    label = parts[0].strip().lower()
    if "right" in label:
        side = "right"
    elif "left" in label:
        side = "left"
    else:
        return None

    if "wrist" in label:
        kind = "wrist"
        expected = WRIST_VALUE_COUNT
    elif "landmarks" in label:
        kind = "landmarks"
        expected = LANDMARK_VALUE_COUNT
    else:
        return None

    values = _parse_floats(parts[1:])
    if values is None or len(values) != expected:
        return None

    return side, kind, values


def _format_wrist(wrist: tuple[float, ...] | None) -> str:
    if wrist is None:
        return "wrist=None"
    x, y, z, qx, qy, qz, qw = wrist
    return f"wrist=pos({x:+.4f}, {y:+.4f}, {z:+.4f}) quat({qx:+.4f}, {qy:+.4f}, {qz:+.4f}, {qw:+.4f})"


def _format_landmarks(landmarks: tuple[tuple[float, float, float], ...] | None) -> str:
    if landmarks is None:
        return "landmarks=None"
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    little_tip = landmarks[20]
    return (
        f"landmarks={len(landmarks)} "
        f"wrist({wrist[0]:+.4f}, {wrist[1]:+.4f}, {wrist[2]:+.4f}) "
        f"tips thumb({thumb_tip[0]:+.4f}, {thumb_tip[1]:+.4f}, {thumb_tip[2]:+.4f}) "
        f"index({index_tip[0]:+.4f}, {index_tip[1]:+.4f}, {index_tip[2]:+.4f}) "
        f"middle({middle_tip[0]:+.4f}, {middle_tip[1]:+.4f}, {middle_tip[2]:+.4f}) "
        f"ring({ring_tip[0]:+.4f}, {ring_tip[1]:+.4f}, {ring_tip[2]:+.4f}) "
        f"little({little_tip[0]:+.4f}, {little_tip[1]:+.4f}, {little_tip[2]:+.4f})"
    )


def print_state(state: HTSState) -> None:
    now = time.monotonic()
    for side in ("right", "left"):
        hand = state.hands[side]
        wrist_age = now - hand.wrist_time_s if hand.wrist_time_s else None
        landmarks_age = now - hand.landmarks_time_s if hand.landmarks_time_s else None
        wrist_age_text = "None" if wrist_age is None else f"{wrist_age:.3f}s"
        landmarks_age_text = "None" if landmarks_age is None else f"{landmarks_age:.3f}s"
        logging.info(
            "%s | %s | %s | age wrist=%s landmarks=%s packets=%d errors=%d",
            side.upper(),
            _format_wrist(hand.wrist),
            _format_landmarks(hand.landmarks),
            wrist_age_text,
            landmarks_age_text,
            hand.packets,
            hand.parse_errors,
        )
    logging.info("lines=%d malformed=%d", state.total_lines, state.malformed_lines)


def handle_line(line: str, state: HTSState, print_raw: bool) -> None:
    state.total_lines += 1
    if print_raw:
        logging.info("raw: %s", line)

    parsed = parse_hts_line(line)
    if parsed is None:
        state.malformed_lines += 1
        return

    side, kind, values = parsed
    hand = state.hands[side]
    if kind == "wrist":
        hand.update_wrist(values)
    elif kind == "landmarks":
        hand.update_landmarks(values)


def serve_tcp(host: str, port: int, print_period_s: float, print_raw: bool) -> None:
    state = HTSState()
    running = True

    def _stop(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    server.settimeout(0.5)

    logging.info("Phase 0 dry-run only: no robot hardware will be controlled.")
    logging.info("TCP server listening on %s:%d", host, port)
    logging.info("For wired Quest TCP, run: adb reverse tcp:%d tcp:%d", port, port)

    next_print = time.monotonic() + print_period_s
    try:
        while running:
            try:
                conn, addr = server.accept()
            except TimeoutError:
                if time.monotonic() >= next_print:
                    print_state(state)
                    next_print += print_period_s
                continue

            logging.info("Accepted HTS connection from %s", addr)
            with conn:
                conn.settimeout(0.5)
                buffer = ""
                while running:
                    data: bytes | None
                    try:
                        data = conn.recv(8192)
                    except TimeoutError:
                        data = None
                    except OSError as exc:
                        logging.warning("Connection error: %s", exc)
                        break

                    if data is None:
                        pass
                    elif data:
                        try:
                            buffer += data.decode("utf-8")
                        except UnicodeDecodeError:
                            state.malformed_lines += 1
                            continue

                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if line:
                                handle_line(line, state, print_raw=print_raw)
                    else:
                        break

                    now = time.monotonic()
                    if now >= next_print:
                        print_state(state)
                        next_print += print_period_s

            logging.info("HTS connection closed")
    finally:
        server.close()
        logging.info("TCP server stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 TCP print receiver for HTS.")
    parser.add_argument("--host", default="0.0.0.0", help="TCP bind host.")
    parser.add_argument("--port", type=int, default=8000, help="TCP bind port.")
    parser.add_argument(
        "--print-period-s",
        type=float,
        default=0.5,
        help="How often to print the latest parsed hand state.",
    )
    parser.add_argument("--print-raw", action="store_true", help="Print every raw HTS CSV line.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    serve_tcp(
        host=args.host,
        port=args.port,
        print_period_s=max(args.print_period_s, 0.05),
        print_raw=args.print_raw,
    )


if __name__ == "__main__":
    main()
