"""Non-invasive adapters for the LeRobot v0.4.4 control loops."""

from __future__ import annotations

import inspect
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator


def _call_optional(obj: object | None, name: str, *args, **kwargs):
    if obj is None:
        return None
    method = getattr(obj, name, None)
    if method is None:
        return None
    try:
        return method(*args, **kwargs)
    except NotImplementedError:
        return None


def _begin_episode(teleop: object | None, robot: object, events: dict[str, Any] | None) -> None:
    """Call either the current or legacy SO-DexARM episode hook signature."""
    if teleop is None:
        return
    method = getattr(teleop, "begin_episode", None)
    if method is None:
        return
    parameters = inspect.signature(method).parameters.values()
    supports_events = "events" in inspect.signature(method).parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    kwargs = {"events": events} if supports_events else {}
    try:
        method(robot, **kwargs)
    except NotImplementedError:
        pass


def prepare_control_session(teleop: object | None, robot: object) -> None:
    """Run the SO-DexARM teleoperator/robot handshake exactly once."""
    if teleop is None or getattr(teleop, "_sodexarm_session_prepared", False):
        return
    _call_optional(teleop, "prepare_control_session", robot)
    metadata = _call_optional(teleop, "get_episode_metadata")
    if metadata:
        logging.info("Teleoperator metadata: %s", metadata)
    setattr(teleop, "_sodexarm_session_prepared", True)


@contextmanager
def adapt_control_io(
    teleop: object | None,
    robot: object,
    *,
    fps: int,
    events: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Add SO-DexARM feedback/readiness behavior around an upstream loop.

    The upstream loop itself remains LeRobot code.  Only the two device methods
    it calls are temporarily decorated, then restored in ``finally``.
    """
    if teleop is None:
        yield
        return

    original_observation = robot.get_observation
    original_action = teleop.get_action
    latest_observation: dict[str, Any] = {}

    def stopped() -> bool:
        return bool(events and (events.get("exit_early") or events.get("stop_recording")))

    def feedback_observation():
        observation = original_observation()
        latest_observation.clear()
        latest_observation.update(observation)
        _call_optional(teleop, "send_feedback", observation)

        ready = getattr(teleop, "is_control_ready", None)
        while callable(ready) and not ready() and not stopped():
            time.sleep(max(1.0 / fps, 0.001))
            observation = original_observation()
            latest_observation.clear()
            latest_observation.update(observation)
            _call_optional(teleop, "send_feedback", observation)
        return observation

    def complete_action():
        action = original_action()
        # Hand-only operation still needs a complete LeRobot action vector.
        for key in robot.action_features:
            if key not in action and key in latest_observation:
                action[key] = latest_observation[key]
        return action

    robot.get_observation = feedback_observation
    teleop.get_action = complete_action
    try:
        yield
    finally:
        robot.get_observation = original_observation
        teleop.get_action = original_action


def _json_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}/{key}" if prefix else str(key), child, output)
    else:
        output[prefix] = _json_scalar(value)


def collect_episode_metadata(robot: object, teleop: object | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for device in (teleop, robot):
        value = _call_optional(device, "get_episode_metadata")
        if value:
            _flatten("", value, metadata)
    return metadata


def install_metadata_sidecar(dataset: object, robot: object, teleop: object | None) -> None:
    """Persist custom metadata without changing LeRobotDataset's schema or source."""
    if getattr(dataset, "_sodexarm_metadata_installed", False):
        return

    original_save: Callable = dataset.save_episode

    def save_episode(*args, **kwargs):
        episode_index = int(dataset.num_episodes)
        metadata = collect_episode_metadata(robot, teleop)
        result = original_save(*args, **kwargs)
        if metadata:
            path = Path(dataset.root) / "meta" / "sodexarm_episode_metadata.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"episode_index": episode_index, **metadata}) + "\n")
        return result

    dataset.save_episode = save_episode
    setattr(dataset, "_sodexarm_metadata_installed", True)


def episode_aborted(robot: object, teleop: object | None) -> bool:
    teleop_abort = bool(_call_optional(teleop, "should_abort_episode"))
    robot_abort = bool(_call_optional(robot, "should_abort_episode"))
    if teleop_abort or robot_abort:
        logging.warning(
            "Discarding invalid SO-DexARM episode: %s",
            collect_episode_metadata(robot, teleop),
        )
        _call_optional(teleop, "clear_episode_abort_state")
        return True
    return False


def make_teleop_loop_adapter(upstream_loop: Callable) -> Callable:
    def loop(*args, **kwargs):
        teleop = kwargs.get("teleop", args[0] if args else None)
        robot = kwargs.get("robot", args[1] if len(args) > 1 else None)
        fps = int(kwargs.get("fps", args[2] if len(args) > 2 else 60))
        prepare_control_session(teleop, robot)
        with adapt_control_io(teleop, robot, fps=fps):
            return upstream_loop(*args, **kwargs)

    return loop


def make_record_loop_adapter(upstream_loop: Callable) -> Callable:
    def loop(*args, **kwargs):
        robot = kwargs.get("robot", args[0] if args else None)
        events = kwargs.get("events", args[1] if len(args) > 1 else None)
        fps = int(kwargs.get("fps", args[2] if len(args) > 2 else 30))
        teleop = kwargs.get("teleop")
        dataset = kwargs.get("dataset")

        # An invalid episode is re-recorded by LeRobot. Skip its reset phase so
        # faulty hardware is not driven for reset_time_s before the retry.
        if dataset is None and events and events.pop("_sodexarm_abort_pending", False):
            return None

        prepare_control_session(teleop, robot)
        if dataset is not None:
            install_metadata_sidecar(dataset, robot, teleop)
            _begin_episode(teleop, robot, events)

        with adapt_control_io(teleop, robot, fps=fps, events=events):
            result = upstream_loop(*args, **kwargs)

        if dataset is not None:
            _call_optional(teleop, "end_episode", robot)
            if episode_aborted(robot, teleop):
                events["rerecord_episode"] = True
                events["_sodexarm_abort_pending"] = True
        return result

    return loop
