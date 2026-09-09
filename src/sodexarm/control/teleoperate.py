"""SO-DexARM entry point backed by LeRobot's teleoperation command."""

from lerobot.scripts import lerobot_teleoperate as upstream

import sodexarm  # noqa: F401 - registers external configs before CLI parsing

from ._adapters import make_teleop_loop_adapter


def main() -> None:
    if not getattr(upstream.teleop_loop, "_sodexarm_adapter", False):
        upstream.teleop_loop = make_teleop_loop_adapter(upstream.teleop_loop)
        upstream.teleop_loop._sodexarm_adapter = True
    upstream.main()


if __name__ == "__main__":
    main()
