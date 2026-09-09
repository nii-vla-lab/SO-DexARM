"""SO-DexARM entry point backed by LeRobot's recording command."""

from lerobot.scripts import lerobot_record as upstream

import sodexarm  # noqa: F401 - registers external configs before CLI parsing

from ._adapters import make_record_loop_adapter


def main() -> None:
    if not getattr(upstream.record_loop, "_sodexarm_adapter", False):
        upstream.record_loop = make_record_loop_adapter(upstream.record_loop)
        upstream.record_loop._sodexarm_adapter = True
    upstream.main()


if __name__ == "__main__":
    main()
