"""Real-robot policy evaluation alias for the SO-DexARM CLI."""

import sys

import sodexarm  # noqa: F401 - registers external configs before CLI parsing
from sodexarm.cli.main import main as cli_main


def main() -> None:
    # SO-DexARM real-hardware evaluation is LeRobot record with a policy, not
    # the generic environment-evaluation command.
    sys.argv.insert(1, "eval")
    cli_main()

if __name__ == "__main__":
    main()
