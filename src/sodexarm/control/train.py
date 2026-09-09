"""SO-DexARM-aware wrapper for LeRobot training."""

from lerobot.scripts.lerobot_train import main

import sodexarm  # noqa: F401 - keep registration behavior consistent across commands

if __name__ == "__main__":
    main()
