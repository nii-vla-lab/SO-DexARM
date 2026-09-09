"""SO-DexARM-aware wrapper for LeRobot calibration."""

from lerobot.scripts.lerobot_calibrate import main

import sodexarm  # noqa: F401 - registers external configs before CLI parsing

if __name__ == "__main__":
    main()
