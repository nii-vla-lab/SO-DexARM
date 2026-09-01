# SO-DexARM CAD

This directory contains the printable CAD for the two mechanical systems used by SO-DexARM:

- `AmazingHand/`: the complete upstream AmazingHand CAD set (STEP and STL).
- `SO-ARM101/`: SO-ARM101 arm-body parts only (STEP and STL).
- `Wrist_Roll_Pitch_SO101.step`: the project-specific wrist-roll/pitch part supplied for SO-DexARM. This file replaces the upstream part with the same name.

## Upstream sources

| Component | Repository | Revision |
| --- | --- | --- |
| AmazingHand | <https://github.com/pollen-robotics/AmazingHand> | `3e8241074df3436a3044ced4881e3bb2133aa725` |
| SO-ARM101 | <https://github.com/TheRobotStudio/SO-ARM100> | `7629d2ad9853d10fb903093a33ef6114099d97e5` |

Both upstream projects are distributed under the Apache License 2.0. A copy of each upstream license is kept in its component directory.

## SO-ARM101 exclusions

SO-DexARM uses AmazingHand instead of the standard SO-ARM101 gripper. The following upstream parts are intentionally not included:

- `Moving_Jaw_SO101`
- `Wrist_Roll_Follower_SO101`
- `SO101 Assembly` (contains the standard gripper)
- follower print-bed assemblies (contain the standard gripper)

Leader-only handle, trigger, wrist-roll, and print-bed parts are also omitted because SO-DexARM uses Meta Quest control instead of a physical SO-ARM101 leader.

The upstream `Wrist_Roll_Pitch_SO101` STEP and STL are omitted. Use the project-specific `Wrist_Roll_Pitch_SO101.step` at the root of this directory. A matching STL has not been generated from that custom STEP file, so no potentially incompatible STL is provided.
