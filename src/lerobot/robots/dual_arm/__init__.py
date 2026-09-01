from .config_dual_arm import DualArmConfig

__all__ = ["DualArm", "DualArmConfig"]


def __getattr__(name: str):
    if name == "DualArm":
        from .dual_arm import DualArm

        return DualArm
    raise AttributeError(name)
