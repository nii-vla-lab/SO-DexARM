from .config_so101_amazinghand_right import SO101AmazingHandRightConfig

__all__ = ["SO101AmazingHandRight", "SO101AmazingHandRightConfig"]


def __getattr__(name: str):
    if name == "SO101AmazingHandRight":
        from .so101_amazinghand_right import SO101AmazingHandRight

        return SO101AmazingHandRight
    raise AttributeError(name)
