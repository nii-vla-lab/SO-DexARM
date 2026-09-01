from .config_so101_amazinghand_left import SO101AmazingHandLeftConfig

__all__ = ["SO101AmazingHandLeft", "SO101AmazingHandLeftConfig"]


def __getattr__(name: str):
    if name == "SO101AmazingHandLeft":
        from .so101_amazinghand_left import SO101AmazingHandLeft

        return SO101AmazingHandLeft
    raise AttributeError(name)
