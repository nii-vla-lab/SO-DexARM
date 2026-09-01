from .config_so101_amazinghand import SO101AmazingHandFollowerConfig

__all__ = ["SO101AmazingHandFollower", "SO101AmazingHandFollowerConfig"]


def __getattr__(name: str):
    if name == "SO101AmazingHandFollower":
        from .so101_amazinghand_follower import SO101AmazingHandFollower

        return SO101AmazingHandFollower
    raise AttributeError(name)
