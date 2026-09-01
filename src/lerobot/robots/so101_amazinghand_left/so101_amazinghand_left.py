from lerobot.robots.so101_amazinghand_right.so101_amazinghand_right import SO101AmazingHandRight

from .config_so101_amazinghand_left import SO101AmazingHandLeftConfig


class SO101AmazingHandLeft(SO101AmazingHandRight):
    """Left-side SO-101 + AmazingHand preset with `l_` action and observation keys."""

    config_class = SO101AmazingHandLeftConfig
    name = "so101_amazinghand_left"
