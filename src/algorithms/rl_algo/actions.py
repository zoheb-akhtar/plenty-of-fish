from enum import IntEnum

class Action(IntEnum):
    """Action enumeration for the shark."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ATTACK = 4
    REST = 5
