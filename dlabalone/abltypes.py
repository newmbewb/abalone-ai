from enum import Enum


# Player
class Player(Enum):
    black = 1
    white = 2

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white


# Directions
class Direction(Enum):
    WEST = (-1, 0)
    NORTHWEST = (-1, -1)
    NORTHEAST = (0, -1)
    EAST = (1, 0)
    SOUTHEAST = (1, 1)
    SOUTHWEST = (0, 1)

    @staticmethod
    def is_valid(direction):
        for d in Direction:
            if direction == d.value:
                return True
        return False

    @staticmethod
    def to_int(d1):
        if isinstance(d1, Direction):
            d1 = d1.value
        i = 0
        for d2 in Direction:
            if d1 == d2.value:
                return i
            i += 1
        assert False, "Invalid direction!"

    @staticmethod
    def from_int(arg):
        i = 0
        for d in Direction:
            if i == arg:
                return d.value
            i += 1
        assert False, "Invalid direction (int type)!"


def sub(a, b):
    x_diff = a[0] - b[0]
    y_diff = a[1] - b[1]
    return x_diff, y_diff


def add(a, b):
    x_diff = a[0] + b[0]
    y_diff = a[1] + b[1]
    return x_diff, y_diff


def mul(t, a):
    return t[0] * a, t[1] * a
