from enum import Enum


# Player
class Player(Enum):
    black = 1
    white = 2

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white


# Directions
max_xy = 9


class Direction(Enum):
    WEST = -1
    NORTHWEST = max_xy*(-1)-1
    NORTHEAST = max_xy*(-1)
    EAST = 1
    SOUTHEAST = max_xy+1
    SOUTHWEST = max_xy

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


# def sub(a, b):
#     return a - b
#
#
# def add(a, b):
#     return a + b
#
#
# def mul(t, a):
#     return t * a
