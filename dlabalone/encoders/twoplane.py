import numpy as np

from dlabalone.abltypes import Player
from dlabalone.encoders.base import Encoder


class TwoPlaneEncoder(Encoder):
    def __init__(self, board_size):
        super().__init__(board_size)
        self.num_planes = 2

    def name(self):
        return type(self).__name__

    def encode_board(self, board, next_player=Player.black):
        board_matrix = np.zeros(self.shape())
        for point, player in board.grid.items():
            x, y = point
            if player == next_player:
                board_matrix[0, y, x] = 1
            else:
                board_matrix[1, y, x] = 1
        return board_matrix

    def shape(self):
        return self.num_planes, self.max_xy, self.max_xy


def create(board_size):
    return TwoPlaneEncoder(board_size)
