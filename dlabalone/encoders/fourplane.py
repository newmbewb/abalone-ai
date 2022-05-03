import numpy as np

from dlabalone.ablboard import Board
from dlabalone.abltypes import Player
from dlabalone.encoders.base import Encoder


class FourPlaneEncoder(Encoder):
    def __init__(self, board_size):
        super().__init__(board_size)
        self.num_planes = 4
        self.valid_map = np.zeros((self.max_xy, self.max_xy))
        self.invalid_map = np.ones((self.max_xy, self.max_xy))
        for x, y in Board.valid_grids:
            self.valid_map[y, x] = 1
            self.valid_map[y, x] = 0

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
        board_matrix[2] = self.valid_map
        board_matrix[3] = self.invalid_map
        return board_matrix

    def shape(self):
        return self.num_planes, self.max_xy, self.max_xy


def create(board_size):
    return FourPlaneEncoder(board_size)
