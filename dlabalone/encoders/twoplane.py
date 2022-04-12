import numpy as np

from dlabalone.encoders.base import Encoder


class TwoPlaneEncoder(Encoder):
    def __init__(self, board_size):
        self.max_xy = board_size * 2 - 1
        self.num_planes = 2

    def name(self):
        return 'twoplane'

    def encode(self, game_state):
        board_matrix = np.zeros(self.shape())
        next_player = game_state.next_player
        for point, player in game_state.board.grid.items():
            x, y = point
            if player == next_player:
                board_matrix[0, y, x] = 1
            else:
                board_matrix[1, y, x] = 1
        return board_matrix

    def encode_point(self, point):
        x, y = point
        return self.max_xy * y + x

    def decode_point_index(self, index):
        y = index // self.max_xy
        x = index % self.max_xy
        return x, y

    def num_points(self):
        return self.max_xy * self.max_xy

    def shape(self):
        return self.num_planes, self.max_xy, self.max_xy


def create(board_size):
    return TwoPlaneEncoder(board_size)
