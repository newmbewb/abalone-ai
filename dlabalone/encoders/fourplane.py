import numpy as np

from dlabalone.ablboard import Board
from dlabalone.abltypes import Player
from dlabalone.encoders.base import Encoder


class FourPlaneEncoder(Encoder):
    def __init__(self, board_size, mode=None, data_format="channels_last"):
        super().__init__(board_size, mode)
        self.num_planes = 4
        self.valid_map = np.zeros((self.max_xy, self.max_xy))
        self.invalid_map = np.ones((self.max_xy, self.max_xy))
        self.data_format = data_format
        for index in Board.valid_grids:
            x, y = Board.coord_index2xy(index)
            self.valid_map[y, x] = 1
            self.valid_map[y, x] = 0
        assert data_format == "channels_first" or data_format == "channels_last", \
            "data_format should be channels_first or channels_last"

    def name(self):
        return f'{type(self).__name__}_{self.data_format}'

    def encode_board(self, board, next_player=Player.black, **kwargs):
        map_next_player = np.zeros((self.max_xy, self.max_xy))
        map_opp_player = np.zeros((self.max_xy, self.max_xy))
        for point, player in board.grid.items():
            x, y = Board.coord_index2xy(point)
            if player == next_player:
                map_next_player[y, x] = 1
            else:
                map_opp_player[y, x] = 1
        plane_list = [map_next_player, map_opp_player, self.valid_map, self.invalid_map]
        if self.data_format == 'channels_first':
            return np.array(plane_list)
        elif self.data_format == 'channels_last':
            return np.dstack(plane_list)
        else:
            return None

    def shape(self):
        if self.data_format == 'channels_first':
            return self.num_planes, self.max_xy, self.max_xy
        elif self.data_format == 'channels_last':
            return self.max_xy, self.max_xy, self.num_planes
        else:
            return None


def create(board_size, mode, **kwargs):
    return FourPlaneEncoder(board_size, mode, **kwargs)
