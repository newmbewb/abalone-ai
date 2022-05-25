import queue
from tensorflow.keras.utils import to_categorical

import numpy as np

from dlabalone.ablboard import Board, GameState
from dlabalone.abltypes import Player, Direction
from dlabalone.encoders.base import Encoder
from dlabalone.encoders.plane_generator import calc_opp_layers
from dlabalone.utils import load_file_board_move_pair


class AlphaAbaloneEncoder(Encoder):
    def __init__(self, board_size, mode, data_format="channels_last"):
        super().__init__(board_size, mode)
        self.num_planes = 45
        self.valid_map = np.zeros((self.max_xy, self.max_xy))
        self.invalid_map = np.ones((self.max_xy, self.max_xy))
        self.data_format = data_format
        for x, y in Board.valid_grids:
            self.valid_map[y, x] = 1
            self.valid_map[y, x] = 0
        assert data_format == "channels_first" or data_format == "channels_last", \
            "data_format should be channels_first or channels_last"

    def name(self):
        return f'{type(self).__name__}_{self.data_format}'

    def encode_board(self, game_board, next_player=Player.black, **kwargs):
        game = GameState(game_board, next_player)
        push_moves = []
        if 'kill_moves' in kwargs:
            kill_moves = kwargs['kill_moves']
        else:
            kill_moves, normal_moves = game.legal_moves(separate_kill=True, push_moves=push_moves)
        player_attack_plains = self._generate_player_attack_plains(kill_moves, push_moves)
        opp_attack_plains = self._generate_opp_attack_plains(game_board, next_player)
        basic_plains = self._generate_basic_plains(game_board, next_player)
        plane_list = player_attack_plains + opp_attack_plains + basic_plains
        if self.data_format == 'channels_first':
            return np.array(plane_list)
        elif self.data_format == 'channels_last':
            return np.dstack(plane_list)
        else:
            return None

    def _generate_basic_plains(self, board, next_player=Player.black):
        map_next_player = np.zeros((self.max_xy, self.max_xy))
        map_opp_player = np.zeros((self.max_xy, self.max_xy))
        for point, player in board.grid.items():
            x, y = point
            if player == next_player:
                map_next_player[y, x] = 1
            else:
                map_opp_player[y, x] = 1

        return [map_next_player, map_opp_player, self.valid_map, self.invalid_map]

    def _generate_player_attack_plains(self, kill_moves, push_moves):
        # Init plains
        plain_kill = {}
        plain_push = {}
        for length in [2, 3]:
            for direction in Direction:
                key = length, direction.value
                plain_kill[key] = np.zeros((self.max_xy, self.max_xy))
                plain_push[key] = np.zeros((self.max_xy, self.max_xy))

        # Fill plains
        self._move_list2np_arr(plain_kill, kill_moves)
        self._move_list2np_arr(plain_push, push_moves)

        # Make list
        plain_kill_list = []
        plain_push_list = []
        for length in [2, 3]:
            for direction in Direction:
                key = length, direction.value
                plain_kill_list.append(plain_kill[key])
                plain_push_list.append(plain_push[key])

        return plain_kill_list + plain_push_list

    def _generate_opp_attack_plains(self, board, next_player=Player.black):
        plain_opp_sumito_stones = np.zeros((self.max_xy, self.max_xy))
        plain_danger_stones = np.zeros((self.max_xy, self.max_xy))
        plain_danger_points = np.zeros((self.max_xy, self.max_xy))
        plain_vuln_stones = {2: np.zeros((self.max_xy, self.max_xy)), 3: np.zeros((self.max_xy, self.max_xy))}
        plain_vuln_points = {}
        for length in [2, 3]:
            for direction in Direction:
                key = length, direction.value
                plain_vuln_points[key] = np.zeros((self.max_xy, self.max_xy))

        # Fill array
        opp_sumito_stones, danger_stones, danger_points, vuln_stones, vuln_points = calc_opp_layers(board, next_player)
        self._stone_list2np_arr(plain_opp_sumito_stones, opp_sumito_stones)
        self._stone_list2np_arr(plain_danger_stones, danger_stones)
        self._stone_list2np_arr(plain_danger_points, danger_points)
        for length in [2, 3]:
            self._stone_list2np_arr(plain_vuln_stones[length], vuln_stones[length])
        for key, stone_list in vuln_points.items():
            self._stone_list2np_arr(plain_vuln_points[key], stone_list)

        # Make list
        plain_vuln_stones_list = []
        plain_vuln_points_list = []
        for length in [2, 3]:
            plain_vuln_stones_list.append(plain_vuln_stones[length])
            for direction in Direction:
                key = length, direction.value
                plain_vuln_points_list.append(plain_vuln_points[key])
        return [plain_opp_sumito_stones, plain_danger_stones, plain_danger_points] + plain_vuln_stones_list + \
               plain_vuln_points_list

    @staticmethod
    def _move_list2np_arr(np_arr, move_list):
        for move in move_list:
            length = len(move.stones)
            direction = move.direction
            key = length, direction
            for stone in move.stones:
                x, y = stone
                np_arr[key][y, x] = 1

    @staticmethod
    def _stone_list2np_arr(np_arr, stone_list):
        for stone in stone_list:
            x, y = stone
            np_arr[y, x] = 1

    def shape(self):
        if self.data_format == 'channels_first':
            return self.num_planes, self.max_xy, self.max_xy
        elif self.data_format == 'channels_last':
            return self.max_xy, self.max_xy, self.num_planes
        else:
            return None


def create(board_size, mode, **kwargs):
    return AlphaAbaloneEncoder(board_size, mode, **kwargs)
