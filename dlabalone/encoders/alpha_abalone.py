import queue
from tensorflow.keras.utils import to_categorical

import numpy as np

from dlabalone.ablboard import Board, GameState
from dlabalone.abltypes import Player, Direction
from dlabalone.encoders.base import Encoder
from dlabalone.encoders.plane_generator import calc_opp_layers
from dlabalone.utils import load_file_board_move_pair


class AlphaAbaloneEncoder(Encoder):
    def __init__(self, board_size, mode):
        super().__init__(board_size)
        self.num_planes = 45
        self.valid_map = np.zeros((self.max_xy, self.max_xy))
        self.invalid_map = np.ones((self.max_xy, self.max_xy))
        self.mode = mode
        for x, y in Board.valid_grids:
            self.valid_map[y, x] = 1
            self.valid_map[y, x] = 0

    def name(self):
        return type(self).__name__

    def encode_board(self, game_board, next_player=Player.black):
        game = GameState(game_board, next_player)
        push_moves = []
        kill_moves, normal_moves = game.legal_moves(separate_kill=True, push_moves=push_moves)
        player_attack_plains = self._generate_player_attack_plains(kill_moves, push_moves)
        opp_attack_plains = self._generate_opp_attack_plains(game_board, next_player)
        basic_plains = self._generate_basic_plains(game_board, next_player)
        return np.concatenate([player_attack_plains, opp_attack_plains, basic_plains])

    def _generate_basic_plains(self, board, next_player=Player.black):
        board_matrix = np.zeros((4, self.max_xy, self.max_xy))
        for point, player in board.grid.items():
            x, y = point
            if player == next_player:
                board_matrix[0, y, x] = 1
            else:
                board_matrix[1, y, x] = 1
        board_matrix[2] = self.valid_map
        board_matrix[3] = self.invalid_map
        return board_matrix

    def _generate_player_attack_plains(self, kill_moves, push_moves):
        # Init plains
        plain_kill = {}
        plain_push = {}
        for length in [2, 3]:
            for direction in Direction:
                key = length, direction.value
                plain_kill[key] = np.zeros((1, self.max_xy, self.max_xy))
                plain_push[key] = np.zeros((1, self.max_xy, self.max_xy))

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

        return np.concatenate(plain_kill_list + plain_push_list)

    def _generate_opp_attack_plains(self, board, next_player=Player.black):
        plain_opp_sumito_stones = np.zeros((1, self.max_xy, self.max_xy))
        plain_danger_stones = np.zeros((1, self.max_xy, self.max_xy))
        plain_danger_points = np.zeros((1, self.max_xy, self.max_xy))
        plain_vuln_stones = {2: np.zeros((1, self.max_xy, self.max_xy)), 3: np.zeros((1, self.max_xy, self.max_xy))}
        plain_vuln_points = {}
        for length in [2, 3]:
            for direction in Direction:
                key = length, direction.value
                plain_vuln_points[key] = np.zeros((1, self.max_xy, self.max_xy))

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
        return np.concatenate([plain_opp_sumito_stones, plain_danger_stones, plain_danger_points] +
                              plain_vuln_stones_list +
                              plain_vuln_points_list)

    @staticmethod
    def encode_data_worker(encoder, batch_size, filename, q_infile, q_ggodari, index, lock):
        num_classes = encoder.num_moves()
        feature_list = []
        policy_list = []
        value_list = []
        while True:
            try:
                file = q_infile.get_nowait()
            except queue.Empty:
                break
            pair_list = load_file_board_move_pair(file, with_value=True)
            print(f'Encoding {file}')
            for board, move, advantage, value in pair_list:
                move_num = encoder.encode_move(move)
                policy = np.zeros(num_classes)
                policy[move_num] = advantage
                feature = encoder.encode_board(board)

                feature = np.expand_dims(feature, axis=0)
                policy = np.expand_dims(policy, axis=0)
                value = np.expand_dims(value, axis=0)
                feature_list.append(feature)
                policy_list.append(policy)
                value_list.append(value)
                if len(feature_list) == batch_size:
                    features = np.concatenate(feature_list, axis=0)
                    policies = np.concatenate(policy_list, axis=0)
                    values = np.concatenate(value_list, axis=0)
                    feature_list = []
                    policy_list = []
                    value_list = []
                    with lock:
                        idx = index.value
                        index.value += 1
                    np.savez_compressed(filename % idx, feature=features, policy=policies, value=values)

        q_ggodari.put((feature_list, policy_list, value_list))
        return

    @staticmethod
    def ggodari_merger(ggodari_list, batch_size, out_filename, index):
        feature_list = []
        policy_list = []
        value_list = []
        for feature, policy, value in ggodari_list:
            feature_list += feature
            policy_list += policy
            value_list += value

        while len(feature_list) > 0:
            features = np.concatenate(feature_list[:batch_size], axis=0)
            policies = np.concatenate(policy_list[:batch_size], axis=0)
            values = np.concatenate(value_list[:batch_size], axis=0)
            feature_list = feature_list[batch_size:]
            policy_list = policy_list[batch_size:]
            value_list = value_list[batch_size:]
            np.savez_compressed(out_filename % index, feature=features, policy=policies, value=values)
            index += 1

    @staticmethod
    def _move_list2np_arr(np_arr, move_list):
        for move in move_list:
            length = len(move.stones)
            direction = move.direction
            key = length, direction
            for stone in move.stones:
                x, y = stone
                np_arr[key][0, y, x] = 1

    @staticmethod
    def _stone_list2np_arr(np_arr, stone_list):
        for stone in stone_list:
            x, y = stone
            np_arr[0, y, x] = 1

    def load(self, file):
        loaded = np.load(file)
        return loaded['feature'], loaded[self.mode]

    def shape(self):
        return self.num_planes, self.max_xy, self.max_xy

    def label_shape(self):
        if self.mode == 'policy':
            return self.num_moves(),
        elif self.mode == 'value':
            return 1,
        else:
            assert False, "Wrong mode"


def create(board_size, mode):
    return AlphaAbaloneEncoder(board_size, mode)
