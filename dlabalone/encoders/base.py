import importlib
import queue
import numpy as np
from dlabalone.ablboard import Board, Move
from dlabalone.abltypes import Direction
from tensorflow.keras.utils import to_categorical

__all__ = [
    'Encoder',
    'get_encoder_by_name',
]

from dlabalone.utils import load_file_board_move_pair


class Encoder:
    def __init__(self, board_size, mode):
        self.max_xy = board_size * 2 - 1
        self.move_list = []  # index -> move
        self.move_dict = {}  # move -> index
        self.mode = mode
        Board.set_size(board_size)

        # Generate move list
        for string_length in range(1, 4):
            if string_length == 1:
                string_directions = [0]
            else:
                string_directions = list(map(lambda x: x.value,
                                             [Direction.EAST, Direction.SOUTHEAST, Direction.SOUTHWEST]))
            for head_point in Board.valid_grids:
                for string_direction in string_directions:
                    # Make stone list
                    invalid_move = False
                    stones = []
                    next_stone = head_point
                    for _ in range(string_length):
                        if next_stone not in Board.valid_grids:
                            invalid_move = True
                            # break
                        stones.append(next_stone)
                        next_stone = next_stone + string_direction
                    if invalid_move:
                        continue
                    stones.sort()
                    for move_direction in map(lambda x: x.value, Direction):
                        # Check whether it is valid move direction
                        invalid_move = False
                        for stone in stones:
                            next_point = stone + move_direction
                            if next_point not in Board.valid_grids:
                                invalid_move = True
                                break
                        if invalid_move:
                            continue
                        # Add move
                        self.move_list.append((tuple(stones), Direction.to_int(move_direction)))

        # Generate move dictionary
        for index, move in enumerate(self.move_list):
            self.move_dict[move] = index

    def name(self):
        raise NotImplementedError()

    def encode_board(self, game_board, next_player, **kwargs):
        raise NotImplementedError()

    def encode_move(self, move):
        stones = move.stones
        stones.sort()
        stones = tuple(stones)
        direction = Direction.to_int(move.direction)
        return self.move_dict[(stones, direction)]

    def decode_move_index(self, index):
        stones, direction = self.move_list[index]
        return Move(list(stones), Direction.from_int(direction))

    def num_moves(self):
        return len(self.move_list)

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

    def load(self, file):
        loaded = np.load(file)
        return loaded['feature'], loaded[self.mode]

    def label_shape(self):
        if self.mode == 'policy':
            return self.num_moves(),
        elif self.mode == 'value':
            return 1,
        else:
            assert False, "Wrong mode"

    def shape(self):
        raise NotImplementedError()


def get_encoder_by_name(name, board_size, *args, **kwargs):
    module = importlib.import_module('dlabalone.encoders.' + name)
    constructor = getattr(module, 'create')
    return constructor(board_size, *args, **kwargs)
