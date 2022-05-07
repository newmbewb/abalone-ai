import importlib
import queue
import numpy as np
from dlabalone.ablboard import Board, Move
from dlabalone.abltypes import Direction, add
from tensorflow.keras.utils import to_categorical

__all__ = [
    'Encoder',
    'get_encoder_by_name',
]

from dlabalone.utils import load_file_board_move_pair


class Encoder:
    def __init__(self, board_size):
        self.max_xy = board_size * 2 - 1
        self.move_list = []  # index -> move
        self.move_dict = {}  # move -> index
        Board.set_size(board_size)

        # Generate move list
        for string_length in range(1, 4):
            if string_length == 1:
                string_directions = [(0, 0)]
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
                        next_stone = add(next_stone, string_direction)
                    if invalid_move:
                        continue
                    stones.sort()
                    for move_direction in map(lambda x: x.value, Direction):
                        # Check whether it is valid move direction
                        invalid_move = False
                        for stone in stones:
                            next_point = add(stone, move_direction)
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

    def encode_board(self, game_board, next_player):
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
        label_list = []
        while True:
            try:
                file = q_infile.get_nowait()
            except queue.Empty:
                break
            pair_list = load_file_board_move_pair(file)
            print(f'Encoding {file}')
            for board, move in pair_list:
                label = encoder.encode_move(move)
                label = to_categorical(label, num_classes)
                feature = encoder.encode_board(board)

                feature = np.expand_dims(feature, axis=0)
                label = np.expand_dims(label, axis=0)
                feature_list.append(feature)
                label_list.append(label)
                if len(feature_list) == batch_size:
                    features = np.concatenate(feature_list, axis=0)
                    labels = np.concatenate(label_list, axis=0)
                    feature_list = []
                    label_list = []
                    with lock:
                        idx = index.value
                        index.value += 1
                    np.savez_compressed(filename % idx, feature=features, label=labels)

        q_ggodari.put((feature_list, label_list))
        return

    @staticmethod
    def ggodari_merger(ggodari_list, batch_size, out_filename, index):
        feature_list = []
        label_list = []
        for feature, label in ggodari_list:
            feature_list += feature
            label_list += label

        while len(feature_list) > 0:
            features = np.concatenate(feature_list[:batch_size], axis=0)
            labels = np.concatenate(label_list[:batch_size], axis=0)
            feature_list = feature_list[batch_size:]
            label_list = label_list[batch_size:]
            np.savez_compressed(out_filename % index, feature=features, label=labels)
            index += 1

    def load(self, file):
        loaded = np.load(file)
        return loaded['feature'], loaded['label']

    def label_shape(self):
        return self.num_moves(),

    def shape(self):
        raise NotImplementedError()


def get_encoder_by_name(name, board_size, *args):
    module = importlib.import_module('dlabalone.encoders.' + name)
    constructor = getattr(module, 'create')
    return constructor(board_size, *args)
