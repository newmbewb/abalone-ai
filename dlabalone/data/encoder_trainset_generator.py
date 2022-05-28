import os
import random

import h5py
import numpy as np

from dlabalone.ablboard import Board
from dlabalone.abltypes import Player
from dlabalone.encoders.base import Encoder
from dlabalone.utils import print_board, load_file_board_move_pair


def _place_stones(grid, count, player):
    stone_count = 0
    while stone_count < count:
        point = random.choice(Board.valid_grids)
        if point not in grid:
            grid[point] = player
            stone_count += 1


def get_encoder_trainset_filelist(dataset_dir, valid_ratio=0.2):
    file_list = []
    for file in os.listdir(dataset_dir):
        file_list.append(os.path.join(dataset_dir, file))
    train_length = int(len(file_list) * (1 - valid_ratio))
    return file_list[:train_length], file_list[train_length:]


def encoder_trainset_generator(file_list):
    while True:
        for file in file_list:
            with h5py.File(file, 'r') as h5file:
                yield h5file['experience']['input'], h5file['experience']['output']


def encoder_validation_generator(dataset_dir):
    file_list = []
    for file in os.listdir(dataset_dir):
        file_list.append(os.path.join(dataset_dir, file))
    while True:
        for file in file_list:
            with h5py.File(file, 'r') as h5file:
                yield h5file['experience']['input'], h5file['experience']['output']


def yield_board(input_dir):
    file_list = os.listdir(input_dir)
    for f in file_list:
        print(f'loading {f}...')
        full_path = os.path.join(input_dir, f)
        pair_list = load_file_board_move_pair(full_path, with_value=True)
        for board, move, advantage, value in pair_list:
            yield board


def generate_encoder_trainset(input_encoder: Encoder, output_encoder: Encoder, batch_size, step_count, input_dir,
                              output_dir):
    Board.set_size(5)
    output_name_format = os.path.join(
        output_dir, f'encoder_trainset_{input_encoder.name()}_to_{output_encoder.name()}_{random.random()}_%d.h5')
    board_history = set()
    board_iter = yield_board(input_dir)
    for step in range(step_count):
        inputs = []
        outputs = []
        for i in range(batch_size):
            try:
                while True:
                    board = next(board_iter)
                    if board in board_history:
                        continue
                    else:
                        board_history.add(board)
                        break
            except StopIteration:
                break
            inputs.append(input_encoder.encode_board(board, Player.black))
            outputs.append(output_encoder.encode_board(board, Player.black))

        with h5py.File(output_name_format % step, 'w') as h5file:
            h5file.create_group('experience')
            h5file['experience'].create_dataset('input', data=np.array(inputs), compression='lzf')
            h5file['experience'].create_dataset('output', data=np.array(outputs), compression='lzf')
