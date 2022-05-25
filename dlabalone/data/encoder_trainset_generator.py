import os
import random

import h5py
import numpy as np

from dlabalone.ablboard import Board
from dlabalone.abltypes import Player
from dlabalone.encoders.base import Encoder
from dlabalone.utils import print_board


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


def generate_encoder_trainset(input_encoder: Encoder, output_encoder: Encoder, batch_size, step_count, dataset_dir):
    Board.set_size(5)
    valid_stone_counts = list(range(9, 14+1))
    output_name_format = os.path.join(
        dataset_dir, f'encoder_trainset_{input_encoder.name()}_to_{output_encoder.name()}_{random.random()}_%d.h5')
    for step in range(step_count):
        inputs = []
        outputs = []
        for i in range(batch_size):
            black_count = random.choice(valid_stone_counts)
            white_count = random.choice(valid_stone_counts)
            dead_stones_black = 14 - black_count
            dead_stones_white = 14 - white_count
            grid = {}
            _place_stones(grid, black_count, Player.black)
            _place_stones(grid, white_count, Player.white)
            board = Board(grid, dead_stones_black, dead_stones_white)
            inputs.append(input_encoder.encode_board(board, Player.black))
            outputs.append(output_encoder.encode_board(board, Player.black))

        with h5py.File(output_name_format % step, 'w') as h5file:
            h5file.create_group('experience')
            h5file['experience'].create_dataset('input', data=np.array(inputs), compression='lzf')
            h5file['experience'].create_dataset('output', data=np.array(outputs), compression='lzf')
