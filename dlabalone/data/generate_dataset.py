import os
import random

from dlabalone.utils import board_move_seperator


def generate_dataset(game_dir, data_per_file, dataset_file, remove_duplicated=False):
    game_files = [f for f in map(lambda x: os.path.join(game_dir, x), os.listdir(game_dir)) if os.path.isfile(f)]
    dataset = []
    size = 0
    print(f'Start to read {len(game_files)} files...')
    file_count = 0
    # Read all files
    for file in game_files:
        file_count += 1
        if file_count % 100 == 0:
            print(f'{file_count}/{len(game_files)} Done..')
        fd = open(file)
        size, line = fd.readline().split(',')
        # size = int(size)
        for line in fd:
            dataset.append(line)

    if remove_duplicated:
        # Remove duplicated
        board_set = set()
        dataset_wo_duplicated = []
        for line in dataset:
            board_str = line.split(board_move_seperator)[0]
            if board_str not in board_set:
                board_set.add(board_str)
                dataset_wo_duplicated.append(line)
        print(f'{len(dataset) - len(dataset_wo_duplicated)} duplicated board deleted.')
        dataset = dataset_wo_duplicated

    # Shuffle dataset
    random.shuffle(dataset)
    dataset_len = len(dataset)
    print(f'Start to save {dataset_len} data')
    for i in range(0, len(dataset), data_per_file):
        print(f'{i}/{dataset_len} Done..')
        fd = open(dataset_file % (i//data_per_file), 'w')
        fd.write(f'{size},{min(i + data_per_file, dataset_len) - i}\n')
        for j in range(i, min(i + data_per_file, dataset_len)):
            fd.write(dataset[j])
        fd.close()
