import os
import random


def generate_dataset(game_dir, data_per_file, dataset_file):
    game_files = [f for f in map(lambda x: os.path.join(game_dir, x), os.listdir(game_dir)) if os.path.isfile(f)]
    dataset = []
    size = 0
    print(f'Start to read {len(game_files)} files...')
    file_count = 0
    for file in game_files:
        file_count += 1
        if file_count % 100 == 0:
            print(f'{file_count}/{len(game_files)} Done..')
        fd = open(file)
        size, line = fd.readline().split(',')
        size = int(size)
        for line in fd:
            dataset.append(line)
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
