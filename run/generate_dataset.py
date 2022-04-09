import os
import random

if __name__ == '__main__':
    game_dir = '../populated_games'
    data_per_file = 256
    dataset_file = f'../dataset/data{data_per_file}_%06d.txt'
    game_files = [f for f in map(lambda x: os.path.join(game_dir, x), os.listdir(game_dir)) if os.path.isfile(f)]
    dataset = []
    for file in game_files:
        fd = open(file)
        size, line = fd.readline().split(',')
        size = int(size)
        line = int(line)
        for line in fd:
            dataset.append(line)
    random.shuffle(dataset)
    dataset_len = len(dataset)
    print(dataset_len)
    for i in range(0, len(dataset), data_per_file):
        print(i)
        fd = open(dataset_file % (i//data_per_file), 'w')
        for j in range(i, min(i + data_per_file, dataset_len)):
            fd.write(dataset[j])
        fd.close()
