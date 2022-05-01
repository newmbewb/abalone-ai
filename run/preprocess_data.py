import os

from dlabalone.data.generate_dataset import generate_dataset
from dlabalone.data.populate_games import GamePopulator

if __name__ == '__main__':
    board_size = 5
    data_per_file = 4096
    game_dir_parent = '../data/generated_games'
    game_dir_paths = []
    for f in os.listdir(game_dir_parent):
        path = os.path.join(game_dir_parent, f)
        if os.path.isdir(path):
            game_dir_paths.append(path)

    populated_game_dir = '../data/populated_games/'
    dataset_file = f'../data/dataset/data{data_per_file}_%06d.txt'
    populated_game_name = os.path.join(populated_game_dir, 'game_%d.txt')

    print('Start to populate games')
    populator = GamePopulator(board_size)
    populator.populate_games(game_dir_paths, populated_game_name)
    print('Start to generate dataset')
    generate_dataset(populated_game_dir, data_per_file, dataset_file)
