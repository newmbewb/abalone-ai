import os
from dlabalone.data.data_converter import plaindata2valuedata
from dlabalone.data.generate_dataset import generate_dataset
from dlabalone.data.populate_games import GamePopulator

if __name__ == '__main__':
    board_size = 5
    data_per_file = 4096
    plain_game_dir_parent = '../data/generated_games'
    plain_game_dir_paths = []
    game_dir_path = '../data/data_with_value/generated_games'
    for f in os.listdir(plain_game_dir_parent):
        path = os.path.join(plain_game_dir_parent, f)
        if os.path.isdir(path):
            plain_game_dir_paths.append(path)

    populated_game_dir = '../data/data_with_value/populated_games/'
    dataset_file = f'../data/data_with_value/dataset/data{data_per_file}_%06d.txt'
    populated_game_name = os.path.join(populated_game_dir, 'game_%d.txt')

    # Convert normal game data to game with value
    for path in plain_game_dir_paths:
        plaindata2valuedata(path, game_dir_path)

    print('Start to populate games')
    populator = GamePopulator(board_size)
    populator.populate_games([game_dir_path], populated_game_name, with_value=True)

    print('Start to generate dataset')
    generate_dataset(populated_game_dir, data_per_file, dataset_file)
