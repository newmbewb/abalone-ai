import os
from pathlib import Path

from dlabalone.data.generate_dataset import generate_dataset
from dlabalone.data.populate_games import GamePopulator

if __name__ == '__main__':
    board_size = 5
    data_per_file = 4096
    home_dir = '../data/rl_mcts/generation01_manual'
    game_dir_parent = os.path.join(home_dir, 'generated_games')
    game_dir_paths = [game_dir_parent]
    # for f in os.listdir(game_dir_parent):
    #     path = os.path.join(game_dir_parent, f)
    #     if os.path.isdir(path):
    #         game_dir_paths.append(path)

    populated_game_dir = os.path.join(home_dir, 'populated_games')
    shuffled_move_dir = os.path.join(home_dir, 'shuffled_moves')
    dataset_dir = os.path.join(home_dir, f'dataset')

    populated_game_name = os.path.join(populated_game_dir, 'game_%d.txt')
    shuffled_move_file = os.path.join(shuffled_move_dir, f'data{data_per_file}_%06d.txt')
    dataset_file = os.path.join(dataset_dir, f'data{data_per_file}_%06d.txt')

    # Generate directories
    Path(populated_game_dir).mkdir(parents=True, exist_ok=True)
    Path(shuffled_move_dir).mkdir(parents=True, exist_ok=True)
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)

    print('Shuffle moves')
    generate_dataset(game_dir_parent, data_per_file, shuffled_move_file)
    print('Start to populate games')
    populator = GamePopulator(board_size)
    populator.populate_games(game_dir_paths, populated_game_name, with_value=True)
    print('Start to generate dataset')
    generate_dataset(populated_game_dir, data_per_file, dataset_file)
