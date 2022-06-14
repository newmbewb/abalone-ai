import os
from pathlib import Path

from dlabalone.data.data_converter import plaindata2valuedata
from dlabalone.data.generate_dataset import generate_dataset
from dlabalone.data.populate_games import GamePopulator

if __name__ == '__main__':
    board_size = 5
    data_per_file = 1024
    data_home = '../data/rl_mcts/generation03_manual'
    mode = 'policy'
    if mode == 'value':
        orig_data_dir_path = os.path.join(data_home, 'win_probability_evaluation')
        shuffled_dir = None
    elif mode == 'policy':
        orig_data_dir_path = os.path.join(data_home, 'generated_games_pc')
        shuffled_dir = os.path.join(data_home, 'shuffled_moves')
    else:
        assert False
    if shuffled_dir is not None:
        Path(shuffled_dir).mkdir(parents=True, exist_ok=True)
        generate_dataset(orig_data_dir_path, data_per_file, os.path.join(shuffled_dir, f'data{data_per_file}_%06d.txt'),
                         remove_duplicated=True)
    populated_data_dir = os.path.join(data_home, f'{mode}_populated_games')
    dataset_dir = os.path.join(data_home, f'{mode}_dataset')
    populated_data_name = os.path.join(populated_data_dir, 'game_%d.txt')
    dataset_file_name = os.path.join(dataset_dir, f'data{data_per_file}_%06d.txt')
    Path(populated_data_dir).mkdir(parents=True, exist_ok=True)
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)

    print('Start to populate games')
    populator = GamePopulator(board_size)
    populator.populate_games([orig_data_dir_path], populated_data_name, with_value=True)

    print('Start to generate dataset')
    generate_dataset(populated_data_dir, data_per_file, dataset_file_name)
