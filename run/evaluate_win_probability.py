import os
import random

from keras.models import load_model

from dlabalone.ablboard import GameState
from dlabalone.abltypes import Player
from dlabalone.encoders.base import get_encoder_by_name
from dlabalone.rl.move_selectors.exponential import ExponentialMoveSelector
from dlabalone.rl.simulate import experience_simulation
from dlabalone.rl.trainer import predict_convertor_single_model
from dlabalone.utils import load_file_board_move_pair, encode_board_str

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    # Arguments
    num_games = 100
    dataset_dir = '../data/data_with_value/dataset/'
    output_dir = '../data/rl_mcts/win_probability_evaluation'
    encoder = get_encoder_by_name('fourplane', 5, '', data_format='channels_last')
    model_policy = load_model(
            '../data/checkpoints/models/ACSimple1Policy_dropout0.1_FourPlaneEncoder_channels_last_epoch_13.h5')

    # Code start from here
    file_list = os.listdir(dataset_dir)
    random.shuffle(file_list)
    move_selector = ExponentialMoveSelector(fixed_exponent=1)
    predict_convertor = predict_convertor_single_model
    for f in file_list:
        output_filepath = os.path.join(output_dir, f'generation_0_{f}.txt')
        if os.path.exists(output_filepath):
            continue
        print(f'Evaluating {f}...')
        fd_out = open(output_filepath, 'w')

        full_path = os.path.join(dataset_dir, f)
        pair_list = load_file_board_move_pair(full_path, with_value=True)
        for board, move, advantage, value in pair_list:
            wins = 0
            losses = 0
            draws = 0
            win_count = 0
            start_state = GameState(board, Player.black)
            stat_list = experience_simulation(num_games,
                                              [model_policy], [model_policy],
                                              encoder, encoder,
                                              move_selector, move_selector,
                                              predict_convertor, predict_convertor,
                                              populate_games=False, start_state=start_state)
            for stat in stat_list:
                if stat['winner'] == Player.black:
                    wins += 1
                    win_count += 1
                elif stat['winner'] == Player.white:
                    losses += 1
                else:
                    draws += 1
                    win_count += 0.5
            win_rate = win_count / len(stat_list)
            output = f'{encode_board_str(board)}&{value}&{win_rate}'
            fd_out.write(output)
            fd_out.flush()
            print(f'{output} (W/L/D: {wins}/{losses}/{draws})')
        fd_out.close()