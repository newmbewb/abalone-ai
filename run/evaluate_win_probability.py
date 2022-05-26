import copy
import multiprocessing
import os
import queue
import random
import time

import numpy as np
from keras.models import load_model

from dlabalone.ablboard import GameState, Board
from dlabalone.abltypes import Player
from dlabalone.encoders.base import get_encoder_by_name
from dlabalone.rl.move_selectors.exponential import ExponentialMoveSelector
from dlabalone.rl.simulate import experience_simulation
from dlabalone.rl.trainer import predict_convertor_single_model
from dlabalone.utils import load_file_board_move_pair, encode_board_str
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class GameWrapper(object):
    def __init__(self, game_state, game_info):
        self.game_state = game_state
        self.game_info = game_info
        self.step_count = 0
        self.too_many_repeat = False
        self.past_boards = {}

    def legal_moves(self, **kwargs):
        return self.game_state.legal_moves(**kwargs)

    def can_last_attack(self):
        return self.game_state.can_last_attack()

    def apply_move_lite(self, *args):
        board_str = encode_board_str(self.game_state.board)
        count = self.past_boards.get(board_str, 0) + 1
        self.past_boards[board_str] = count
        if count >= 20:
            self.too_many_repeat = True

        self.step_count += 1
        return self.game_state.apply_move_lite(*args)

    def is_draw(self):
        if self.step_count >= 500:
            return True
        return self.too_many_repeat

    def is_over(self):
        return self.game_state.is_over() or self.is_draw()

    def winner(self):
        return self.game_state.winner()

    def key(self):
        board, move, advantage, value = self.game_info
        return f'{encode_board_str(board)}&{value}'


def simulate_games(device, q_board_info, encoder, model_path, output_filename, num_games, batch_size):
    Board.set_size(5)
    game_list = []
    pending_game_list = []
    stat = {}
    move_selector = ExponentialMoveSelector(fixed_exponent=1)

    # Open output file
    fd_out = open(output_filename, 'w')
    fd_out.write('5,unknown\n')
    fd_out.flush()
    empty_queue = False
    with tf.device(device):  # '/cpu:0'
        model = load_model(model_path)
        start_time = time.time()
        total_count = 0
        while not empty_queue or len(game_list) > 0:
            # Fill game list
            while not empty_queue and len(game_list) + len(pending_game_list) < batch_size:
                # Get new board and put it in pending_game_list
                try:
                    new_board_info = q_board_info.get_nowait()
                    new_game = GameState(new_board_info[0], Player.black)
                    for _ in range(num_games):
                        game_wrapper = GameWrapper(copy.deepcopy(new_game), new_board_info)
                        pending_game_list.append(game_wrapper)
                except queue.Empty:
                    empty_queue = True
                    break
            while len(game_list) < batch_size and len(pending_game_list) > 0:
                game = pending_game_list.pop(0)
                game_list.append(game)

            # Encode game board
            predict_input_list = []
            for game_wrapper in game_list:
                game = game_wrapper.game_state
                predict_input_list.append(encoder.encode_board(game.board, game.next_player))

            # Predict next move
            predict_output = model.predict_on_batch(np.array(predict_input_list))

            # Apply move
            finished_games = []
            for index in range(len(game_list)):
                game_wrapper = game_list[index]
                move_probs = predict_output[index]
                moves_kill, moves_normal = game_wrapper.legal_moves(separate_kill=True)
                if len(moves_kill) > 0 and game_wrapper.can_last_attack():
                    move = moves_kill[0]
                else:
                    move, _ = move_selector(encoder, move_probs, moves_kill + moves_normal)
                game_wrapper.apply_move_lite(move)
                if game_wrapper.is_over():
                    finished_games.append(index)

            # Remove finished games
            for index in finished_games[::-1]:
                game_wrapper = game_list[index]
                game_list.pop(index)
                key = game_wrapper.key()
                result_list = stat.get(key, [])
                if game_wrapper.winner() == Player.black:
                    result_list.append(1)
                elif game_wrapper.winner() == Player.white:
                    result_list.append(0)
                elif game_wrapper.is_draw():
                    result_list.append(0.5)
                else:
                    assert False
                if len(result_list) == num_games:
                    win_rate = sum(result_list) / num_games
                    output = f'{key}&{win_rate}'
                    fd_out.write(output + '\n')
                    fd_out.flush()
                    total_count += 1
                    print(
                        f'{output} (W/L/D: {result_list.count(1)}/{result_list.count(0)}/{result_list.count(0.5)}) '
                        f'{(time.time() - start_time) / total_count:.3f} sec/game')
                    del stat[key]
                else:
                    stat[key] = result_list
    fd_out.close()


if __name__ == '__main__':
    # Arguments
    cpu_threads = 3
    use_gpu = True
    num_games = 100  # recommandation: 50
    batch_size = 1024

    dataset_dir = '../data/data_with_value/dataset/'
    output_dir = '../data/rl_mcts/win_probability_evaluation'
    model_path = '../data/checkpoints/models/ACSimple1Policy_dropout0.1_FourPlaneEncoder_channels_last_epoch_13.h5'
    encoder = get_encoder_by_name('fourplane', 5, '', data_format='channels_last')

    file_batch_size = 10
    # Code start from here
    file_list = os.listdir(dataset_dir)
    random.shuffle(file_list)
    m = multiprocessing.Manager()
    # q_board_info = multiprocessing.Queue()
    q_board_info = m.Queue()
    while len(file_list) > 0:
        for f in file_list[:file_batch_size]:
            print(f'loading {f}...')
            full_path = os.path.join(dataset_dir, f)
            pair_list = load_file_board_move_pair(full_path, with_value=True)
            for pair in pair_list:
                q_board_info.put_nowait(pair)
        file_list = file_list[file_batch_size:]

        # (q_board_info, encoder, model_generator, output_filename, num_games=50, batch_size=1024):
        args = []
        for i in range(cpu_threads):
            output_filepath = os.path.join(output_dir, f'evaluated_win_probability_{random.random()}_{i}.txt')
            args.append((
                '/cpu:0', q_board_info, encoder, model_path, output_filepath, num_games, batch_size))
        if use_gpu:
            threads = 1 + cpu_threads
            output_filepath = os.path.join(output_dir, f'evaluated_win_probability_{random.random()}_gpu.txt')
            args.append((
                '/gpu:0', q_board_info, encoder, model_path, output_filepath, num_games, batch_size))
        else:
            threads = cpu_threads

        with multiprocessing.Pool(processes=threads) as p:
            result_list = p.starmap(simulate_games, args)

    # move_selector = ExponentialMoveSelector(fixed_exponent=1)
    # predict_convertor = predict_convertor_single_model
    # start_time = time.time()
    # total_count = 0
    # for f in file_list:
    #     output_filepath = os.path.join(output_dir, f'generation_0_{f}.txt')
    #     if os.path.exists(output_filepath):
    #         continue
    #     print(f'Evaluating {f}...')
    #     fd_out = open(output_filepath, 'w')
    #     fd_out.write('5,unknown\n')
    #     fd_out.flush()
    #
    #     full_path = os.path.join(dataset_dir, f)
    #     pair_list = load_file_board_move_pair(full_path, with_value=True)
    #     for board, move, advantage, value in pair_list:
    #         wins = 0
    #         losses = 0
    #         draws = 0
    #         win_count = 0
    #         start_state = GameState(board, Player.black)
    #         stat_list = experience_simulation(num_games,
    #                                           [model_policy], [model_policy],
    #                                           encoder, encoder,
    #                                           move_selector, move_selector,
    #                                           predict_convertor, predict_convertor,
    #                                           populate_games=False, start_state=start_state)
    #         for stat in stat_list:
    #             if stat['winner'] == Player.black:
    #                 wins += 1
    #                 win_count += 1
    #             elif stat['winner'] == Player.white:
    #                 losses += 1
    #             else:
    #                 draws += 1
    #                 win_count += 0.5
    #         win_rate = win_count / len(stat_list)
    #         total_count += 1
    #
    #         output = f'{encode_board_str(board)}&{value}&{win_rate}'
    #         fd_out.write(output + '\n')
    #         fd_out.flush()
    #         print(f'{output} (W/L/D: {wins}/{losses}/{draws}) {(time.time() - start_time)/total_count:.3f} sec/game')
    #     fd_out.close()
