import cProfile
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
from dlabalone.networks.base import prepare_tf_custom_objects
from dlabalone.rl.move_selectors.exponential import ExponentialMoveSelector
from dlabalone.rl.simulate import experience_simulation
from dlabalone.rl.trainer import predict_convertor_single_model
from dlabalone.utils import load_file_board_move_pair, encode_board_str
import tensorflow as tf


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


def _simulate_game_worker(device, q_board_info, encoder, model_path, output_filename, num_games, batch_size):
    Board.set_size(5)
    prepare_tf_custom_objects()
    game_list = []
    pending_game_list = []
    stat = {}

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

                # Randomly select move based on the probability
                # We do not use choice because of performance
                ranked_moves = np.argsort(move_probs)[::-1]
                move = None
                p_accum = 0
                for move_idx in range(ranked_moves.shape[0]-1, -1, -1):
                    p = move_probs[move_idx]
                    if random.random() - p_accum < p:
                        move = encoder.decode_move_index(move_idx)
                        if game_wrapper.game_state.is_valid_move(move.stones, move.direction):
                            break
                        else:
                            move = None
                    p_accum += p
                # If non move is selected, we use np.random.choice
                if move is None:
                    eps = 1e-6
                    move_probs = np.nan_to_num(move_probs)
                    move_probs = np.clip(move_probs, eps, 1 - eps)
                    move_probs = move_probs / np.sum(move_probs)
                    num_moves = encoder.num_moves()
                    ranked_moves = np.random.choice(np.arange(num_moves), num_moves, replace=False, p=move_probs)
                    for move_idx in ranked_moves:
                        move = encoder.decode_move_index(move_idx)
                        if game_wrapper.game_state.is_valid_move(move.stones, move.direction):
                            break

                # Apply move
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


def evaluate_winrates(file_list, remain_samples, output_dir, encoder, policy_model_path, num_games, batch_size, cpu_threads, use_gpu):
    m = multiprocessing.Manager()
    q_board_info = m.Queue()
    stop = False
    sample_count = 0
    for f in file_list:
        pair_list = load_file_board_move_pair(f, with_value=True)
        for pair in pair_list:
            q_board_info.put_nowait(pair)
            sample_count += 1
            if sample_count >= remain_samples:
                stop = True
                break
        if stop:
            break

    args = []
    random_tag = random.random()
    for i in range(cpu_threads):
        output_filepath = os.path.join(output_dir, f'evaluated_win_probability_{random_tag}_cpu_{i}.txt')
        args.append((
            '/device:CPU:0', q_board_info, encoder, policy_model_path, output_filepath, num_games, batch_size))
    if use_gpu:
        threads = 1 + cpu_threads
        output_filepath = os.path.join(output_dir, f'evaluated_win_probability_{random_tag}_gpu.txt')
        args.append((
            '/device:GPU:0', q_board_info, encoder, policy_model_path, output_filepath, num_games, batch_size))
    else:
        threads = cpu_threads

    with multiprocessing.Pool(processes=threads) as p:
        p.starmap(_simulate_game_worker, args)
