import multiprocessing
import os.path
import random
from multiprocessing import Manager

import tensorflow as tf
from keras.models import load_model

from dlabalone.ablboard import Board, GameState
from dlabalone.agent.mcts_ac import MCTSACBot
from dlabalone.networks.base import prepare_tf_custom_objects
from dlabalone.utils import save_file_state_move_pair, encode_board_str


def _game_generation_worker(output_dir, encoder, policy_model_path, value_model_path, target_step_count,
                            global_step_count, lock, device, worker_index):
    Board.set_size(5)
    game_index = 0

    done = False
    with tf.device(device):  # '/cpu:0':
        critic = load_model(value_model_path)
        actor = load_model(policy_model_path)
        while not done:
            # Initialize variables and run game
            game = GameState.new_game(5)
            bot = MCTSACBot(encoder, actor, critic, width=3, num_rounds=1000, batch_size=128)
            pair_list = []
            past_boards = {}
            step = 0
            while not game.is_over():
                move = bot.select_move(game)
                pair_list.append((game, move))
                game = game.apply_move(move)
                step += 1

                # If a same board repeats too much, the game is draw
                board_str = encode_board_str(game.board)
                count = past_boards.get(board_str, 0) + 1
                past_boards[board_str] = count
                if count >= 30:
                    is_draw = True
                    break

                if step >= 1000:
                    is_draw = True
                    break

            # Save generated game
            if is_draw:
                draw_str = '_draw'
            else:
                draw_str = ''
            filename = os.path.join(output_dir, f'game_{worker_index}_{game_index}{draw_str}_{random.random()}.txt')
            game_index += 1
            save_file_state_move_pair(filename, pair_list)

            # Check whether we made the enough number of steps
            with lock:
                global_step_count.value += step
                if global_step_count.value >= target_step_count:
                    done = True
            print(f'One game end: {step} steps generated.')


def generate_games(output_dir, encoder, policy_model_path, value_model_path, step_count, cpu_threads, use_gpu):
    manager = Manager()
    global_step_count = manager.Value('i', 0)
    lock = manager.Lock()

    args = []
    worker_index = 0
    for i in range(cpu_threads):
        args.append((output_dir, encoder, policy_model_path, value_model_path, step_count,
                     global_step_count, lock, '/device:CPU:0', worker_index))
        worker_index += 1
    if use_gpu:
        threads = 1 + cpu_threads
        args.append((output_dir, encoder, policy_model_path, value_model_path, step_count,
                     global_step_count, lock, '/device:GPU:0', worker_index))
    else:
        threads = cpu_threads

    with multiprocessing.Pool(processes=threads) as p:
        p.starmap(_game_generation_worker, args)
