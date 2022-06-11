import multiprocessing
import os
import time

from keras.models import load_model
import tensorflow as tf

from dlabalone.ablboard import GameState, Board
from dlabalone.abltypes import Player
from dlabalone.agent.mcts_ac import MCTSACBot
from dlabalone.agent.opening import OpeningBot
from dlabalone.agent.random_kill_first import RandomKillBot
from dlabalone.agent.alphabeta import AlphaBetaBot
from dlabalone.agent.mcts import MCTSBot
from dlabalone.encoders.base import get_encoder_by_name
from dlabalone.networks.base import prepare_tf_custom_objects
from dlabalone.utils import print_board, encode_board_str, save_file_state_move_pair, profiler, \
    load_file_board_move_pair, decode_board_from_str
import cProfile
import random
import sys
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

draw_name = 'Draw'
date = datetime.now().strftime('%y%m%d%H%M%S')
global_filename = f'../data/rl_mcts/generation02_manual/generated_games/game_{date}_%d-%s.txt'


def visualize_game(infile, outfile):
    pair_list = load_file_board_move_pair(infile)
    old_stdout = sys.stdout
    sys.stdout = open(outfile, 'w')
    for board, move in pair_list:
        print_board(board)
        print('-----------------')
    sys.stdout = old_stdout


def run_game(idx, bot_generator):
    opening_bot = OpeningBot('../data/opening/')
    bot = bot_generator()
    game = GameState.new_game(5, reverse=True)
    step = 0
    is_draw = False
    pair_list = []
    past_boards = {}
    start_time = time.time()
    depth = []
    while not game.is_over():
        if not opening_bot.is_end():
            move = opening_bot.select_move(game)
        else:
            move = bot.select_move(game)
        pair_list.append((game, move))
        game = game.apply_move(move)
        step += 1

        # If a same board repeats too much, the game is draw
        board_str = encode_board_str(game.board)
        count = past_boards.get(board_str, 0) + 1
        past_boards[board_str] = count
        if count >= 5:
            is_draw = True
            break

        if step >= 10000:
            is_draw = True
            break
    runtime = time.time() - start_time
    #########
    # print(f'Max repeated board: {max(past_boards.values())}')
    if is_draw:
        draw_str = '_draw'
    else:
        draw_str = ''
    filename = global_filename % (idx, draw_str)
    save_file_state_move_pair(filename, pair_list)

    current_time = datetime.now().strftime('%y/%m/%d %H:%M:%S')

    print(f'({current_time})  step:{step:4d}, time: {runtime:.2f}sec ({runtime/step:8.05f} sec/step)')
    return step


def run_game_start_from_board(idx, bot_generator_board_pair):
    Board.set_size(5)
    bot_generator, board_str = bot_generator_board_pair
    print(board_str)
    board_index, board_str = board_str.split(':')
    board = decode_board_from_str(board_str, 9, Player.black)
    bot = bot_generator()
    game = GameState(board, Player.black)
    step = 0
    is_draw = False
    pair_list = []
    past_boards = {}
    start_time = time.time()
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

        if step >= 10000:
        # if step >= 10:
            is_draw = True
            break

        # if step % 2 == 0:
        #     runtime = time.time() - start_time
        #     print(f'({runtime/step:8.05f} sec/step)')
    runtime = time.time() - start_time
    #########
    # print(f'Max repeated board: {max(past_boards.values())}')
    if is_draw:
        draw_str = '_draw'
    else:
        draw_str = ''
    filename = global_filename % (idx, draw_str) + '.continue.txt'
    save_file_state_move_pair(filename, pair_list)

    current_time = datetime.now().strftime('%y/%m/%d %H:%M:%S')

    print(f'({current_time}) ({board_index}) step:{step:4d}, time: {runtime:.2f}sec ({runtime/step:8.05f} sec/step)')
    return step


def generate_data(bot_generator, run_iter=100, threads=None):
    args = [bot_generator] * run_iter

    if threads is None:
        threads = os.cpu_count()
    with multiprocessing.Pool(processes=threads) as p:
        p.starmap(run_game, enumerate(args))


def generate_data_from_board(bot_generator, threads=None):
    fd = open('last_boards.txt')
    args = []
    for line in fd:
        args.append((bot_generator, line.strip()))

    if threads is None:
        threads = os.cpu_count()
    with multiprocessing.Pool(processes=threads) as p:
        p.starmap(run_game_start_from_board, enumerate(args))


def mcts_ac_bot_generator():
    tf.debugging.disable_traceback_filtering()
    prepare_tf_custom_objects()

    num_rounds = 3000
    width = 'dynamic'
    # tf.compat.v1.disable_eager_execution()
    encoder = get_encoder_by_name('fourplane', 5, None, data_format='channels_last')
    actor = load_model(
        '../data/rl_mcts/generation01_manual/policy_model.h5')
    critic = load_model(
        '../data/rl_mcts/generation00/ACSimple1Value_dropout0.5_FourPlaneEncoder_channels_last_epoch_100.h5')
    return MCTSACBot(encoder, actor, critic, name=f"bot_mcts_ac_w{width}_r{num_rounds}",
                     width=width, num_rounds=num_rounds, temperature=0.01)
    # return MCTSACBot(encoder, actor, critic, name="bot_mcts_ac_w3_r3000", width=3, num_rounds=3000, temperature=0.01)


if __name__ == '__main__':
    # generate_data_from_board(mcts_ac_bot_generator, 3)
    # print('------------------------- start next -------------------------')
    generate_data(mcts_ac_bot_generator, 100000, 1)
