from datetime import datetime
import multiprocessing
import os
import time

from dlabalone.ablboard import GameState
from dlabalone.abltypes import Player
from dlabalone.agent.mcts_ac import MCTSACBot
from dlabalone.agent.mcts_with_critic import MCTSBotCritic
from dlabalone.agent.network_naive import NetworkNaiveBot
from dlabalone.agent.random_kill_first import RandomKillBot
from dlabalone.agent.alphabeta import AlphaBetaBot
from dlabalone.agent.mcts import MCTSBot
from dlabalone.encoders.base import get_encoder_by_name
from dlabalone.networks.base import prepare_tf_custom_objects
from dlabalone.utils import print_board, encode_board_str, profiler
from keras.models import load_model
import tensorflow as tf
import cProfile
import random
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

draw_name = 'Draw'


def test1():
    game = GameState.new_game(5)
    # bot = RandomKillBot()
    # bot = AlphaBetaBot(depth=3, width=100)
    # bot = MCTSBot(name='MCTS', num_rounds=1000, temperature=0.1)
    # bot = MCTSBot(name='MCTS20000r0.01t', num_rounds=20000, temperature=0.01)
    # bot = network_bot_generator()
    # bot = mcts_with_critic_bot_generator()
    # prepare_tf_custom_objects()
    # encoder = get_encoder_by_name('fourplane', 5, None, data_format='channels_last')
    # actor = load_model(
    #     '../data/rl_mcts/generation00/ACSimple1Policy_dropout0.3_FourPlaneEncoder_channels_last_epoch_100.h5')
    # critic = load_model(
    #     '../data/rl_mcts/generation00/ACSimple1Value_dropout0.5_FourPlaneEncoder_channels_last_epoch_100.h5')
    # bot = MCTSACBot(encoder, actor, critic, name="bot_mcts_ac", width=3, num_rounds=300, temperature=0.01)
    # bot = mcts_ac_bot_generator2()
    bot = mcts_ac_bot_generator1()
    print(bot.name)

    step = 0
    max_depths = []
    start_time = time.time()
    while not game.is_over():
        stat = {}
        game = game.apply_move(bot.select_move(game, stat=stat))
        # max_depths.append(stat['max_depth'])
        print(f'step: {step}')
        step += 1
        if step == 3:
            break
        # if step % 1 == 0:
        #     print(f'{step}')
        #     print(f"{'----' * 5} {step} steps {'----' * 5}")
        #     print_board(game.board)
    print_board(game.board)
    print('====' * 10)
    runtime = time.time() - start_time
    print(f'step: {step}, time: {runtime}, {runtime / step} sec/step')
    print(max_depths)


def run_game(idx, bot_pair):
    print(f'new worker {idx}')
    bot_black, bot_white = bot_pair
    if hasattr(bot_black, '__call__'):
        bot_black = bot_black()
    if hasattr(bot_white, '__call__'):
        bot_white = bot_white()
    game = GameState.new_game(5)
    step = 0
    is_draw = False
    past_boards = {}
    runtime_black = 0
    runtime_white = 0
    game_start_time = time.time()
    while not game.is_over():
        if game.next_player == Player.black:
            start_time = time.time()
            move = bot_black.select_move(game)
            game = game.apply_move(move)
            runtime_black += time.time() - start_time
        else:
            start_time = time.time()
            move = bot_white.select_move(game)
            game = game.apply_move(move)
            runtime_white += time.time() - start_time
        step += 1
        # if step % 10 == 0:
        #     game_runtime = time.time() - game_start_time
        #     current_time = datetime.now().strftime('%y/%m/%d %H:%M:%S')
        #     print(f'({current_time}) {idx} step: {step} ({game_runtime/step} sec/step)', flush=True)

        # If a same board repeats too much, the game is draw
        board_str = encode_board_str(game.board)
        count = past_boards.get(board_str, 0)
        if count >= 5:
            is_draw = True
            break
        past_boards[board_str] = count + 1

        if step >= 10000:
            is_draw = True
            break
    print(f'worker {idx} finish. {bot_black.name}: {runtime_black} sec, {bot_white.name}: {runtime_white} sec')

    if is_draw:
        winner_name = draw_name
    elif game.winner() == Player.black:
        winner_name = bot_black.name
    else:
        winner_name = bot_white.name
    print(f'step:{step:4d}, winner: {winner_name}')
    return winner_name, step


def compare_bot(bot1, bot2, run_iter=100, threads=None):
    args = [(bot1, bot2), (bot2, bot1)] * (run_iter // 2)
    if run_iter % 2 == 1:
        args.append((bot1, bot2))

    if threads is None:
        threads = os.cpu_count()
    with multiprocessing.Pool(processes=threads) as p:
        result_list = p.starmap(run_game, enumerate(args))
    winner_list = list(map(lambda x: x[0], result_list))
    total_step = sum(map(lambda x: x[1], result_list))

    def print_win_rate(name):
        win_count = winner_list.count(name)
        print("%s: %d/%d (%3.3f%%)" % (name, win_count, run_iter, win_count / run_iter * 100))
    print(f'Total steps: {total_step}')

    if hasattr(bot1, '__call__'):
        bot1 = bot1()
    if hasattr(bot2, '__call__'):
        bot2 = bot2()

    print_win_rate(bot1.name)
    print_win_rate(bot2.name)
    print_win_rate(draw_name)


def network_bot_generator1():
    prepare_tf_custom_objects()
    encoder = get_encoder_by_name('fourplane', 5, None, data_format='channels_last')
    return NetworkNaiveBot(
        encoder, '../data/checkpoints/rl_mcts/gen00_policy_ACSimple1Policy_dropout0.3_FourPlaneEncoder_channels_last_epoch_100.h5',
        name='old_policy', selector='exponential')


def network_bot_generator2():
    prepare_tf_custom_objects()
    encoder = get_encoder_by_name('fourplane', 5, None, data_format='channels_last')
    return NetworkNaiveBot(
        encoder, '../data/checkpoints/rl_mcts/gen01_policy_new_only_ACSimple1Policy_dropout0.3_FourPlaneEncoder_channels_last_epoch_100.h5',
        name='new_new_only_policy_e100', selector='exponential')


def network_bot_generator3():
    prepare_tf_custom_objects()
    encoder = get_encoder_by_name('fourplane', 5, None, data_format='channels_last')
    return NetworkNaiveBot(
        encoder, '../data/checkpoints/rl_mcts/gen01_policy_old+new_ACSimple1Policy_dropout0.3_FourPlaneEncoder_channels_last_epoch_50.h5',
        name='new_after_old_policy_e42', selector='exponential')


def network_bot_generator4():
    prepare_tf_custom_objects()
    encoder = get_encoder_by_name('fourplane', 5, None, data_format='channels_last')
    return NetworkNaiveBot(
        encoder, '../data/checkpoints/rl_mcts/gen01_policy_old+new_ACSimple1Policy_dropout0.3_FourPlaneEncoder_channels_last_epoch_100.h5',
        name='new_after_old_policy_e100', selector='exponential')


def mcts_ac_bot_generator1():
    prepare_tf_custom_objects()
    # tf.compat.v1.disable_eager_execution()
    encoder = get_encoder_by_name('fourplane', 5, None, data_format='channels_last')
    actor = load_model(
        '../data/rl_mcts/generation00/ACSimple1Policy_dropout0.3_FourPlaneEncoder_channels_last_epoch_100.h5')
    critic = load_model(
        '../data/rl_mcts/generation00/ACSimple1Value_dropout0.5_FourPlaneEncoder_channels_last_epoch_100.h5')
    return MCTSACBot(encoder, actor, critic, name="bot_mcts_ac_w3_r3000", width=3, num_rounds=3000, temperature=0.01)


def mcts_ac_bot_generator2():
    prepare_tf_custom_objects()
    # tf.compat.v1.disable_eager_execution()
    encoder = get_encoder_by_name('fourplane', 5, None, data_format='channels_last')
    actor = load_model(
        '../data/rl_mcts/generation00/ACSimple1Policy_dropout0.3_FourPlaneEncoder_channels_last_epoch_100.h5')
    critic = load_model(
        '../data/rl_mcts/generation00/ACSimple1Value_dropout0.5_FourPlaneEncoder_channels_last_epoch_100.h5')
    return MCTSACBot(encoder, actor, critic, name="bot_mcts_ac_wdynamic_r3000", width='dynamic', num_rounds=3000, temperature=0.01)
    # return MCTSACBot(encoder, actor, critic, name="bot_mcts_ac_w3_r3000", width=3, num_rounds=3000, temperature=0.01)


def mcts_ac_bot_generator3():
    prepare_tf_custom_objects()
    # tf.compat.v1.disable_eager_execution()
    encoder = get_encoder_by_name('fourplane', 5, None, data_format='channels_last')
    actor = load_model(
        '../data/rl_mcts/generation00/ACSimple1Policy_dropout0.3_FourPlaneEncoder_channels_last_epoch_100.h5')
    critic = load_model(
        '../data/rl_mcts/generation00/ACSimple1Value_dropout0.5_FourPlaneEncoder_channels_last_epoch_100.h5')
    return MCTSACBot(encoder, actor, critic, name="bot_mcts_ac_wdynamic_r2000", width='dynamic', num_rounds=2000, temperature=0.01)
    # return MCTSACBot(encoder, actor, critic, name="bot_mcts_ac_w3_r3000", width=3, num_rounds=3000, temperature=0.01)


if __name__ == '__main__':
    # random.seed(0)
    # np.random.seed(0)
    # tf.random.set_seed(0)
    enable_profile = True
    print(f"Profile: {str(enable_profile)}")
    tf.debugging.disable_traceback_filtering()
    tf.compat.v1.disable_eager_execution()

    pr = []
    if enable_profile:
        pr.append(cProfile.Profile())
        pr[0].enable()

    ####################################
    profiler.start('game')

    # if True:
    if False:
    #     while True:
        test1()
    else:
        # bot_B = MCTSBot(name='MCTS20000r0.01t', num_rounds=20000, temperature=0.01)
        # bot_A = mcts_ac_bot_generator2
        # bot_B = mcts_ac_bot_generator3
        bot_A = network_bot_generator2
        bot_B = network_bot_generator2
        # compare_bot(bot_A, bot_B, run_iter=10, threads=2)
        compare_bot(bot_A, bot_B, run_iter=50, threads=2)

    profiler.end('game')
    profiler.print('game')

    if enable_profile:
        pr[0].disable()
        pr[0].dump_stats('profile.pstat')
