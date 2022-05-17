import multiprocessing
import os

from dlabalone.ablboard import GameState
from dlabalone.abltypes import Player
from dlabalone.agent.network_naive import NetworkNaiveBot
from dlabalone.agent.random_kill_first import RandomKillBot
from dlabalone.agent.alphabeta import AlphaBetaBot
from dlabalone.agent.mcts import MCTSBot
from dlabalone.encoders.base import get_encoder_by_name
from dlabalone.utils import print_board, encode_board_str, profiler
import cProfile
import random


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


draw_name = 'Draw'


def test1():
    game = GameState.new_game(5)
    # bot = RandomKillBot()
    # bot = AlphaBetaBot(depth=3, width=100)
    # bot = MCTSBot(name='MCTS', num_rounds=1000, temperature=0.1)
    bot = MCTSBot(name='MCTS20000r0.01t', num_rounds=20000, temperature=0.01)
    step = 0
    profiler.start('game')
    while not game.is_over():
        game = game.apply_move(bot.select_move(game))
        step += 1
        if step == 20:
            break
        # if step % 1 == 0:
        #     print(f'{step}')
        #     print(f"{'----' * 5} {step} steps {'----' * 5}")
        #     print_board(game.board)
    profiler.end('game')
    print_board(game.board)
    print('====' * 10)
    print(step)
    profiler.print('game')


def run_game(idx, bot_pair):
    bot_black, bot_white = bot_pair
    if hasattr(bot_black, '__call__'):
        bot_black = bot_black()
    if hasattr(bot_white, '__call__'):
        bot_white = bot_white()
    game = GameState.new_game(5)
    step = 0
    is_draw = False
    past_boards = {}
    while not game.is_over():
        if game.next_player == Player.black:
            game = game.apply_move(bot_black.select_move(game))
        else:
            game = game.apply_move(bot_white.select_move(game))
        step += 1

        # If a same board repeats too much, the game is draw
        board_str = encode_board_str(game.board)
        count = past_boards.get(board_str, 0)
        if count >= 5:
            is_draw = True
            break
        past_boards[board_str] = count + 1

        if step >= 1000:
            is_draw = True
            break

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


def network_bot_generator():
    encoder = get_encoder_by_name('fourplane', 5, None, data_format='channels_last')
    return NetworkNaiveBot(
        encoder, '../../data/checkpoints/models/ACSimple1Policy_dropout0.1_FourPlaneEncoder_channels_last_epoch_13.h5',
        selector='greedy')


if __name__ == '__main__':
    random.seed(0)
    enable_profile = False
    print(f"Profile: {str(enable_profile)}")
    pr = None
    if enable_profile:
        pr = cProfile.Profile()
        pr.enable()

    ####################################
    profiler.start('game')

    # if True:
    if False:
        test1()
    else:
        bot_A = MCTSBot(name='MCTS20000r0.01t', num_rounds=4000, temperature=0.001)
        bot_B = network_bot_generator
        compare_bot(bot_A, bot_B, run_iter=10, threads=2)

    profiler.end('game')
    profiler.print('game')

    if enable_profile:
        pr.disable()
        pr.dump_stats('profile.pstat')
