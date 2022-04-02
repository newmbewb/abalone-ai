import multiprocessing
import os

from dlabalone.ablboard import GameState
from dlabalone.abltypes import Player
from dlabalone.agent.random_kill_first import RandomKillBot
from dlabalone.agent.alphabeta import AlphaBetaBot
from dlabalone.agent.mcts import MCTSBot
from dlabalone.utils import print_board
from dlabalone.tests.test_utils import profiler
import cProfile
import random


def test1():
    game = GameState.new_game(5)
    # bot = RandomKillBot()
    # bot = AlphaBetaBot(depth=4, width=10)
    bot = MCTSBot(name='MCTS', num_rounds=1000, temperature=1.5)
    step = 0
    profiler.start('game')
    while not game.is_over():
        game = game.apply_move(bot.select_move(game))
        step += 1
        # if step % 100 == 0:
        #     print(f"{'----' * 5} {step} steps {'----' * 5}")
        #     print_board(game.board)
    profiler.end('game')
    print_board(game.board)
    print('====' * 10)
    print(step)
    profiler.print('game')


def run_game(bot_black, bot_white):
    game = GameState.new_game(5)
    step = 0
    while not game.is_over():
        if game.next_player == Player.black:
            game = game.apply_move(bot_black.select_move(game))
        else:
            game = game.apply_move(bot_white.select_move(game))
        step += 1

    if game.winner() == Player.black:
        bot_winner = bot_black
    else:
        bot_winner = bot_white
    print(f'step: {step}, winner: {bot_winner.name}')
    return bot_winner.name, step


def compare_bot(bot1, bot2, run_iter=100, threads=None):
    args = [(bot1, bot2), (bot2, bot1)] * (run_iter // 2)
    if run_iter % 2 == 1:
        args.append((bot1, bot2))

    if threads is None:
        threads = os.cpu_count()
    with multiprocessing.Pool(processes=threads) as p:
        result_list = p.starmap(run_game, args)
    winner_list = list(map(lambda x: x[0], result_list))
    total_step = sum(map(lambda x: x[1], result_list))

    def print_win_rate(bot):
        win_count = winner_list.count(bot.name)
        print("%s: %d/%d (%3.3f%%)" % (bot.name, win_count, run_iter, win_count / run_iter * 100))
    print(f'Total steps: {total_step}')
    print_win_rate(bot1)
    print_win_rate(bot2)


if __name__ == '__main__':
    random.seed(0)
    enable_profile = False
    pr = None
    if enable_profile:
        pr = cProfile.Profile()
        pr.enable()

    profiler.start('game')

    if False:
        test1()
    else:
        bot_A = MCTSBot(name='MCTS1000r1.5t', num_rounds=1000, temperature=1.5)
        bot_B = MCTSBot(name='MCTS1000r0.1t', num_rounds=1000, temperature=0.1)
        compare_bot(bot_A, bot_B, run_iter=10, threads=3)

    profiler.end('game')
    profiler.print('game')

    if enable_profile:
        pr.disable()
        pr.dump_stats('profile.pstat')
