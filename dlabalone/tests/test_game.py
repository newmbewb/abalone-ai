import multiprocessing
import os

from dlabalone.ablboard import GameState
from dlabalone.abltypes import Player
from dlabalone.agent.random_kill_first import RandomKillBot
from dlabalone.agent.alphabeta import AlphaBetaBot
from dlabalone.utils import print_board
from dlabalone.tests.test_utils import profiler
import cProfile
import random


def test1():
    game = GameState.new_game(5)
    # bot = RandomKillBot()
    bot = AlphaBetaBot(depth=3)
    step = 0
    profiler.start('game')
    while not game.is_over():
        # game = game.apply_move(random.choice(game.legal_moves()))
        game = game.apply_move(bot.select_move(game))
        step += 1
        # if step % 20 == 0:
        #     print_board(game.board)
        #     print('----' * 10)
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
        return bot_black.name
    else:
        return bot_white.name


def compare_bot(bot1, bot2, run_iter=100, threads=None):
    args = [(bot1, bot2), (bot2, bot1)] * (run_iter // 2)
    if run_iter % 2 == 1:
        args.append((bot1, bot2))

    if threads is None:
        threads = os.cpu_count()
    with multiprocessing.Pool(processes=threads) as p:
        winner_list = p.starmap(run_game, args)

    def print_win_rate(bot):
        win_count = winner_list.count(bot.name)
        print("%s: %d/%d (%3.3f%%)"%(bot.name, win_count, run_iter, win_count / run_iter * 100))
    print_win_rate(bot1)
    print_win_rate(bot2)


if __name__ == '__main__':
    random.seed(0)
    pr = cProfile.Profile()
    pr.enable()

    profiler.start('game')

    test1()
    # compare_bot(AlphaBetaBot("ABBot2", depth=2), AlphaBetaBot("ABBot3", depth=3), run_iter=8)

    profiler.end('game')
    profiler.print('game')

    pr.disable()
    pr.dump_stats('profile.pstat')
