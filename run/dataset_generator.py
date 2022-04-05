import multiprocessing
import os
from dlabalone.ablboard import GameState
from dlabalone.abltypes import Player
from dlabalone.agent.random_kill_first import RandomKillBot
from dlabalone.agent.alphabeta import AlphaBetaBot
from dlabalone.agent.mcts import MCTSBot
from dlabalone.utils import print_board
from dlabalone.tests.test_utils import profiler, save_file_board_move_pair, load_file_board_move_pair
import cProfile
import random
import sys


draw_name = 'Draw'
global_filename = '../generated_games/game_%d.txt'


def test1():
    game = GameState.new_game(5)
    bot = RandomKillBot()
    # bot = AlphaBetaBot(depth=3, width=100)
    # bot = MCTSBot(name='MCTS', num_rounds=4000, temperature=0.1)
    step = 0
    profiler.start('game')
    pair_list = []
    while not game.is_over():
        move = bot.select_move(game)
        pair_list.append((game, move))
        game = game.apply_move(move)
        step += 1
    profiler.end('game')
    print_board(game.board)
    print('====' * 10)
    print(step)
    profiler.print('game')
    save_file_board_move_pair('test.txt', pair_list)


def visualize_game(infile, outfile):
    pair_list = load_file_board_move_pair(infile)
    old_stdout = sys.stdout
    sys.stdout = open(outfile, 'w')
    for board, move in pair_list:
        print_board(board)
        print('-----------------')
    sys.stdout = old_stdout


def run_game(idx, bot_pair):
    bot_black, bot_white = bot_pair
    game = GameState.new_game(5)
    step = 0
    is_draw = False
    pair_list = []
    while not game.is_over():
        if game.next_player == Player.black:
            move = bot_black.select_move(game)
        else:
            move = bot_white.select_move(game)
        pair_list.append((game, move))
        game = game.apply_move(move)
        step += 1
        if step > 3000:
            is_draw = True
            break
    save_file_board_move_pair(global_filename % idx, pair_list)

    if is_draw:
        winner_name = draw_name
    elif game.winner() == Player.black:
        winner_name = bot_black.name
    else:
        winner_name = bot_white.name
    print(f'step:{step:4d}, winner: {winner_name}')
    return winner_name, step


def generate_data(bot1, bot2, run_iter=100, threads=None):
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
    print_win_rate(bot1.name)
    print_win_rate(bot2.name)
    print_win_rate(draw_name)


if __name__ == '__main__':
    random.seed(0)

    bot_A = MCTSBot(name='MCTS4000r0.0001t', num_rounds=4000, temperature=0.001)
    bot_B = MCTSBot(name='MCTS4000r0.0001t', num_rounds=4000, temperature=0.001)
    global_filename = '../generated_games/mcts/game_%d.txt'
    generate_data(bot_A, bot_B, 100, 3)
    # visualize_game('../generated_games/game_0.txt', 'vis.txt')
