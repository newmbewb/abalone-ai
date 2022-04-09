import multiprocessing
import os
from dlabalone.ablboard import GameState
from dlabalone.abltypes import Player
from dlabalone.agent.random_kill_first import RandomKillBot
from dlabalone.agent.alphabeta import AlphaBetaBot
from dlabalone.agent.mcts import MCTSBot
from dlabalone.utils import print_board, encode_board_str
from dlabalone.tests.test_utils import profiler, save_file_board_move_pair, load_file_board_move_pair
import cProfile
import random
import sys
from datetime import datetime


draw_name = 'Draw'
date = datetime.now().strftime('%y%m%d%H%M%S')
global_filename = f'../generated_games/mcts/game_MCTS20000r0.01t_{date}_%d-%s.txt'


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
    past_boards = {}
    while not game.is_over():
        if game.next_player == Player.black:
            move = bot_black.select_move(game)
        else:
            move = bot_white.select_move(game)
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

    #########
    # print(f'Max repeated board: {max(past_boards.values())}')
    if is_draw:
        draw_str = '_draw'
    else:
        draw_str = ''
    filename = global_filename % (idx, draw_str)
    save_file_board_move_pair(filename, pair_list)

    if is_draw:
        winner_name = draw_name
    elif game.winner() == Player.black:
        winner_name = bot_black.name
    else:
        winner_name = bot_white.name
    current_time = datetime.now().strftime('%y/%m/%d %H:%M:%S')
    print(f'({current_time})  step:{step:4d}, winner: {winner_name}')
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
    bot_A = MCTSBot(name='MCTS20000r0.01t', num_rounds=20000, temperature=0.01)
    bot_B = MCTSBot(name='MCTS20000r0.01t', num_rounds=20000, temperature=0.01)

    generate_data(bot_A, bot_B, 1000, 3)
    # visualize_game('../generated_games/game_0.txt', 'vis.txt')
