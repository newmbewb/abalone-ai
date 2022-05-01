import multiprocessing
import os
import time
from dlabalone.ablboard import GameState
from dlabalone.abltypes import Player
from dlabalone.agent.mcts import MCTSBot
from dlabalone.utils import encode_board_str, save_file_state_move_pair
import threading
from datetime import datetime
from sys import argv


draw_name = 'Draw'
# date = datetime.now().strftime('%y%m%d%H%M%S')
# global_filename = f'../generated_games/mcts/game_MCTS20000r0.01t_{date}_%d-%s.txt'
global_timer = None
draw_repeat_count = 30


def kill_self():
    print('Livelock detected.', flush = True)
    exit(1)


def stop_timer():
    global global_timer
    if global_timer is not None:
        global_timer.cancel()


def reset_timer(t):
    global global_timer
    if global_timer is not None:
        global_timer.cancel()
    global_timer = threading.Timer(t, kill_self)
    global_timer.start()


def run_game(idx, bot_pair):
    bot_black, bot_white = bot_pair
    game = GameState.new_game(5)
    step = 0
    is_draw = False
    pair_list = []
    past_boards = {}
    while not game.is_over():
        reset_timer(300)
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
        if count >= draw_repeat_count:
            is_draw = True
            break

        if step >= 1000:
            is_draw = True
            break
    stop_timer()

    #########
    if is_draw:
        draw_str = '_draw'
    else:
        draw_str = ''
    filename = global_filename % (idx, draw_str)
    save_file_state_move_pair(filename, pair_list)

    if is_draw:
        winner_name = draw_name
    elif game.winner() == Player.black:
        winner_name = bot_black.name
    else:
        winner_name = bot_white.name
    current_time = datetime.now().strftime('%y/%m/%d %H:%M:%S')
    print(f'({current_time})  step:{step:4d}, winner: {winner_name}', flush=True)
    return winner_name, step


def generate_data(bot1, bot2, run_iter):
    for i in range(run_iter):
        run_game(i, (bot1, bot2))
        tmp = bot1
        bot1 = bot2
        bot2 = tmp


if __name__ == '__main__':
    if len(argv) < 2:
        print(f'Usage: {argv[0]} <name>')
        exit(1)
    date = datetime.now().strftime('%y%m%d%H%M%S')
    global_filename = f'../data/generated_games/mcts/game_MCTS20000r0.01t_{date}_{argv[1]}_%d-%s.txt'
    print(f'Start {argv[1]} script: draw_repeat_count {draw_repeat_count}', flush=True)
    bot_A = MCTSBot(name='MCTS20000r0.01t', num_rounds=20000, temperature=0.01)
    bot_B = MCTSBot(name='MCTS20000r0.01t', num_rounds=20000, temperature=0.01)

    generate_data(bot_A, bot_B, 100000)
