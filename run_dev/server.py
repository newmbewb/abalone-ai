import asyncio
import time
from datetime import datetime
import pathlib
import ssl
import websockets
from sys import executable
import os

from keras.models import load_model

from dlabalone.ablboard import Board, GameState, Move
from dlabalone.abltypes import Player, Direction
from dlabalone.agent.alphabeta import AlphaBetaBot
from dlabalone.agent.mcts import MCTSBot
from dlabalone.agent.mcts_ac import MCTSACBot
from dlabalone.agent.random_kill_first import RandomKillBot
from dlabalone.utils import encode_state_to_str, encode_board_str, decode_board_from_str
from dlabalone.encoders.base import get_encoder_by_name
from dlabalone.agent.network_naive import NetworkNaiveBot
from dlabalone.networks.base import prepare_tf_custom_objects


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


board_size = 5
max_xy = 9
update_delay = 1
server_global_lock = asyncio.Lock()
bot_mcts_ac_r1000 = None
bot_mcts_ac_r2000 = None


def index2xy(index):
    x = index % max_xy
    y = index // max_xy
    return x, y


def num2direction(num):
    nums = [-1, max_xy*(-1)-1, max_xy*(-1), 1, max_xy+1, max_xy]
    index = nums.index(num)
    return Direction.from_int(index)


async def send(websocket, movable, board, new_holes='8', new_stones='8'):
    await websocket.send(':'.join([movable, board, new_stones, new_holes]))


def get_before_and_after(move):
    stones_before = move.stones
    stones_after = list(map(lambda x: x + move.direction, stones_before))
    stones_before = list(set(stones_before) - set(stones_after))
    stones_after = ','.join(map(str, stones_after))
    stones_before = ','.join(map(str, stones_before))
    return stones_before, stones_after


async def self_play(websocket):
    data = await websocket.recv()
    win_msg = None
    if data.split(':')[1] == 'record':
        print(f'{str(datetime.now())}: {data}', flush=True)
    elif 'black:start' in data or 'white:start' in data:
        print(f'{str(datetime.now())}: new game (black); ' + data, flush=True)
        game = GameState.new_game(board_size, reverse=True)
        await send(websocket, 'true', encode_board_str(game.board, Player.black))
    else:
        tag, player, board, selected, direction = data.split(':')

        # Decode board
        board = decode_board_from_str(board, max_xy)
        if player == 'black':
            next_player = Player.black
        else:
            next_player = Player.white
        game = GameState(board, next_player)

        # Decode Move
        stones = map(int, selected.split(','))
        move = Move(stones, num2direction(int(direction)))
        game = game.apply_move(move)
        await send(websocket, 'true', encode_board_str(game.board, Player.black))
        winner = game.winner()
        if winner:
            if winner == Player.black:
                win_msg = f"{tag}:win:black"
                await websocket.send('false:black_win')
            else:
                win_msg = f"{tag}:win:white"
                await websocket.send('false:white_win')
        else:
            t = time.time()
            winner = game.winner()
            # await asyncio.sleep(update_delay - (time.time() - t))
            if winner:
                await send(websocket, 'false', encode_board_str(game.board, Player.black))
                if winner == Player.black:
                    await websocket.send('false:black_win')
                else:
                    await websocket.send('false:white_win')
    print(data, flush=True)
    if win_msg:
        print(win_msg)


async def accept(websocket, path):
    global server_global_lock
    global bot_mcts_ac_r1000, bot_mcts_ac_r2000
    async with server_global_lock:
        if path == '/mcts':
            bot = MCTSBot(name='MCTS20000r0.01t', num_rounds=20000, temperature=0.01)
        elif path == '/self':
            return await self_play(websocket)
        elif path == '/ab3':
            bot = AlphaBetaBot(depth=3)
            # bot = RandomKillBot()
        elif path == '/mcts_ac_r1000':
            bot = bot_mcts_ac_r1000
        elif path == '/mcts_ac_r2000':
            bot = bot_mcts_ac_r2000
        else:
            bot = None

        data = await websocket.recv()
        win_msg = None
        if data.split(':')[1] == 'record':
            print(f'{str(datetime.now())}: {data}', flush=True)
        elif 'black:start' in data:
            reverse = True
            if 'flip_board' in data:
                reverse = not reverse
            print(f'{str(datetime.now())}: new game (black); ' + data, flush=True)
            game = GameState.new_game(board_size, reverse=reverse)
            await send(websocket, 'true', encode_board_str(game.board, Player.black))
        elif 'white:start' in data:
            reverse = False
            if 'flip_board' in data:
                reverse = not reverse
            print(f'{str(datetime.now())}: new game (white); ' + data, flush=True)
            game = GameState.new_game(board_size, reverse=reverse)
            await send(websocket, 'false', encode_board_str(game.board, Player.black))
            t = time.time()
            move = bot.select_move(game)
            stones_before, stones_after = get_before_and_after(move)
            game = game.apply_move(move)
            await asyncio.sleep(update_delay - (time.time() - t))
            await send(websocket, 'true', encode_board_str(game.board, Player.black), stones_before, stones_after)
        else:
            tag, player, board, selected, direction = data.split(':')

            # Decode board
            board = decode_board_from_str(board, max_xy)
            if player == 'black':
                next_player = Player.black
            else:
                next_player = Player.white
            game = GameState(board, next_player)

            # Decode Move
            stones = map(int, selected.split(','))
            move = Move(stones, num2direction(int(direction)))
            game = game.apply_move(move)
            # await send(websocket, 'false', encode_board_str(game.board, Player.black))
            winner = game.winner()
            if winner:
                if winner == Player.black:
                    win_msg = f"{tag}:win:black"
                    await websocket.send('false:black_win')
                else:
                    win_msg = f"{tag}:win:white"
                    await websocket.send('false:white_win')
            else:
                t = time.time()
                move = bot.select_move(game)
                stones_before, stones_after = get_before_and_after(move)
                game = game.apply_move(move)
                winner = game.winner()
                await asyncio.sleep(update_delay - (time.time() - t))
                if winner:
                    await send(websocket, 'false', encode_board_str(game.board, Player.black), stones_before,
                               stones_after)
                    if winner == Player.black:
                        await websocket.send('false:black_win')
                    else:
                        await websocket.send('false:white_win')
                else:
                    await send(websocket, 'true', encode_board_str(game.board, Player.black), stones_before,
                               stones_after)
        print(data, flush=True)
        if win_msg:
            print(win_msg)


def mcts_ac_bot_generator_gen01(width=3, num_rounds=3000):
    prepare_tf_custom_objects()
    # tf.compat.v1.disable_eager_execution()
    encoder = get_encoder_by_name('fourplane', 5, None, data_format='channels_last')
    actor = load_model(
        '../data/rl_mcts/generation01_manual/policy_model.h5')
    critic = load_model(
        '../data/rl_mcts/generation01_manual/value_model.h5')
    bot = MCTSACBot(encoder, actor, critic, name=f"bot_mcts_ac_w{width}_r{num_rounds}_gen01", width=width,
                    num_rounds=num_rounds, temperature=0.01)
    bot.train = False
    return bot


def mcts_ac_bot_generator_gen02(width=3, num_rounds=3000):
    prepare_tf_custom_objects()
    # tf.compat.v1.disable_eager_execution()
    encoder = get_encoder_by_name('fourplane', 5, None, data_format='channels_last')
    actor = load_model(
        '../data/rl_mcts/generation02_manual/policy_model.h5')
    critic = load_model(
        '../data/rl_mcts/generation01_manual/value_model.h5')
    bot = MCTSACBot(encoder, actor, critic, name=f"bot_mcts_ac_w{width}_r{num_rounds}_gen02", width=width,
                    num_rounds=num_rounds, temperature=0.01)
    bot.train = False
    return bot


def main():
    global bot_mcts_ac_r1000, bot_mcts_ac_r2000
    bot_mcts_ac_r1000 = mcts_ac_bot_generator_gen02(num_rounds=1000)
    bot_mcts_ac_r2000 = mcts_ac_bot_generator_gen02(num_rounds=2000)
    start_server = websockets.serve(accept, "0.0.0.0", 9000)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


if __name__ == '__main__':
    print(executable)
    main()
