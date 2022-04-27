import asyncio
import time
from datetime import datetime

import websockets
from sys import executable
import os

from dlabalone.ablboard import Board, GameState, Move
from dlabalone.abltypes import Player, Direction
from dlabalone.agent.alphabeta import AlphaBetaBot
from dlabalone.agent.mcts import MCTSBot
from dlabalone.utils import encode_state_to_str, encode_board_str, decode_board_from_str

board_size = 5
max_xy = 9
update_delay = 1


def index2xy(index):
    x = index % max_xy
    y = index // max_xy
    return x, y


def num2direction(num):
    nums = [-1, max_xy*(-1)-1, max_xy*(-1), 1, max_xy+1, max_xy]
    index = nums.index(num)
    return Direction.from_int(index)


async def accept(websocket, path):
    if path == '/mcts':
        bot = MCTSBot(name='MCTS20000r0.01t', num_rounds=20000, temperature=0.01)
    elif path == '/ab3':
        bot = AlphaBetaBot(depth=3)
    else:
        bot = None

    data = await websocket.recv()
    if data == 'black:start':
        print(f'{str(datetime.now())}: new game (black)')
        game = GameState.new_game(board_size, reverse=True)
        await websocket.send('true:'+encode_board_str(game.board, Player.black))
    elif data == 'white:start':
        print(f'{str(datetime.now())}: new game (white)')
        game = GameState.new_game(board_size)
        await websocket.send('false:'+encode_board_str(game.board, Player.black))
        t = time.time()
        game = game.apply_move(bot.select_move(game))
        await asyncio.sleep(update_delay - (time.time() - t))
        await websocket.send('true:'+encode_board_str(game.board, Player.black))
    else:
        player, board, selected, direction = data.split(':')

        # Decode board
        board = decode_board_from_str(board, max_xy)
        if player == 'black':
            next_player = Player.black
        else:
            next_player = Player.white
        game = GameState(board, next_player)

        # Decode Move
        stones = map(lambda c: index2xy(int(c)), selected.split(','))
        move = Move(stones, num2direction(int(direction)))
        game = game.apply_move(move)
        await websocket.send('false:' + encode_board_str(game.board, Player.black))
        winner = game.winner()
        if winner:
            if winner == Player.black:
                await websocket.send('false:black_win')
            else:
                await websocket.send('false:white_win')
            return
        t = time.time()
        game = game.apply_move(bot.select_move(game))
        winner = game.winner()
        await asyncio.sleep(update_delay - (time.time() - t))
        if winner:
            if winner == Player.black:
                await websocket.send('false:black_win')
            else:
                await websocket.send('false:white_win')
            return
        else:
            await websocket.send('true:' + encode_board_str(game.board, Player.black))


def main():
    start_server = websockets.serve(accept, "0.0.0.0", 9000)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


if __name__ == '__main__':
    print(executable)
    main()
