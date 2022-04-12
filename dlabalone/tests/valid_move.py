import random

from dlabalone.ablboard import Board, GameState
from dlabalone.abltypes import Player
from dlabalone.utils import save_file_move_list, load_file_move_list, is_same_move_set
from dlabalone.utils import print_board


def test1():
    board = Board(5)
    # blacks = [(0, 1), (1, 1), (2, 1),
    #           (1, 2), (2, 2)]
    # white = [(2, 3), (3, 4),
    #           (3, 1), (4, 1)]
    blacks = [(0, 1), (1, 1), (2, 1),
              (0, 3), (0, 4),
              (2, 6), (3, 7)]
    whites = [(3, 1), (4, 1),
              (1, 3),
              (4, 8)]
    for point in blacks:
        board.grid[point] = Player.black
    for point in whites:
        board.grid[point] = Player.white

    game = GameState(board, Player.black)
    correct = load_file_move_list('data/ut_valid_move_1_correct.txt')
    if not is_same_move_set(correct, game.legal_moves()):
        print('Test failed: Valid move 1')
    # print_board(board)
    # print(len(game.legal_moves()))
    # print('====' * 10)
    #
    # for move in game.legal_moves():
    #     next_game = game.apply_move(move)
    #     print_board(next_game.board)
    #     print('----' * 10)


if __name__ == '__main__':
    test1()
