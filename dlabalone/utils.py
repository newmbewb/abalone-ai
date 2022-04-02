from dlabalone.abltypes import Player


def print_board(board):
    max_xy = type(board).max_xy
    for y in range(max_xy):
        indent = abs(board.size - 1 - y)
        print(' ' * indent, end='')
        for x in range(max_xy):
            point = x, y
            if not board.is_on_grid(point):
                continue
            player = board.grid.get(point, None)
            if player is None:
                print('. ', end='')
            if player == Player.black:
                print('O ', end='')
            if player == Player.white:
                print('X ', end='')
        print('')
