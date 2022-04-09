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


def encode_board_str(board):
    max_xy = board.max_xy
    board_char = [['.' for _ in range(max_xy)] for _ in range(max_xy)]
    char_black = 'o'
    char_white = 'x'
    for xy, player in board.grid.items():
        x, y = xy
        if player == Player.black:
            board_char[y][x] = char_black
        elif player == Player.white:
            board_char[y][x] = char_white
        else:
            assert False
    return ''.join(map(lambda c: ''.join(c), board_char))

