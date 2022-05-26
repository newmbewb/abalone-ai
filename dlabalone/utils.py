from dlabalone.ablboard import Move, Board
import time
from dlabalone.abltypes import Player

move_list_delimiter = '&'
board_move_seperator = '&'
char_next_player = 'o'
char_other_player = 'x'


def is_same_move_set(list_a, list_b):
    if len(list_a) != len(list_b):
        return False
    for move_b in list_b:
        same_elem = False
        for move_a in list_a:
            if move_a == move_b:
                same_elem = True
                list_a.remove(move_a)
                break
        if not same_elem:
            return False
    assert len(list_a) == 0, 'Wrong code'
    return True


##################################
# Data Encode/Decode Functions
##################################
def encode_board_str(board, next_player=Player.black):
    max_xy = board.max_xy
    board_char = [['.' for _ in range(max_xy)] for _ in range(max_xy)]
    if next_player == Player.black:
        char_black = char_next_player
        char_white = char_other_player
    else:
        char_white = char_next_player
        char_black = char_other_player
    for index, player in board.grid.items():
        x, y = Board.coord_index2xy(index)
        if player == Player.black:
            board_char[y][x] = char_black
        elif player == Player.white:
            board_char[y][x] = char_white
        else:
            assert False
    return ''.join(map(lambda c: ''.join(c), board_char))


def encode_state_to_str(state):
    return encode_board_str(state.board, state.next_player)


def decode_board_from_str(board_str, max_xy, next_player=Player.black):
    if next_player == Player.black:
        char_black = char_next_player
        char_white = char_other_player
    else:
        char_white = char_next_player
        char_black = char_other_player
    black_stones = 0
    white_stones = 0
    grid = {}
    for index, c in enumerate(board_str):
        if c == char_black:
            grid[index] = Player.black
            black_stones += 1
        elif c == char_white:
            grid[index] = Player.white
            white_stones += 1
    return Board(grid, 14 - black_stones, 14 - white_stones)


##################################
# File I/O Functions
##################################
def load_file_move_list(filename):
    fd = open(filename, 'r')
    move_str_list = fd.read().split(move_list_delimiter)
    fd.close()
    move_list = []
    for move_str in move_str_list:
        move_list.append(Move.str_to_move(move_str))
    return move_list


def save_file_move_list(filename, move_list):
    fd = open(filename, 'w')
    move_str_list = move_list_delimiter.join(map(str, move_list))
    fd.write(move_str_list)
    fd.close()


def save_file_state_move_pair(filename, pair_list):
    fd = open(filename, 'w')
    # Save metadata
    fd.write(f'{pair_list[0][0].board.size},{len(pair_list)}\n')
    for state, move in pair_list:
        fd.write(f'{encode_state_to_str(state)}{board_move_seperator}{str(move)}\n')
    fd.close()


def save_file_board_move_pair(filename, pair_list, with_value=False):
    fd = open(filename, 'w')
    # Save metadata
    fd.write(f'{pair_list[0][0].size},{len(pair_list)}\n')
    if with_value:
        for board, move, advantage, value in pair_list:
            fd.write(f'{encode_board_str(board)}{board_move_seperator}{str(move)}' +
                     f'{board_move_seperator}{advantage}{board_move_seperator}{value}\n')
    else:
        for board, move in pair_list:
            fd.write(f'{encode_board_str(board)}{board_move_seperator}{str(move)}\n')
    fd.close()


def load_file_board_move_pair(filename, with_value=False, recover_player=False):
    fd = open(filename, 'r')
    size, length = fd.readline().split(',')
    Board.set_size(int(size))
    max_xy = Board.max_xy
    pair_list = []
    next_player = Player.black
    for line in fd:
        if with_value:
            board_str, move_str, advantage, value = line.strip().split(board_move_seperator)
            pair_list.append((decode_board_from_str(board_str, max_xy, next_player), Move.str_to_move(move_str),
                              float(advantage), float(value)))
        else:
            board_str, move_str = line.strip().split(board_move_seperator)
            pair_list.append((decode_board_from_str(board_str, max_xy, next_player), Move.str_to_move(move_str)))
        if recover_player:
            next_player = next_player.other
    fd.close()
    return pair_list


##################################
# Profiling Functions
##################################
class Profiler(object):
    def __init__(self):
        self.profile_start = {}
        self.profile_end = {}

    def start(self, name):
        self.profile_start[name] = time.time()

    def end(self, name):
        self.profile_end[name] = time.time()

    def print(self, name):
        print('Runtime (%s): %3.05f' % (name, self.profile_end[name] - self.profile_start[name]))


profiler = Profiler()


##################################
# Visualization Functions
##################################
def print_board(board):
    max_xy = type(board).max_xy
    for y in range(max_xy):
        indent = abs(board.size - 1 - y)
        print(' ' * indent, end='')
        for x in range(max_xy):
            point = Board.coord_xy2index((x, y))
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
