from dlabalone.ablboard import Board, Move, GameState
from dlabalone.abltypes import Direction, Player
import os
from dlabalone.utils import print_board, encode_board_str, load_file_board_move_pair, save_file_board_move_pair
import numpy as np


class GamePopulator:
    def __init__(self, board_size):
        self.board_size = board_size
        Board.set_size(board_size)
        self.center = Board.coord_xy2index((board_size - 1, board_size - 1))

        # Initialize rotation_map
        self.rotation_map = {}
        for index in Board.valid_grids:
            new_index = self._rotate_point(index)
            self.rotation_map[index] = new_index

    ###################################
    # Mirror game methods
    def mirror_point(self, point):
        x, y = Board.coord_index2xy(point)
        a = self.board_size - 1 + y
        new_x = a - x
        return Board.coord_xy2index((new_x, y))

    def mirror_move(self, move):
        new_move_stones = []
        for stone in move.stones:
            new_move_stones.append(self.mirror_point(stone))
        if move.direction == Direction.NORTHWEST.value:
            new_move_direction = Direction.NORTHEAST.value
        elif move.direction == Direction.NORTHEAST.value:
            new_move_direction = Direction.NORTHWEST.value
        elif move.direction == Direction.EAST.value:
            new_move_direction = Direction.WEST.value
        elif move.direction == Direction.SOUTHEAST.value:
            new_move_direction = Direction.SOUTHWEST.value
        elif move.direction == Direction.SOUTHWEST.value:
            new_move_direction = Direction.SOUTHEAST.value
        elif move.direction == Direction.WEST.value:
            new_move_direction = Direction.EAST.value
        else:
            assert False, 'Do not reach here'
        return Move(new_move_stones, new_move_direction)

    def mirror_board(self, board):
        new_board = Board()
        for point, player in board.grid.items():
            new_point = self.mirror_point(point)
            new_board.grid[new_point] = player
        return new_board

    def mirror_pair_list(self, pair_list, with_value):
        ret = []
        if with_value:
            for board, move, advantage, value in pair_list:
                ret.append((self.mirror_board(board), self.mirror_move(move), advantage, value))
        else:
            for board, move in pair_list:
                ret.append((self.mirror_board(board), self.mirror_move(move)))
        return ret

    ###################################
    # Rotate numpy methods
    def rotate_numpy_board(self, board):
        new_board = np.array(board)
        for index in Board.valid_grids:
            new_index = self.rotation_map[index]
            x, y = Board.coord_index2xy(index)
            new_x, new_y = Board.coord_index2xy(new_index)
            new_board[new_y, new_x] = board[y, x]
        return new_board

    def rotate_numpy_encoded(self, encoded):
        ret = np.zeros(encoded.shape)
        for i, board in enumerate(encoded):
            ret[i] = self.rotate_numpy_board(board)
        return ret

    ###################################
    # Rotate game methods
    def rotate_point_one_step(self, point):
        if point == self.center:
            return point
        x, y = Board.coord_index2xy(point)
        x_center, y_center = Board.coord_index2xy(self.center)
        if x < x_center and x >= y:
            part = 1
        elif x >= x_center and y < y_center:
            part = 2
        elif y >= y_center and x > y:
            part = 3
        elif x > x_center and x <= y:
            part = 4
        elif x <= x_center and y > y_center:
            part = 5
        elif y <= y_center and x < y:
            part = 6
        else:
            assert False, 'Do not reach here'

        if part == 1:
            direction = Direction.EAST
        elif part == 2:
            direction = Direction.SOUTHEAST
        elif part == 3:
            direction = Direction.SOUTHWEST
        elif part == 4:
            direction = Direction.WEST
        elif part == 5:
            direction = Direction.NORTHWEST
        elif part == 6:
            direction = Direction.NORTHEAST
        else:
            assert False, 'Do not reach here'

        return point + direction.value

    def distance(self, point):
        if point == self.center:
            return 0
        x, y = Board.coord_index2xy(point)
        x_center, y_center = Board.coord_index2xy(self.center)
        if x <= x_center and y <= y_center or x >= x_center and y >= y_center:
            return max(abs(x - x_center), abs(y - y_center))
        else:
            return abs(x - x_center) + abs(y - y_center)

    def _rotate_point(self, point):
        dist = self.distance(point)
        for _ in range(dist):
            point = self.rotate_point_one_step(point)
        return point

    def _rotate_point_fast(self, point):
        return self.rotation_map[point]

    def rotate_point(self, point, count):
        for _ in range(count):
            point = self._rotate_point_fast(point)
        return point

    def _rotate_board(self, board):
        new_board = Board()
        for point, player in board.grid.items():
            new_point = self.rotate_point(point, 1)
            new_board.grid[new_point] = player
        return new_board

    def rotate_board(self, board, count):
        for _ in range(count):
            board = self._rotate_board(board)
        return board

    def _rotate_move(self, move):
        new_move_stones = []
        for stone in move.stones:
            new_move_stones.append(self.rotate_point(stone, 1))
        if move.direction == Direction.NORTHWEST.value:
            new_move_direction = Direction.NORTHEAST.value
        elif move.direction == Direction.NORTHEAST.value:
            new_move_direction = Direction.EAST.value
        elif move.direction == Direction.EAST.value:
            new_move_direction = Direction.SOUTHEAST.value
        elif move.direction == Direction.SOUTHEAST.value:
            new_move_direction = Direction.SOUTHWEST.value
        elif move.direction == Direction.SOUTHWEST.value:
            new_move_direction = Direction.WEST.value
        elif move.direction == Direction.WEST.value:
            new_move_direction = Direction.NORTHWEST.value
        else:
            assert False, 'Do not reach here'
        return Move(new_move_stones, new_move_direction)

    def rotate_move(self, move, count):
        for _ in range(count):
            move = self._rotate_move(move)
        return move

    def validate_rotate(self, _game):
        pair_list = load_file_board_move_pair(_game)
        board_start = pair_list[0][0]
        board_final = pair_list[-1][0]
        for rotate_count in range(1, 6):
            board = self.rotate_board(board_start, rotate_count)
            game = GameState.new_game(5)
            game.board = board
            for pair in pair_list[:-1]:
                move = self.rotate_move(pair[1], rotate_count)
                game = game.apply_move(move)
            rotate_board_final = self.rotate_board(board_final, rotate_count)
            if encode_board_str(game.board) != encode_board_str(rotate_board_final):
                print('Wrong!!')
                print_board(game.board)
                print('='*20)
                print_board(rotate_board_final)

    def validate(self, _game, with_value):
        pair_list = load_file_board_move_pair(_game, with_value=with_value)
        board_start = pair_list[0][0]
        board_final = pair_list[-1][0]
        for rotate_count in range(1):
            board = self.rotate_board(board_start, rotate_count)
            game = GameState.new_game(5)
            game.board = board
            step = 0
            for pair in pair_list[:-1]:
                move = self.rotate_move(pair[1], rotate_count)
                game = game.apply_move(move)
                step += 1
            rotate_board_final = self.rotate_board(board_final, rotate_count)
            rotate_board_final_str = encode_board_str(rotate_board_final)
            if step % 2 == 1:
                rotate_board_final_str = rotate_board_final_str.replace('o', 'r')
                rotate_board_final_str = rotate_board_final_str.replace('x', 'o')
                rotate_board_final_str = rotate_board_final_str.replace('r', 'x')
            if encode_board_str(game.board) != rotate_board_final_str:
                print('Wrong!!')
                print_board(game.board)
                print('='*20)
                print_board(rotate_board_final)

    def rotate_pair_list(self, pair_list, count, with_value):
        ret = []
        if with_value:
            for board, move, advantage, value in pair_list:
                ret.append((self.rotate_board(board, count), self.rotate_move(move, count), advantage, value))
        else:
            for board, move in pair_list:
                ret.append((self.rotate_board(board, count), self.rotate_move(move, count)))
        return ret

    def populate_games(self, in_dir_paths, populated_game_name, with_value=False):
        idx = 0
        for in_path in in_dir_paths:
            game_files = [f for f in map(lambda x: os.path.join(in_path, x), os.listdir(in_path)) if os.path.isfile(f)]
            print(f'Populating directory {in_path}, file count: {len(game_files)}')
            file_count = 0
            for game_name in game_files:
                file_count += 1
                if file_count % 100 == 0:
                    print(f'{file_count}/{len(game_files)} Done..')
                # if 'draw' in game_name:
                #     continue
                pair_list = load_file_board_move_pair(game_name, with_value)
                for rotate_count in range(6):
                    # Rotate
                    rotated_pair_list = self.rotate_pair_list(pair_list, rotate_count, with_value)
                    save_file_board_move_pair(populated_game_name % idx, rotated_pair_list, with_value)
                    # self.validate(populated_game_name % idx, with_value)
                    idx += 1
                    # Mirror
                    mirrored_pair_list = self.mirror_pair_list(rotated_pair_list, with_value)
                    save_file_board_move_pair(populated_game_name % idx, mirrored_pair_list, with_value)
                    # self.validate(populated_game_name % idx, with_value)
                    idx += 1
