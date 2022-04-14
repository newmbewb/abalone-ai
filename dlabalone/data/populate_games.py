from dlabalone.ablboard import Board, Move, GameState
from dlabalone.abltypes import Direction, add, Player
import os
from dlabalone.utils import print_board, encode_board_str, load_file_board_move_pair, save_file_board_move_pair


class GamePopulator:
    def __init__(self, board_size):
        self.board_size = board_size
        Board.set_size(board_size)
        self.center = (board_size - 1, board_size - 1)

    def rotate_point_one_step(self, point):
        if point == self.center:
            return point
        x, y = point
        x_center, y_center = self.center
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

        return add(point, direction.value)

    def distance(self, point):
        if point == self.center:
            return 0
        x, y = point
        x_center, y_center = self.center
        if x <= x_center and y <= y_center or x >= x_center and y >= y_center:
            return max(abs(x - x_center), abs(y - y_center))
        else:
            return abs(x - x_center) + abs(y - y_center)

    def _rotate_point(self, point):
        dist = self.distance(point)
        for _ in range(dist):
            point = self.rotate_point_one_step(point)
        return point

    def rotate_point(self, _point, count):
        point = tuple(_point)
        for _ in range(count):
            point = self._rotate_point(point)
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

    def validate(self, _game):
        pair_list = load_file_board_move_pair(_game)
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

    def rotate_pair_list(self, pair_list, count):
        ret = []
        for board, move in pair_list:
            ret.append((self.rotate_board(board, count), self.rotate_move(move, count)))
        return ret

    def populate_games(self, in_dir_paths, populated_game_name):
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
                pair_list = load_file_board_move_pair(game_name)
                for rotate_count in range(6):
                    rotated_pair_list = self.rotate_pair_list(pair_list, rotate_count)
                    save_file_board_move_pair(populated_game_name % idx, rotated_pair_list)
                    # validate(populated_game_name % idx)
                    idx += 1
