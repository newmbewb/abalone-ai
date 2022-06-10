from dlabalone.abltypes import Player, Direction
import copy

from dlabalone.encoders.plane_generator import count_unique_push_moves, count_opp_sumito

_win_rate = list()
_win_rate.append([0.5, 0.435231234, 0.330374753, 0.202020202, 0.025316456, 0])
_win_rate.append([0.564768766, 0.5, 0.407775872, 0.277913534, 0.080802603, 0.010663199])
_win_rate.append([0.6696252469999999, 0.592224128, 0.5, 0.389032225, 0.1937105, 0.020094433])
_win_rate.append([0.797979798, 0.7220864659999999, 0.610967775, 0.5, 0.318860244, 0.083547778])
_win_rate.append([0.974683544, 0.919197397, 0.8062895, 0.681139756, 0.5, 0.24539859])
_win_rate.append([1, 0.989336801, 0.979905567, 0.916452222, 0.75460141, 0.5])


class Board(object):
    max_xy = 9
    size = 5
    point2distance = None
    valid_grids = []

    @classmethod
    def coord_index2xy(cls, point_int):
        x = point_int % cls.max_xy
        y = point_int // cls.max_xy
        return x, y

    @classmethod
    def coord_xy2index(cls, xy):
        x = xy[0]
        y = xy[1]
        return y * cls.max_xy + x

    @classmethod
    def _distance_from_center(cls, point):
        point = Board.coord_index2xy(point)
        center = (cls.size - 1, cls.size - 1)
        if point == center:
            return 0
        x, y = point
        x_center, y_center = center
        if x <= x_center and y <= y_center or x >= x_center and y >= y_center:
            return max(abs(x - x_center), abs(y - y_center))
        else:
            return abs(x - x_center) + abs(y - y_center)

    @classmethod
    def distance_from_center(cls, point):
        return cls.point2distance[point]

    @classmethod
    def set_size(cls, size):
        cls.size = size
        cls.max_xy = size * 2 - 1
        cls.valid_grids = []
        for index in range(cls.max_xy * cls.max_xy):
            if cls.is_on_grid(index):
                cls.valid_grids.append(index)

        # Set point2distance
        if cls.point2distance is None:
            cls.point2distance = {}
            for point in range(cls.max_xy * cls.max_xy):
                cls.point2distance[point] = cls._distance_from_center(point)

    def __init__(self, grid=None, dead_stones_black=0, dead_stones_white=0):
        if grid is None:
            self.grid = {}
        else:
            self.grid = grid
        self.dead_stones_black = dead_stones_black
        self.dead_stones_white = dead_stones_white


    def __deepcopy__(self, memodict={}):
        return Board(copy.copy(self.grid), self.dead_stones_black, self.dead_stones_white)

    def _move_single_stone(self, stone, direction, stones_to_move, undo_move=None):
        stones_to_move.discard(stone)
        owner = self.grid[stone]
        if undo_move is not None:
            undo_move.append((stone, owner))
        new_place = stone + direction

        if not self.is_on_grid(new_place):
            # The stone dies
            if owner == Player.black:
                self.dead_stones_black += 1
            else:
                self.dead_stones_white += 1
            del self.grid[stone]
            return

        if new_place in self.grid:
            # Another stone is there
            self._move_single_stone(new_place, direction, stones_to_move, undo_move)
        else:
            if undo_move is not None:
                undo_move.append((new_place, None))

        del self.grid[stone]
        self.grid[new_place] = owner

    def move_stones(self, stones, direction, undo_move=None):
        stones_to_move = set(stones)
        while stones_to_move:
            stone = stones_to_move.pop()
            self._move_single_stone(stone, direction, stones_to_move, undo_move)

    @classmethod
    def is_on_grid(cls, point):
        x, y = Board.coord_index2xy(point)
        if x < 0 or y < 0:
            return False
        elif x >= cls.max_xy or y >= cls.max_xy:
            return False
        elif abs(x - y) >= cls.size:
            return False
        else:
            return True


class GameState(object):
    def __init__(self, board, next_player):
        self.board = board
        self.next_player = next_player

    def is_same(self, other_state):
        if self.next_player != other_state.next_player:
            return False
        board = self.board
        other_board = other_state.board
        if board.dead_stones_white != other_board.dead_stones_white or \
                board.dead_stones_black != other_board.dead_stones_black:
            return False
        if len(board.grid) != len(other_board.grid):
            return False
        for stone, owner in board.grid.items():
            other_owner = other_board.grid.get(stone, None)
            if other_owner != owner:
                return False
        return True

    def is_valid_move(self, stones, direction):
        next_player = self.next_player
        # Check whether all stones belong to next_player
        for stone in stones:
            player = self.board.grid.get(stone, None)
            if player != next_player:
                return False

        # Check whether stones are in single file
        stone_count = len(stones)
        string_direction = None
        if stone_count == 1:
            pass
        elif stone_count == 2:
            string_direction = stones[0] - stones[1]
            if not Direction.is_valid(string_direction):
                return False
        elif stone_count == 3:
            stones.sort()
            direction1 = stones[0] - stones[1]
            direction2 = stones[1] - stones[2]
            if direction1 != direction2:
                return False
            string_direction = direction1
            if not Direction.is_valid(string_direction):
                return False
        else:
            return False

        return self._can_move(stones, string_direction, direction)

    def _can_move(self, stones, string_direction, direction, kill=None, push=None):
        string_len = len(stones)
        if kill is None:
            kill = [False]
        if push is None:
            push = [False]
        kill[0] = False
        if string_len > 1 and (direction == string_direction or direction == (string_direction * -1)):
            cursor = stones[0]
            owner = self.board.grid[cursor]
            opp_len = 0
            while True:
                cursor = cursor + direction
                if cursor in stones:
                    continue
                elif not self.board.is_on_grid(cursor):
                    if opp_len == 0:
                        return False
                    else:
                        kill[0] = True
                        return True
                elif cursor not in self.board.grid:
                    return True
                elif self.board.grid[cursor] == owner:
                    return False
                elif self.board.grid[cursor] == owner.other:
                    opp_len += 1
                    push[0] = True
                    if opp_len >= string_len:
                        return False
                else:
                    assert False, "Cannot reach here"
        else:
            for stone in stones:
                new_place = stone + direction
                if not self.board.is_on_grid(new_place):
                    return False
                if new_place in self.board.grid:
                    return False
            return True

    def apply_move_lite(self, move):
        dead_stone_white = self.board.dead_stones_white
        dead_stone_black = self.board.dead_stones_black
        undo_grid = []
        self.next_player = self.next_player.other
        self.board.move_stones(move.stones, move.direction, undo_grid)
        return dead_stone_white, dead_stone_black, undo_grid

    def undo(self, undo_move):
        self.board.dead_stones_white, self.board.dead_stones_black, undo_grid = undo_move
        self.next_player = self.next_player.other
        for stone, owner in reversed(undo_grid):
            if owner is not None:
                self.board.grid[stone] = owner
            else:
                del self.board.grid[stone]

    def apply_move(self, move):
        next_board = copy.deepcopy(self.board)
        next_board.move_stones(move.stones, move.direction)
        return GameState(next_board, self.next_player.other)

    @classmethod
    def new_game(cls, board_size, reverse=False):
        Board.set_size(board_size)
        board = Board()
        if reverse:
            player_up = Player.white
            player_down = Player.black
        else:
            player_up = Player.black
            player_down = Player.white
        # Black
        for x in range(0, 4 + 1):
            point = Board.coord_xy2index((x, 0))
            board.grid[point] = player_up
        for x in range(0, 5 + 1):
            point = Board.coord_xy2index((x, 1))
            board.grid[point] = player_up
        for x in range(2, 4 + 1):
            point = Board.coord_xy2index((x, 2))
            board.grid[point] = player_up

        # White
        for x in range(4, 6 + 1):
            point = Board.coord_xy2index((x, 6))
            board.grid[point] = player_down
        for x in range(3, 8 + 1):
            point = Board.coord_xy2index((x, 7))
            board.grid[point] = player_down
        for x in range(4, 8 + 1):
            point = Board.coord_xy2index((x, 8))
            board.grid[point] = player_down

        return GameState(board, Player.black)

    def is_over(self):
        if self.board.dead_stones_black >= 6 or self.board.dead_stones_white >= 6:
            return True
        return False

    def legal_moves(self, separate_kill=False, push_moves=None):
        ret_kill = []
        ret_normal = []
        if push_moves is None:
            push_moves = []
        for point in type(self.board).valid_grids:
            player = self.board.grid.get(point, None)
            if player != self.next_player:
                continue

            # For single stone move
            for direction in Direction:
                next_point = point + direction.value
                if self.board.is_on_grid(next_point) and next_point not in self.board.grid:
                    ret_normal.append(Move([point], direction.value))

            # For multiple stone move
            for string_direction in [Direction.EAST.value, Direction.SOUTHEAST.value, Direction.SOUTHWEST.value]:
                for length in range(2, 3+1):
                    # Make stone list
                    stones = [point]
                    next_point = point
                    for i in range(1, length):
                        next_point = next_point + string_direction
                        stones.append(next_point)

                    # Check stones' player
                    valid = True
                    for stone in stones[1:]:
                        if self.next_player != self.board.grid.get(stone, None):
                            valid = False
                            break
                    if not valid:
                        break

                    for move_direction in Direction:
                        move_direction = move_direction.value
                        kill = [False]
                        push = [False]
                        valid = self._can_move(stones, string_direction, move_direction, kill, push)
                        if valid:
                            if kill[0]:
                                ret_kill.append(Move(stones, move_direction))
                            else:
                                ret_normal.append(Move(stones, move_direction))
                            if push[0]:
                                push_moves.append(Move(stones, move_direction))
        if separate_kill:
            return ret_kill, ret_normal
        else:
            return ret_kill + ret_normal

    def winner(self):
        if self.board.dead_stones_black >= 6:
            return Player.white
        elif self.board.dead_stones_white >= 6:
            return Player.black
        return None

    def win_probability_naive(self, player):
        black_remain = 6 - self.board.dead_stones_black
        white_remain = 6 - self.board.dead_stones_white
        if player == Player.white:
            return white_remain / (white_remain + black_remain)
        else:
            return black_remain / (white_remain + black_remain)

    def win_probability_v1(self, player):
        if player == Player.black:
            my_kill_score = self.board.dead_stones_white
            opp_kill_score = self.board.dead_stones_black
        else:
            my_kill_score = self.board.dead_stones_black
            opp_kill_score = self.board.dead_stones_white
        if my_kill_score >= 6:
            return 1
        if opp_kill_score >= 6:
            return 1
        return _win_rate[my_kill_score][opp_kill_score]

    @staticmethod
    def _distance2score(distance):
        if distance < 24:
            return 400
        if distance < 30:
            return 300
        if distance < 35:
            return 200
        if distance < 40:
            return 100
        return 0

    @staticmethod
    def _grouping2score(grouping):
        if grouping > 55:
            return 320
        if grouping > 50:
            return 240
        if grouping > 45:
            return 160
        if grouping > 40:
            return 80
        return 0

    def pascal_score(self, player, next_push_moves):
        winner = self.winner()
        if winner == player:
            return 1000000
        elif winner == player.other:
            return -1000000
        # Lost marble score
        score_lost_marble = (self.board.dead_stones_white - self.board.dead_stones_black) * 1000
        if player == Player.white:
            score_lost_marble *= -1

        # Center distance score
        distance_black = 0
        distance_white = 0
        for point, player in self.board.grid.items():
            if player == Player.black:
                distance_black += self.board.distance_from_center(point)
            if player == Player.white:
                distance_white += self.board.distance_from_center(point)
        score_distance = self._distance2score(distance_black) - self._distance2score(distance_white)

        # Grouping score
        grouping_black = 0
        grouping_white = 0
        for point, player in self.board.grid.items():
            group_count = 0
            for d in Direction:
                neighbor = self.board.grid.get(point + d.value, None)
                if player == neighbor:
                    group_count += 1
            if player == Player.black:
                grouping_black += group_count
            if player == Player.white:
                grouping_white += group_count
        score_grouping = self._grouping2score(grouping_black) - self._grouping2score(grouping_white)

        # Attack score
        if self.next_player == player:
            score_push = (count_unique_push_moves(next_push_moves) - count_opp_sumito(self.board, player.other)) * 40
        else:
            score_push = (count_opp_sumito(self.board, player) - count_unique_push_moves(next_push_moves)) * 40

        return score_lost_marble + score_distance + score_grouping + score_push



    def can_last_attack(self):
        if self.next_player == Player.black:
            if self.board.dead_stones_white == 5:
                return True
        elif self.next_player == Player.white:
            if self.board.dead_stones_black == 5:
                return True
        return False


class Move(object):
    def __init__(self, stones, direction):
        self.stones = stones
        self.direction = direction

    def __eq__(self, other):
        my_stones = copy.copy(self.stones)
        other_stones = copy.copy(other.stones)
        my_stones.sort()
        other_stones.sort()
        if my_stones == other_stones and self.direction == other.direction:
            return True
        else:
            return False

    def __str__(self):
        str_list = [str(Direction.to_int(self.direction))]
        for stone in self.stones:
            x, y = Board.coord_index2xy(stone)
            str_list.append('%d,%d' % (x, y))
        return ':'.join(str_list)

    @staticmethod
    def str_to_move(arg):
        tokens = arg.split(':')
        direction = Direction.from_int(int(tokens[0]))
        stones = []
        for stone_str in tokens[1:]:
            x, y = stone_str.split(',')
            point = Board.coord_xy2index((int(x), int(y)))
            stones.append(point)
        return Move(stones, direction)
