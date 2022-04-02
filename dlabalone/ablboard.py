from dlabalone.abltypes import Player, Direction, add, sub, mul
import copy


class Board(object):
    def __init__(self, size):
        self.size = size
        self.max_xy = size * 2 - 1
        self.grid = {}
        self.dead_stones = {Player.black: 0, Player.white: 0}
        self.valid_grids = []
        for y in range(self.max_xy):
            for x in range(self.max_xy):
                if self.is_on_grid((x, y)):
                    self.valid_grids.append((x, y))

    def _move_single_stone(self, stone, direction, stones_to_move):
        stones_to_move.discard(stone)
        owner = self.grid[stone]
        new_place = add(stone, direction)

        if not self.is_on_grid(new_place):
            # The stone dies
            self.dead_stones[owner] += 1
            del self.grid[stone]
            return

        if new_place in self.grid:
            # Another stone is there
            self._move_single_stone(new_place, direction, stones_to_move)

        del self.grid[stone]
        self.grid[new_place] = owner

    def move_stones(self, stones, direction):
        stones_to_move = set(stones)
        while stones_to_move:
            stone = stones_to_move.pop()
            self._move_single_stone(stone, direction, stones_to_move)

    def is_on_grid(self, point):
        x, y = point
        if x < 0 or y < 0:
            return False
        elif x >= self.max_xy or y >= self.max_xy:
            return False
        elif abs(x - y) >= self.size:
            return False
        else:
            return True


class GameState(object):
    def __init__(self, board, next_player):
        self.board = board
        self.next_player = next_player

    def is_valid_move(self, stones, direction):
        # Check whether all stones are on grid
        for stone in stones:
            if not self.board.is_on_grid(stone):
                return False

        # Check whether stones are in single file
        stone_count = len(stones)
        string_direction = None
        if stone_count == 1:
            pass
        elif stone_count == 2:
            string_direction = sub(stones[0], stones[1])
            if not Direction.is_valid(string_direction):
                return False
        elif stone_count == 3:
            stones.sort()
            direction1 = sub(stones[0], stones[1])
            direction2 = sub(stones[1], stones[2])
            if direction1 != direction2:
                return False
            string_direction = direction1
            if not Direction.is_valid(string_direction):
                return False
        else:
            return False

        return self._can_move(stones, string_direction, direction)

    def _can_move(self, stones, string_direction, direction, kill=None):
        string_len = len(stones)
        if kill is None:
            kill = [False]
        kill[0] = False
        if string_len > 1 and (direction == string_direction or direction == mul(string_direction, -1)):
            cursor = stones[0]
            owner = self.board.grid[cursor]
            opp_len = 0
            while True:
                cursor = add(cursor, direction)
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
                    if opp_len >= string_len:
                        return False
                else:
                    assert False, "Cannot reach here"
        else:
            for stone in stones:
                new_place = add(stone, direction)
                if not self.board.is_on_grid(new_place):
                    return False
                if new_place in self.board.grid:
                    return False
            return True

    def apply_move(self, move):
        next_board = copy.deepcopy(self.board)
        next_board.move_stones(move.stones, move.direction)
        return GameState(next_board, self.next_player.other)

    @classmethod
    def new_game(cls, board_size):
        board = Board(board_size)
        # Black
        for x in range(0, 4 + 1):
            point = x, 0
            board.grid[point] = Player.black
        for x in range(0, 5 + 1):
            point = x, 1
            board.grid[point] = Player.black
        for x in range(2, 4 + 1):
            point = x, 2
            board.grid[point] = Player.black

        # White
        for x in range(4, 6 + 1):
            point = x, 6
            board.grid[point] = Player.white
        for x in range(3, 8 + 1):
            point = x, 7
            board.grid[point] = Player.white
        for x in range(4, 8 + 1):
            point = x, 8
            board.grid[point] = Player.white

        return GameState(board, Player.black)

    def is_over(self):
        for dead_stones in self.board.dead_stones.values():
            if dead_stones >= 6:
                return True
        return False

    def legal_moves(self, separate_kill=False):
        ret_kill = []
        ret_normal = []
        for x, y in self.board.valid_grids:
            point = (x, y)
            player = self.board.grid.get(point, None)
            if player != self.next_player:
                continue

            # For single stone move
            for direction in Direction:
                next_point = add(point, direction.value)
                if self.board.is_on_grid(next_point) and next_point not in self.board.grid:
                    ret_normal.append(Move([point], direction.value))

            # For multiple stone move
            for string_direction in [Direction.EAST.value, Direction.SOUTHEAST.value, Direction.SOUTHWEST.value]:
                for length in range(2, 3+1):
                    # Make stone list
                    stones = [point]
                    next_point = point
                    for i in range(1, length):
                        next_point = add(next_point, string_direction)
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
                        valid = self._can_move(stones, string_direction, move_direction, kill)
                        if valid:
                            if kill[0]:
                                ret_kill.append(Move(stones, move_direction))
                            else:
                                ret_normal.append(Move(stones, move_direction))
        if separate_kill:
            return ret_kill, ret_normal
        else:
            return ret_kill + ret_normal

    def winner(self):
        for player, dead_stones in self.board.dead_stones.items():
            if dead_stones >= 6:
                return player.other
        return None


class Move(object):
    def __init__(self, stones, direction):
        self.stones = stones
        self.direction = direction

    def __eq__(self, other):
        my_stones = copy.deepcopy(self.stones)
        other_stones = copy.deepcopy(other.stones)
        my_stones.sort()
        other_stones.sort()
        if my_stones == other_stones and self.direction == other.direction:
            return True
        else:
            return False

    def __str__(self):
        str_list = [str(Direction.to_int(self.direction))]
        for stone in self.stones:
            str_list.append('%d,%d' % stone)
        return ':'.join(str_list)

    @staticmethod
    def str_to_move(arg):
        tokens = arg.split(':')
        direction = Direction.from_int(int(tokens[0]))
        stones = []
        for stone_str in tokens[1:]:
            x, y = stone_str.split(',')
            stones.append((int(x), int(y)))
        return Move(stones, direction)
