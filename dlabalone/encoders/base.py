import importlib
from dlabalone.ablboard import Board, Move
from dlabalone.abltypes import Direction, add

__all__ = [
    'Encoder',
    'get_encoder_by_name',
]


class Encoder:
    def __init__(self, board_size):
        self.max_xy = board_size * 2 - 1
        self.move_list = []  # index -> move
        self.move_dict = {}  # move -> index
        Board.set_size(board_size)

        # Generate move list
        for string_length in range(1, 4):
            if string_length == 1:
                string_directions = [(0, 0)]
            else:
                string_directions = list(map(lambda x: x.value,
                                             [Direction.EAST, Direction.SOUTHEAST, Direction.SOUTHWEST]))
            for head_point in Board.valid_grids:
                for string_direction in string_directions:
                    # Make stone list
                    invalid_move = False
                    stones = []
                    next_stone = head_point
                    for _ in range(string_length):
                        if next_stone not in Board.valid_grids:
                            invalid_move = True
                            # break
                        stones.append(next_stone)
                        next_stone = add(next_stone, string_direction)
                    if invalid_move:
                        continue
                    stones.sort()
                    for move_direction in map(lambda x: x.value, Direction):
                        # Check whether it is valid move direction
                        invalid_move = False
                        for stone in stones:
                            next_point = add(stone, move_direction)
                            if next_point not in Board.valid_grids:
                                invalid_move = True
                                break
                        if invalid_move:
                            continue
                        # Add move
                        self.move_list.append((tuple(stones), Direction.to_int(move_direction)))

        # Generate move dictionary
        for index, move in enumerate(self.move_list):
            self.move_dict[move] = index

    def name(self):
        raise NotImplementedError()

    def encode_board(self, game_board):
        raise NotImplementedError()

    def encode_move(self, move):
        stones = move.stones
        stones.sort()
        stones = tuple(stones)
        direction = Direction.to_int(move.direction)
        return self.move_dict[(stones, direction)]

    def decode_move_index(self, index):
        stones, direction = self.move_list[index]
        return Move(list(stones), Direction.from_int(direction))

    def num_moves(self):
        return len(self.move_list)

    def shape(self):
        raise NotImplementedError()


def get_encoder_by_name(name, board_size):
    module = importlib.import_module('dlabalone.encoders.' + name)
    constructor = getattr(module, 'create')
    return constructor(board_size)
