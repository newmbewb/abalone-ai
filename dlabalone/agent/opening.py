import os
import random
import re
from pathlib import Path

from dlabalone.ablboard import Move
from dlabalone.abltypes import Direction
from dlabalone.agent.base import Agent


__all__ = ['OpeningBot']


class OpeningBot(Agent):
    def __init__(self, opening_dir, name=None, **kwargs):
        super().__init__(name)
        file_list = os.listdir(opening_dir)
        file = random.choice(file_list)
        fd = open(os.path.join(opening_dir, file))
        self.moves = []
        for line in fd:
            self.moves.append(Move.str_to_move(line.strip()))
        self.index = 0

    def select_move(self, game_state):
        next_move = self.moves[self.index]
        self.index += 1
        assert game_state.is_valid_move(next_move.stones, next_move.direction)
        return next_move

    def is_end(self):
        return self.index >= len(self.moves)


def _save_as_file(filename, moves):
    if len(moves) == 0:
        return
    fd = open(filename, 'w')
    for move in moves:
        fd.write(str(move) + '\n')
    fd.close()


def log2files(opening_dir, logfile):
    p = re.compile(r'[0-9.]+:(black|white):[.xo]+:(?P<stones>[0-9,]+):(?P<direction>[0-9\-]+)')
    Path(opening_dir).mkdir(parents=True, exist_ok=True)
    fd = open(logfile)
    moves = []
    file_index = 0
    filename = os.path.join(opening_dir, f'opening_{random.random()}_%d.txt')
    for line in fd:
        if 'new game' in line:
            _save_as_file(filename % file_index, moves)
            moves = []
            file_index += 1
            continue
        m = p.match(line)
        if m:
            stones = list(map(int, m.group('stones').split(',')))
            direction = int(m.group('direction'))
            move = Move(stones, Direction.from_value(direction))
            moves.append(move)
    _save_as_file(filename % file_index, moves)