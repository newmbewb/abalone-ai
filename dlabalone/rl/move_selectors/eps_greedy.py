import random
import numpy as np

from dlabalone.rl.move_selectors.base import MoveSelector


class ExponentialMoveSelector(MoveSelector):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.temp_max = 1
        self.temp_min = 0.02

    def temperature_up(self):
        self.temperature += 0.01
        self.temperature = min(self.temperature, self.temp_max)

    def temperature_down(self):
        self.temperature -= 0.01
        self.temperature = max(self.temperature, self.temp_min)

    def is_over(self):
        if self.temperature <= 0:
            return True
        else:
            return False

    def __call__(self, encoder, move_probs, legal_moves):
        if random.random() < self.temperature:
            return random.choice(legal_moves)

        legal_moves_index = []
        for move in legal_moves:
            legal_moves_index.append(encoder.encode_move(move))

        s = np.argsort(move_probs)
        for move_index in s[::-1]:
            if move_index in legal_moves_index:
                return encoder.decode_move_index(move_index)
        assert False, "Something... Something is wrong!!"