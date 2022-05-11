import numpy as np

from dlabalone.rl.move_selectors.base import MoveSelector


class ExponentialMoveSelector(MoveSelector):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.temp_max = 1
        self.temp_min = 0
        self.max_exponential = 1

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
        move_probs = move_probs ** ((1 - self.temperature) * self.max_exponential)
        eps = 1e-6
        num_moves = encoder.num_moves()

        legal_moves_index = []
        for move in legal_moves:
            legal_moves_index.append(encoder.encode_move(move))

        move_probs = np.clip(move_probs, eps, 1-eps)
        move_probs = move_probs / np.sum(move_probs)
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)
        for move_idx in ranked_moves:
            if move_idx in legal_moves_index:
                return encoder.decode_move_index(move_idx)
