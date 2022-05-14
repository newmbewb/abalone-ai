import numpy as np

from dlabalone.rl.move_selectors.base import MoveSelector


class ExponentialMoveSelector(MoveSelector):
    def __init__(self, temperature=0.5):
        super().__init__(temperature=temperature)
        self.max_exponential = 1

    def __call__(self, encoder, move_probs, legal_moves):
        move_probs = move_probs ** ((1 - self.temperature) * self.max_exponential)
        move_probs = np.nan_to_num(move_probs)
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
