import random
import numpy as np

from dlabalone.rl.move_selectors.base import MoveSelector


class EpsilonGreedyMoveSelector(MoveSelector):
    def __init__(self, temperature=0.5):
        super().__init__(temperature=temperature)

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
