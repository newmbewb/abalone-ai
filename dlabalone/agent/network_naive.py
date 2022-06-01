import copy
import random

import numpy as np

from dlabalone.ablboard import GameState
from dlabalone.abltypes import Player
from dlabalone.agent.base import Agent
from dlabalone.encoders.base import Encoder
from dlabalone.networks.base import Network
from dlabalone.rl.move_selectors.base import MoveSelector
from dlabalone.rl.move_selectors.eps_greedy import EpsilonGreedyMoveSelector
from dlabalone.rl.move_selectors.exponential import ExponentialMoveSelector
from dlabalone.utils import print_board
from keras.models import load_model


__all__ = ['NetworkNaiveBot']
MAX_SCORE = 100000000
MIN_SCORE = -1 * MAX_SCORE


class NetworkNaiveBot(Agent):
    def __init__(self, encoder: Encoder, model, selector="exponential", name=None):
        super().__init__(name)
        self.encoder = encoder

        # Load model
        if isinstance(model, str):
            self.model = load_model(model)
        else:
            self.model = model

        # Load selector
        if isinstance(selector, str):
            if selector == 'exponential':
                self.selector = ExponentialMoveSelector(fixed_exponent=3)
            elif selector == 'greedy':
                self.selector = EpsilonGreedyMoveSelector(temperature=0)
            else:
                assert False, 'Wrong selector name'
        elif isinstance(selector, MoveSelector):
            self.selector = selector
        else:
            assert False, "selector should be string or MoveSelector"

    def select_move(self, game_state: GameState):
        assert not game_state.is_over()
        moves_kill, moves_normal = game_state.legal_moves(separate_kill=True)
        if len(moves_kill) > 0 and self.can_last_attack(game_state):
            return moves_kill[0]
        encoded_state = self.encoder.encode_board(game_state.board, game_state.next_player, moves_kill=moves_kill)
        move_probs = self.model.predict(np.array([encoded_state]))[0]
        move, prob = self.selector(self.encoder, move_probs, moves_kill + moves_normal)
        return move

    @staticmethod
    def can_last_attack(game_state):
        if game_state.next_player == Player.black:
            if game_state.board.dead_stones_white == 5:
                return True
        elif game_state.next_player == Player.white:
            if game_state.board.dead_stones_black == 5:
                return True
        return False
