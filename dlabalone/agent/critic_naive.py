import math
import random

import numpy as np

from dlabalone.abltypes import Player
from dlabalone.agent.base import Agent
from dlabalone.rl.move_selectors.base import MoveSelector
from dlabalone.rl.move_selectors.eps_greedy import EpsilonGreedyMoveSelector
from dlabalone.rl.move_selectors.exponential import ExponentialMoveSelector
from keras.models import load_model


__all__ = [
    'CriticNaiveBot',
]


class CriticNaiveBot(Agent):
    def __init__(self, encoder, critic, selector='base', randomness=0.1, name=None):
        super().__init__(name)
        self.encoder = encoder
        self.randomness = randomness
        self.selector = selector
        if isinstance(critic, str):
            self.critic = load_model(critic)
        else:
            self.critic = critic

    def select_move(self, game_state):
        moves = game_state.legal_moves()
        next_game_states = []

        for move in moves:
            next_game_states.append((game_state.apply_move(move), move))

        # Find move that finish game
        end_move = None
        for state, move in next_game_states:
            if state.is_over():
                end_move = move
                break

        if end_move is not None:
            return end_move

        # Score all next state
        encoded_board = []
        for state, move in next_game_states:
            encoded_board.append(self.encoder.encode_board(state.board, state.next_player))

        predict_output = self.critic.predict_on_batch(np.array(encoded_board))

        ###################### Select one
        if self.selector == 'base':
            # Get candidates
            max_value = np.max(predict_output)
            index_list = []
            value_list = []
            for index, value in enumerate(predict_output):
                if value > max_value - self.randomness * 2:
                    index_list.append(index)
                    value_list.append(value)

            # Run random
            value_list = np.array(value_list)
            value_list = (value_list + 1) / 2
            value_list = value_list ** 3
            eps = 1e-6
            probs = np.clip(value_list, eps, 1 - eps).flatten()
            probs /= np.sum(probs)

            selected_index = np.random.choice(index_list, 1, replace=False, p=probs)[0]
            return next_game_states[selected_index][1]
        elif self.selector == 'greedy':
            selected_index = np.argmax(predict_output)
            return next_game_states[selected_index][1]
