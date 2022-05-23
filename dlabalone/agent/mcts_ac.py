import math
import random

import numpy as np

from dlabalone.abltypes import Player
from dlabalone.agent.base import Agent
from dlabalone.agent.random_kill_first import RandomKillBot


__all__ = [
    'MCTSACBot',
]


class MCTSACNode(object):
    max_width = 5

    def __init__(self, game_state, parent=None):
        self.game_state = game_state
        self.next_player = game_state.next_player
        self.parent = parent
        self.critic_score = None
        self.score = {}
        self.explorable_moves = self.max_width
        self.num_rollouts = 0
        self.children = []
        self.unvisited_moves = None
        self.encoded_board = None

    @classmethod
    def set_max_width(cls, max_width):
        cls.max_width = max_width

    def update(self):
        assert self.critic_score is not None
        # Initialize
        self.num_rollouts = 1
        if self.unvisited_moves is None:
            self.explorable_moves = self.max_width
        else:
            self.explorable_moves = len(self.unvisited_moves)
        next_player = self.next_player
        if len(self.children) == 0:
            max_score = self.critic_score
        else:
            max_score = -1
            # Accumulate children stats
            for child in self.children:
                self.num_rollouts += child.num_rollouts
                self.explorable_moves += child.explorable_moves
                max_score = max(max_score, child.score[next_player])
        self.score[next_player] = max_score
        self.score[next_player.other] = -max_score

        if self.parent is not None:
            self.parent.update()

    def update_score(self, score):
        self.critic_score = score
        self.score[self.next_player] = score
        self.score[self.next_player.other] = -score
        self.parent.update()

    def update_move_list(self, encoder, move_probs):
        num_moves = encoder.num_moves()
        ranked_moves = np.random.choice(np.arange(num_moves), num_moves, replace=False, p=move_probs)
        self.unvisited_moves = []
        for move_idx in ranked_moves:
            move = encoder.decode_move_index(move_idx)
            if self.game_state.is_valid_move(move.stones, move.direction):
                self.unvisited_moves.append(move)

    def get_unvisited_children(self, count):
        ret = []
        for _ in range(count):
            new_move = self.unvisited_moves.pop(0)
            new_game_state = self.game_state.apply_move(new_move)
            new_node = MCTSACNode(new_game_state, self)
            self.children.append(new_node)
            ret.append(new_node)
        return ret

    def is_terminal(self):
        return self.game_state.is_over()


def _uct_score(score, temp, log_rollout, node_rollout):
    win_frac = (score + 1) / 2
    return win_frac + temp * math.sqrt(log_rollout / node_rollout)


class MCTSACBot(Agent):
    def __init__(self, encoder, actor, critic, name=None, num_rounds=5120, temperature=0.01, batch_size=128,
                 randomness=1):
        super().__init__(name)
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.batch_size = batch_size
        self.encoder = encoder
        self.actor = actor
        self.critic = critic
        self.randomness = randomness

    def _get_moves_to_eval(self, node, budget):
        '''
        :param node:
        :param budget:
        :return: actor_list, critic_list
          actor_list: The list of nodes which need to run the actor model
          critic_list(parent, count): The list of move to evaluate with the critic model
        '''
        # If we have to find candidate move for the node first
        if node.unvisited_moves is None:
            return [node], [(node, min(node.max_width, budget))]
        unvisited_len = len(node.unvisited_moves)

        if budget <= unvisited_len:
            return [], [(node, budget)]

        actor_list = []
        critic_list = []
        if unvisited_len > 0:
            critic_list.append((node, unvisited_len))
            budget -= unvisited_len

        # Calculate budget for each move
        # Optimize me!!!!
        next_player = node.next_player
        current_rollouts = [child.num_rollouts for child in node.children]
        scores = [child.score[next_player] for child in node.children]
        budget_for_children = [0 for _ in node.children]
        total_num_rollouts = node.num_rollouts
        while budget > 0:
            # Calculate UCT score
            log_total_num_rollouts = math.log(total_num_rollouts)
            max_score = -1
            max_index = -1
            for i, child in enumerate(node.children):
                if budget_for_children[i] >= child.explorable_moves:
                    continue
                score = _uct_score(scores[i], self.temperature, log_total_num_rollouts, current_rollouts[i])
                if score > max_score:
                    max_score = score
                    max_index = i
            if max_index < 0:
                break
            budget_for_children[max_index] += 1
            current_rollouts[max_index] += 1
            total_num_rollouts += 1
            budget -= 1

        # Get moves from each move
        for idx, child in enumerate(node.children):
            if budget_for_children[idx] == 0:
                pass
            al, ml = self._get_moves_to_eval(child, budget_for_children[idx])
            actor_list += al
            critic_list += ml

        return actor_list, critic_list

    def select_move(self, game_state):
        root = MCTSACNode(game_state)
        remain_rounds = self.num_rounds

        while remain_rounds > 0:
            # Get moves which we evaluate at this iteration
            actor_list, critic_list = self._get_moves_to_eval(root, self.batch_size)
            remain_rounds -= len(critic_list)

            # Run actor and get moves for un-actored nodes
            actor_input = np.array([node.encoded_board for node in actor_list])
            actor_output = self.actor.predict_on_batch(actor_input)
            for node, moves in zip(actor_list, actor_output):
                node.update_move_list(self.encoder, moves)

            # Calculate value for new nodes
            critic_nodes = []
            for parent, child_count in critic_list:
                critic_nodes += parent.get_unvisited_children(child_count)
            critic_input = []
            for node in critic_nodes:
                encoded_board = self.encoder.encode_board(node.game_state.board, node.next_player)
                node.encoded_board = encoded_board
                critic_input.append(encoded_board)
            critic_input = np.array(critic_input)
            critic_output = self.critic.predict_on_batch(critic_input)
            for node, value in critic_nodes, critic_output:
                node.update_score(value)
