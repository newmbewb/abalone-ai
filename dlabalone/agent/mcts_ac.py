import json
import math
import random

import numpy as np
import tensorflow as tf

from dlabalone.abltypes import Player
from dlabalone.agent.base import Agent


__all__ = [
    'MCTSACBot',
]


eps = 1e-6


class MCTSACNode(object):
    max_width = 5
    min_width = 3

    def __init__(self, game_state, move, parent=None):
        self.game_state = game_state
        self.next_player = game_state.next_player
        self.move = move
        self.parent = parent
        self.critic_score = None
        self.score = {}
        self.explorable_moves = self.max_width
        self.num_rollouts = 1
        self.children = []
        self.unvisited_moves = None
        self.encoded_board = None

    @classmethod
    def set_max_width(cls, max_width):
        cls.max_width = max_width

    def update(self):
        assert self.critic_score is not None
        # Initialize
        if self.unvisited_moves is None:
            self.explorable_moves = self.max_width
        else:
            self.explorable_moves = len(self.unvisited_moves)
        next_player = self.next_player
        score_accum = self.critic_score
        self.num_rollouts = 1
        for child in self.children:
            self.num_rollouts += child.num_rollouts
            self.explorable_moves += child.explorable_moves
            score_accum += child.score[next_player] * child.num_rollouts
        score = score_accum / self.num_rollouts

        self.score[next_player] = score
        self.score[next_player.other] = -score

        if self.parent is not None:
            self.parent.update()

    def init_score(self, score):
        self.num_rollouts = 1
        winner = self.game_state.winner()
        if winner == self.next_player:
            score = 1
        elif winner == self.next_player.other:
            score = -1
        self.critic_score = score
        self.score[self.next_player] = score
        self.score[self.next_player.other] = -score

        # if self.parent is not None:
        #     self.parent.update()

    def get_valid_move_list(self, move_args, count, encoder):
        ret = []
        for move_index in move_args:
            move = encoder.decode_move_index(move_index)
            if self.game_state.is_valid_move(move.stones, move.direction):
                ret.append(move)
                if len(ret) >= count:
                    break
        return ret

    def update_unvisited_moves(self, encoder, move_probs):
        self.encoded_board = None  # Deleted not encoded board for memory reuse
        move_probs = np.clip(move_probs, eps, 1-eps)
        move_probs = move_probs / np.sum(move_probs)
        num_moves = encoder.num_moves()

        # We choose only self.max_width * 4 moves, because of performance
        # First choose try
        ranked_moves = np.random.choice(np.arange(num_moves), self.max_width * 4, replace=False, p=move_probs)
        self.unvisited_moves = self.get_valid_move_list(ranked_moves, self.max_width, encoder)

        # If we need more child (at least one)
        if len(self.unvisited_moves) < self.min_width:
            ranked_moves = np.random.choice(np.arange(num_moves), num_moves, replace=False, p=move_probs)
            more_moves = self.get_valid_move_list(ranked_moves, self.min_width - len(self.unvisited_moves), encoder)
            self.unvisited_moves += more_moves

    def get_unvisited_children(self, count):
        ret = []
        for _ in range(count):
            if len(self.unvisited_moves) == 0:
                break
            new_move = self.unvisited_moves.pop(0)
            new_game_state = self.game_state.apply_move(new_move)
            new_node = MCTSACNode(new_game_state, new_move, self)
            self.children.append(new_node)
            ret.append(new_node)
        return ret

    def is_terminal(self):
        return self.game_state.is_over()

    def to_json(self):
        d = dict()
        if self.next_player == Player.black:
            d['next_player'] = 'Black'
            d['type'] = 'type1'
        else:
            d['next_player'] = 'White'
            d['type'] = 'type2'
        d['black'] = f'{self.score[Player.black]:.10f}'
        d['white'] = f'{self.score[Player.white]:.10f}'
        d['rollout'] = str(self.num_rollouts)
        d['explorable_moves'] = str(self.explorable_moves)
        d['critic_score'] = f'{self.critic_score:.10f}'
        children = []
        for child in self.children:
            children.append(child.to_json())
        d['children'] = children
        d['link'] = {"direction": "ASYN"}
        return d


def _uct_score(score, temp, log_rollout, node_rollout):
    win_frac = MCTSACBot.score_to_winrate(score)
    return win_frac + temp * math.sqrt(log_rollout / node_rollout)


class MCTSACBot(Agent):
    def __init__(self, encoder, actor, critic, name=None, width=5, num_rounds=5120, temperature=0.01, batch_size=128,
                 exponent=3):
        super().__init__(name)
        MCTSACNode.max_width = width
        MCTSACNode.min_width = width
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.batch_size = batch_size
        self.encoder = encoder
        self.actor = actor
        self.critic = critic
        self.exponent = exponent
        self.root_cache = None

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

    def select_move(self, game_state, **kwargs):
        # Check whether we can reuse old tree
        root = None
        if self.root_cache is not None:
            if self.root_cache.next_player == game_state.next_player:
                if self.root_cache.game_state.is_same(game_state):
                    root = self.root_cache
            else:
                for child in self.root_cache.children:
                    if child.game_state.is_same(game_state):
                        root = child
                        break
        if root is None:
            # If we cannot reuse old tree
            root = MCTSACNode(game_state, None)
            encoded_board = self.encoder.encode_board(root.game_state.board, root.next_player)
            root.encoded_board = encoded_board
            root.init_score(0)

        while root.num_rollouts < self.num_rounds:
            # Get moves which we evaluate at this iteration
            actor_list, critic_list = self._get_moves_to_eval(root, self.batch_size)

            # Run actor and get moves for un-actored nodes
            if len(actor_list) > 0:
                actor_input = np.array([node.encoded_board for node in actor_list])
                actor_output = self.actor.predict_on_batch(actor_input)
                for node, moves in zip(actor_list, actor_output):
                    node.update_unvisited_moves(self.encoder, moves)

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
            for node, value in zip(critic_nodes, critic_output):
                node.init_score(value[0])

            # Update parents
            for parent, _ in critic_list:
                parent.update()

            # tree = root.to_json()
            # print(json.dumps({'tree': tree}))

        # Select move
        probs = []
        for child in root.children:
            probs.append(self.score_to_winrate(child.score[game_state.next_player]))
        # print(probs)

        ###################### If you want randomly move, use me
        # probs = np.array(probs) ** self.exponent
        # probs = np.clip(probs, eps, 1 - eps)
        # probs = probs / np.sum(probs)
        # chosen_child = np.random.choice(root.children, 1, p=probs)[0]
        max_index = -1
        max_prob = -1
        for index, prob in enumerate(probs):
            if prob > max_prob:
                max_index = index
                max_prob = prob
        chosen_child = root.children[max_index]

        self.root_cache = chosen_child
        return chosen_child.move

    @staticmethod
    def score_to_winrate(score):
        return (score + 1) / 2
