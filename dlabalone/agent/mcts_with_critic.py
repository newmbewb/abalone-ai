import json
import math
import random
import sys

import numpy as np
import tensorflow as tf

from dlabalone.abltypes import Player
from dlabalone.agent.base import Agent
from dlabalone.networks.base import prepare_tf_custom_objects


__all__ = [
    'MCTSCriticBot',
]


eps = 1e-6


class MCTSCriticNode(object):
    max_width = 50
    min_width = 50

    dynamic_width = False
    dynamic_width_factor = 2

    def __init__(self, game_state, move, parent=None):
        self.game_state = game_state
        self.next_player = game_state.next_player
        self.move = move
        self.parent = parent
        self.critic_score = None
        self.score = {}
        self.explorable_moves = self.max_width
        self.num_rollouts = 1
        self.base_num_rollouts = 1
        self.children = []
        self.unvisited_moves = None
        self.not_updated_children = set()

    @classmethod
    def set_max_width(cls, max_width):
        cls.max_width = max_width

    def update(self):
        assert self.critic_score is not None
        # Update children first
        for idx, child in enumerate(self.children):
            if idx not in self.not_updated_children:
                child.update()
        self.not_updated_children = set()
        # Initialize
        if self.unvisited_moves is None:
            self.explorable_moves = self.max_width
        else:
            self.explorable_moves = len(self.unvisited_moves)
        next_player = self.next_player
        score_accum = self.critic_score
        self.num_rollouts = self.base_num_rollouts
        for child in self.children:
            self.num_rollouts += child.num_rollouts
            self.explorable_moves += child.explorable_moves
            score_accum += child.score[next_player] * child.num_rollouts
        if self.is_terminal():
            score = self.critic_score
        else:
            score = score_accum / self.num_rollouts

        self.score[next_player] = score
        self.score[next_player.other] = -score

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

    def update_unvisited_moves(self):
        finish_move = self.game_state.finish_move()
        if finish_move is not None:
            self.unvisited_moves = [finish_move]
        else:
            self.unvisited_moves = self.game_state.legal_moves()

    def get_unvisited_children(self, count):
        ret = []
        for _ in range(count):
            if len(self.unvisited_moves) == 0:
                break
            new_move = self.unvisited_moves.pop(0)
            new_game_state = self.game_state.apply_move(new_move)
            new_node = MCTSCriticNode(new_game_state, new_move, self)
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
    win_frac = MCTSCriticBot.score_to_winrate(score)
    return win_frac + temp * math.sqrt(log_rollout / node_rollout)


class MCTSCriticBot(Agent):
    max_depth = 1000

    def __init__(self, encoder, critic, name=None, num_rounds=5120, temperature=0.01, batch_size=128, **kwargs):
        super().__init__(name)
        prepare_tf_custom_objects()
        sys.setrecursionlimit(100000)
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.batch_size = batch_size
        self.encoder = encoder
        self.critic = critic
        self.root_cache = None
        self.reach_max_depth = False
        self.train = True
        self.dynamic_width_randomness = 0.05

    def _get_moves_to_eval(self, node, budget, depth):
        '''
        :param node:
        :param budget:
        :return: actor_list, critic_list
          actor_list: The list of nodes which need to run the actor model
          critic_list(parent, count): The list of move to evaluate with the critic model
        '''
        assert len(node.not_updated_children) == 0, 'not_updated_children is not empty!'
        if depth >= self.max_depth:
            self.reach_max_depth = True
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
            if child.is_terminal():
                child.base_num_rollouts += budget_for_children[idx]
                child.num_rollouts += budget_for_children[idx]
                continue
            if budget_for_children[idx] == 0:
                node.not_updated_children.add(idx)
                pass
            al, ml = self._get_moves_to_eval(child, budget_for_children[idx], depth + 1)
            actor_list += al
            critic_list += ml

        return actor_list, critic_list

    def select_move(self, game_state, **kwargs):
        # Check whether we can reuse old tree
        self.reach_max_depth = False
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
            root = MCTSCriticNode(game_state, None)
            root.init_score(0)

        loop_count = 0
        while root.num_rollouts < self.num_rounds and not self.reach_max_depth:
            loop_count += 1
            if loop_count == 100000:
                print(f'Too many loop_count!!!!! {loop_count}')
                break
            # Get moves which we evaluate at this iteration
            actor_list, critic_list = self._get_moves_to_eval(root, self.batch_size, 0)

            # Run actor and get moves for un-actored nodes
            for node in actor_list:
                node.update_unvisited_moves()

            # Calculate value for new nodes
            critic_nodes = []
            # Get new children's state
            for parent, child_count in critic_list:
                critic_nodes += parent.get_unvisited_children(child_count)

            if len(critic_nodes) > 0:
                # Encode them
                critic_input = []
                for node in critic_nodes:
                    encoded_board = self.encoder.encode_board(node.game_state.board, node.next_player)
                    critic_input.append(encoded_board)
                # Evaluate the value
                critic_input = np.array(critic_input)
                critic_output = self.critic.predict_on_batch(critic_input)
                for node, value in zip(critic_nodes, critic_output):
                    node.init_score(value[0])

            # update
            root.update()

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
        if not self.train:
            max_index = -1
            max_prob = -1
            for index, prob in enumerate(probs):
                if prob > max_prob:
                    max_index = index
                    max_prob = prob
            chosen_child = root.children[max_index]
        else:
            max_prob = max(probs)
            candidate_idx = []
            candidate_prob_accum = []
            prob_accum = 0
            # Gather moves whose winrate is larger than max winrate - dynamic_width_randomness
            for idx, prob in enumerate(probs):
                if prob > max_prob - self.dynamic_width_randomness:
                    candidate_idx.append(idx)
                    prob_accum += prob
                    candidate_prob_accum.append(prob_accum)
            # Select one
            p_random = random.random() * prob_accum
            selected_index = None
            for idx, p_accum in zip(candidate_idx, candidate_prob_accum):
                if p_random < p_accum:
                    selected_index = idx
                    break
            # if selected_index is None:
            #     selected_index = len(candidate_prob_accum) - 1
            chosen_child = root.children[selected_index]

        chosen_child.parent = None
        self.root_cache = chosen_child
        if 'board_move_pair' in kwargs:
            kwargs['board_move_pair'].append((chosen_child.game_state.board, chosen_child.move))
        return chosen_child.move

    @staticmethod
    def score_to_winrate(score):
        return (score + 1) / 2
