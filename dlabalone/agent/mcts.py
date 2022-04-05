import math
import random
from dlabalone.abltypes import Player
from dlabalone.agent.base import Agent
from dlabalone.agent.random_kill_first import RandomKillBot


__all__ = [
    'MCTSBot',
]


class MCTSNode(object):
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {
            Player.black: 0.,
            Player.white: 0.,
        }
        self.num_rollouts = 0
        self.children = []
        if game_state.is_over():
            self.unvisited_moves = []
        else:
            moves_kill, moves_other = game_state.legal_moves(separate_kill=True)
            random.shuffle(moves_kill)
            random.shuffle(moves_other)
            self.unvisited_moves = moves_kill + moves_other

    def add_random_child(self):
        # Select better move first
        new_move = self.unvisited_moves.pop(0)
        new_game_state = self.game_state.apply_move(new_move)
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node

    def record_black_win_probability(self, probability):
        self.win_counts[Player.black] += probability
        self.win_counts[Player.white] += 1 - probability
        self.num_rollouts += 1

    def can_add_child(self):
        return len(self.unvisited_moves) > 0

    def is_terminal(self):
        return self.game_state.is_over()

    def winning_frac(self, player):
        return float(self.win_counts[player]) / float(self.num_rollouts)


class MCTSBot(Agent):
    def __init__(self, name=None, num_rounds=10, temperature=1.5):
        super().__init__(name)
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game_state):
        root = MCTSNode(game_state)

        for i in range(self.num_rounds):
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            # Add a new child node into the tree.
            if node.can_add_child():
                node = node.add_random_child()

            # Simulate a random game from this node.
            black_win_probability = self.simulate_random_game(node.game_state)

            # Propagate scores back up the tree.
            while node is not None:
                node.record_black_win_probability(black_win_probability)
                node = node.parent

        scored_moves = [
            (child.winning_frac(game_state.next_player), child.move, child.num_rollouts)
            for child in root.children
        ]
        scored_moves.sort(key=lambda x: x[0], reverse=True)

        # Having performed as many MCTS rounds as we have time for, we now pick a move.
        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_frac(game_state.next_player)
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        return best_move

    def select_child(self, node):
        """Select a child according to the upper confidence bound for
        trees (UCT) metric.
        """
        total_rollouts = node.num_rollouts
        log_rollouts = math.log(total_rollouts)

        best_score = -1
        best_child = None
        # Loop over each child.
        for child in node.children:
            # Calculate the UCT score.
            win_percentage = child.winning_frac(node.game_state.next_player)
            exploration_factor = math.sqrt(log_rollouts / child.num_rollouts)
            uct_score = win_percentage + self.temperature * exploration_factor
            # Check if this is the largest we've seen so far.
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child

    @staticmethod
    def simulate_random_game(game):
        return game.win_probability(Player.black)
        # bot = RandomKillBot()
        # for _ in range(50):
        #     if game.is_over():
        #         break
        #     bot_move = bot.select_move(game)
        #     game.apply_move_lite(bot_move)
        # return game.win_probability(Player.black)
