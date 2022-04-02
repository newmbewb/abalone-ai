import copy
import random
from dlabalone.ablboard import GameState
from dlabalone.abltypes import Player
from dlabalone.agent.base import Agent
from dlabalone.utils import print_board


__all__ = ['AlphaBetaBot']
MAX_SCORE = 100000000
MIN_SCORE = -1 * MAX_SCORE


def score_game_state(game_state):
    if game_state.next_player == Player.black:
        return game_state.board.dead_stones_white - game_state.board.dead_stones_black
    else:
        return game_state.board.dead_stones_black - game_state.board.dead_stones_white


class AlphaBetaBot(Agent):
    def __init__(self, name=None, depth=2, width=None):
        super().__init__(name)
        self.max_depth = depth
        self.width = width

    def select_move(self, game_state):
        assert not game_state.is_over()
        best_score = MIN_SCORE
        best_moves = []
        legal_moves = game_state.legal_moves()
        for move in legal_moves:
            undo_move = game_state.apply_move_lite(move)
            score = self.alpha_beta_result(game_state, self.max_depth - 1, MIN_SCORE,
                                           MIN_SCORE, score_game_state) * -1
            game_state.undo(undo_move)
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
            else:
                pass
        return random.choice(best_moves)

    def alpha_beta_result(self, game_state, max_depth, best_black, best_white, eval_fn):
        if game_state.is_over():
            if game_state.winner() == game_state.next_player:
                return MAX_SCORE
            else:
                return MIN_SCORE

        if max_depth <= 0:
            return eval_fn(game_state)

        moves_kill, moves_other = game_state.legal_moves(separate_kill=True)
        if max_depth == 1:
            if len(moves_kill) > 0:
                return eval_fn(game_state) + 1
            else:
                return eval_fn(game_state)

        if self.width is None:
            legal_moves = moves_kill + moves_other
        else:
            random.shuffle(moves_kill)
            len_moves_kill = len(moves_kill)
            if len_moves_kill >= self.width:
                legal_moves = moves_kill[:self.width]
            else:
                random.shuffle(moves_other)
                legal_moves = moves_kill + moves_other[:self.width - len_moves_kill]

        best_so_far = MIN_SCORE
        for candidate_move in legal_moves:
            undo_move = game_state.apply_move_lite(candidate_move)
            opponent_best_result = self.alpha_beta_result(game_state, max_depth - 1, best_black, best_white, eval_fn)
            game_state.undo(undo_move)
            our_result = -1 * opponent_best_result

            if our_result > best_so_far:
                best_so_far = our_result
            if game_state.next_player == Player.white:
                if best_so_far > best_white:
                    best_white = best_so_far
                outcome_for_black = -1 * best_so_far
                if outcome_for_black < best_black:
                    return best_so_far
            elif game_state.next_player == Player.black:
                if best_so_far > best_black:
                    best_black = best_so_far
                outcome_for_white = -1 * best_so_far
                if outcome_for_white < best_white:
                    return best_so_far

        return best_so_far