import random
from dlabalone.agent.base import Agent


__all__ = ['RandomKillBot']


class RandomKillBot(Agent):
    def select_move(self, game_state):
        moves_kill, moves_other = game_state.legal_moves(separate_kill=True)
        if len(moves_kill) > 0:
            return random.choice(moves_kill)
        else:
            return random.choice(moves_other)

    # def select_move(self, game_state):
    #     score = game_state.board.dead_stones[game_state.next_player.other]
    #     moves_kill = []
    #     moves_other = []
    #     for move in game_state.legal_moves():
    #         next_state = game_state.apply_move(move)
    #         new_score = next_state.board.dead_stones[game_state.next_player.other]
    #         if new_score > score:
    #             moves_kill.append(move)
    #         else:
    #             moves_other.append(move)
    #     if len(moves_kill) > 0:
    #         return random.choice(moves_kill)
    #     else:
    #         return random.choice(moves_other)
