__all__ = [
    'Agent',
]


class Agent:
    def __init__(self, name=None):
        if name is None:
            self.name = type(self).__name__
        else:
            self.name = name

    def select_move(self, game_state):
        raise NotImplementedError()

    def diagnostics(self):
        return {}
