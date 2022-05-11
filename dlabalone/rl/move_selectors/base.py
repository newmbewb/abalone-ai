class MoveSelector:
    def __init__(self, name=None, win_rate_max=0.45, win_rate_min=0.20):
        if name is None:
            self._name = type(self).__name__
        else:
            self._name = name
        self.win_rate_max = win_rate_max
        self.win_rate_min = win_rate_min

    def name(self):
        return self._name

    def __call__(self, encoder, move_probs, legal_moves):
        raise NotImplementedError()

    def temperature_up(self):
        raise NotImplementedError()

    def temperature_down(self):
        raise NotImplementedError()

    def is_over(self):
        raise NotImplementedError()

    def adjust_temperature(self, win_rate):
        if win_rate > self.win_rate_max:
            self.temperature_up()
        elif win_rate < self.win_rate_min:
            self.temperature_down()
