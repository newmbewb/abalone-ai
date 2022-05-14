class MoveSelector:
    def __init__(self, name=None, temperature=0.5):
        if name is None:
            self._name = type(self).__name__
        else:
            self._name = name
        self.temperature = temperature
        self.temp_max = 1
        self.temp_min = 0.02

    def name(self):
        return self._name

    def __str__(self):
        return f'{self.name()}/Temp:{self.temperature:.5f}'

    def __call__(self, encoder, move_probs, legal_moves):
        raise NotImplementedError()

    def temperature_up(self):
        self.temperature += 0.01
        self.temperature = min(self.temperature, self.temp_max)

    def temperature_down(self):
        self.temperature -= 0.01
        self.temperature = max(self.temperature, self.temp_min)

    def is_over(self):
        if self.temperature <= 0:
            return True
        else:
            return False

    # def adjust_temperature(self, win_rate):
    #     if win_rate > self.win_rate_max:
    #         self.temperature_up()
    #     elif win_rate < self.win_rate_min:
    #         self.temperature_down()
