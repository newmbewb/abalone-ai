class Network:
    def __init__(self, name=None):
        if name is None:
            self._name = type(self).__name__
        else:
            self._name = name

    def name(self):
        return self._name

    def model(self, input_shape, num_classes):
        raise NotImplementedError()
