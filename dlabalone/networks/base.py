import functools
import keras


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


def compile_model(model, mode, optimizer):
    if mode == 'policy':
        top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
        top3_acc.__name__ = 'top3_acc'
        top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
        top5_acc.__name__ = 'top5_acc'
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', top3_acc, top5_acc])
    elif mode == 'value':
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
    else:
        pass
