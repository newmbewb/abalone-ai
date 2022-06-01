import functools
import keras
import tensorflow.keras.backend as K
from keras.layers.core import Dense, Activation, Flatten
from tensorflow.keras.utils import get_custom_objects

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


def prepare_tf_custom_objects():
    # Define mish activation
    class Mish(Activation):
        def __init__(self, activation, **kwargs):
            super(Mish, self).__init__(activation, **kwargs)
            self.__name__ = 'Mish'

    def mish(x):
        return x * K.tanh(K.softplus(x))

    get_custom_objects().update({'Mish': Mish})
    get_custom_objects().update({'mish': Mish(mish)})

    top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = 'top3_acc'
    top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
    top5_acc.__name__ = 'top5_acc'
    get_custom_objects().update({'top3_acc': top3_acc})
    get_custom_objects().update({'top5_acc': top5_acc})


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
