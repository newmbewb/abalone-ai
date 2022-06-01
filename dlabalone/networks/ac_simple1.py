from __future__ import absolute_import

import functools

import keras
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.models import Sequential
from keras.layers import Dropout
from keras.metrics import RootMeanSquaredError
from keras.losses import Huber
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects

from dlabalone.networks.base import Network, prepare_tf_custom_objects


class ACSimple1(Network):
    def __init__(self, mode, dropout_rate=None, data_format="channels_last"):
        self.mode = mode
        self.dropout_rate = dropout_rate
        self.data_format = data_format
        if mode == 'policy':
            if dropout_rate is None:
                self.dropout_rate = 0.3
            name = f"ACSimple1Policy_dropout{dropout_rate}"
        elif mode == 'value':
            if dropout_rate is None:
                self.dropout_rate = 0.5
            name = f"ACSimple1Value_dropout{dropout_rate}"
        else:
            assert False, "Wrong mode"
        super().__init__(name)
        prepare_tf_custom_objects()


    def model(self, input_shape, num_classes, optimizer=None):
        if self.mode == 'policy':
            model = self._policy_model(input_shape, num_classes, dropout_rate=self.dropout_rate)
            top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
            top3_acc.__name__ = 'top3_acc'
            top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
            top5_acc.__name__ = 'top5_acc'
            if optimizer is None:
                optimizer = SGD(learning_rate=0.1)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', top3_acc, top5_acc])
        elif self.mode == 'value':
            model = self._value_model(input_shape, dropout_rate=self.dropout_rate)
            if optimizer is None:
                optimizer = 'adam'
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        else:
            return None
        return model

    def _policy_model(self, input_shape, num_classes, dropout_rate):
        model = Sequential()
        for layer in self._policy_layers(input_shape, dropout_rate):
            model.add(layer)

        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        return model

    def _policy_layers(self, input_shape, dropout_rate):
        return [
            ZeroPadding2D((2, 2), input_shape=input_shape, data_format=self.data_format),
            Conv2D(64, (5, 5), data_format=self.data_format),
            Activation('mish'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2), data_format=self.data_format),
            Conv2D(64, (5, 5), data_format=self.data_format),
            Activation('mish'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2), data_format=self.data_format),
            Conv2D(64, (5, 5), data_format=self.data_format),
            Activation('mish'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2), data_format=self.data_format),
            Conv2D(48, (5, 5), data_format=self.data_format),
            Activation('mish'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2), data_format=self.data_format),
            Conv2D(48, (5, 5), data_format=self.data_format),
            Activation('mish'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2), data_format=self.data_format),
            Conv2D(32, (5, 5), data_format=self.data_format),
            Activation('mish'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2), data_format=self.data_format),
            Conv2D(32, (5, 5), data_format=self.data_format),
            Activation('mish'),
            Dropout(rate=dropout_rate),
        ]

    def _value_model(self, input_shape, dropout_rate):
        model = Sequential()
        for layer in self._value_layers(input_shape, dropout_rate):
            model.add(layer)

        model.add(Flatten())
        model.add(Dense(256, activation='mish'))
        model.add(Dense(1, activation='tanh'))
        return model

    def _value_layers(self, input_shape, dropout_rate):
        return [
            ZeroPadding2D(padding=3, input_shape=input_shape, data_format=self.data_format),
            Conv2D(32, (5, 5), data_format=self.data_format),
            Activation('mish'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D(padding=2, data_format=self.data_format),
            Conv2D(32, (5, 5), data_format=self.data_format),
            Activation('mish'),
            Dropout(rate=dropout_rate),
        ]
