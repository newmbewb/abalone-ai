from __future__ import absolute_import
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.models import Sequential
from keras.layers import Dropout
from keras.metrics import RootMeanSquaredError
from keras.losses import Huber

from dlabalone.networks.base import Network


class ACSimple(Network):
    def __init__(self, mode, dropout_rate=0.1):
        self.mode = mode
        self.dropout_rate = dropout_rate
        self.data_format = "channels_first"
        if mode == 'policy':
            name = f"ACSimple1Policy_dropout{dropout_rate}"
        elif mode == 'value':
            name = f"ACSimple1Value_dropout{dropout_rate}"
        else:
            assert False, "Wrong mode"
        super().__init__(name)

    def model(self, input_shape, num_classes, optimizer='adam'):
        model = self._model(input_shape, num_classes, dropout_rate=self.dropout_rate)
        if self.mode == 'policy':
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        elif self.mode == 'value':
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        else:
            return None
        return model

    def _model(self, input_shape, num_classes,
               num_filters=192,
               first_kernel_size=5,
               other_kernel_size=3,
               dropout_rate=0.1):

        model = Sequential()
        for layer in self._layers(input_shape, dropout_rate):
            model.add(layer)

        if self.mode == 'policy':
            model.add(Flatten())
            model.add(Dense(num_classes, activation='softmax'))
            return model

        elif self.mode == 'value':
            model.add(ZeroPadding2D((2, 2), data_format=self.data_format))
            model.add(Conv2D(64, (5, 5), data_format=self.data_format))
            model.add(Activation('relu'))
            model.add(Dropout(rate=dropout_rate))

            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dense(1, activation='tanh'))
            return model

    def _layers(self, input_shape, dropout_rate):
        return [
            ZeroPadding2D((2, 2), input_shape=input_shape, data_format=self.data_format),
            Conv2D(64, (5, 5), data_format=self.data_format),
            Activation('relu'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2), data_format=self.data_format),
            Conv2D(64, (5, 5), data_format=self.data_format),
            Activation('relu'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2), data_format=self.data_format),
            Conv2D(64, (5, 5), data_format=self.data_format),
            Activation('relu'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2), data_format=self.data_format),
            Conv2D(48, (5, 5), data_format=self.data_format),
            Activation('relu'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2), data_format=self.data_format),
            Conv2D(48, (5, 5), data_format=self.data_format),
            Activation('relu'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2), data_format=self.data_format),
            Conv2D(32, (5, 5), data_format=self.data_format),
            Activation('relu'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2), data_format=self.data_format),
            Conv2D(32, (5, 5), data_format=self.data_format),
            Activation('relu'),
            Dropout(rate=dropout_rate),
        ]
