from __future__ import absolute_import
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.models import Sequential
from keras.layers import Dropout
from keras.metrics import RootMeanSquaredError
from keras.losses import Huber

from dlabalone.networks.base import Network


class AlphaAbalone(Network):
    def __init__(self, mode, data_format="channels_first"):
        self.mode = mode
        self.data_format = data_format
        if mode == 'policy':
            name = "AlphaAbalonePolicy"
        elif mode == 'value':
            name = "AlphaAbaloneValue"
        else:
            assert False, "Wrong mode"
        super().__init__(name)

    def model(self, input_shape, num_classes, optimizer='adam', dropout_rate=0.1):
        model = self.alphago_model(input_shape, num_classes, dropout_rate)
        if self.mode == 'policy':
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        elif self.mode == 'value':
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        else:
            return None
        return model

    def alphago_model(self, input_shape, num_classes, dropout_rate,
                      num_filters=192,
                      first_kernel_size=5,
                      other_kernel_size=3):

        model = Sequential()
        model.add(
            Conv2D(num_filters, first_kernel_size, input_shape=input_shape, padding='same', activation='relu',
                   data_format=self.data_format))
        model.add(Dropout(rate=dropout_rate))

        for i in range(2, 12):
            model.add(
                Conv2D(num_filters, other_kernel_size, padding='same', activation='relu', data_format=self.data_format))
            model.add(Dropout(rate=dropout_rate))

        if self.mode == 'policy':
            model.add(Flatten())
            model.add(Dense(num_classes, activation='softmax'))
            return model

        elif self.mode == 'value':
            model.add(
                Conv2D(num_filters, other_kernel_size, padding='same', activation='relu', data_format=self.data_format))
            model.add(Dropout(rate=dropout_rate))
            model.add(
                Conv2D(filters=1, kernel_size=1, padding='same', activation='relu', data_format=self.data_format))
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dense(1, activation='tanh'))
            return model
