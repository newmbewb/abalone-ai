from __future__ import absolute_import
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.models import Sequential
from keras.layers import Dropout

from dlabalone.networks.base import Network


class EncoderNetworkSimple1(Network):
    def __init__(self, data_format="channels_last"):
        self.data_format = data_format
        super().__init__()

    def model(self, input_shape, output_shape, optimizer='sgd', dropout_rate=0.1):
        model = Sequential()
        if self.data_format == 'channels_last':
            output_channel = output_shape[-1]
        else:
            output_channel = output_shape[0]
        for layer in self._layers(input_shape, output_channel, dropout_rate):
            model.add(layer)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
        return model

    def _layers(self, input_shape, output_channel, dropout_rate):
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
            Conv2D(output_channel, (5, 5), data_format=self.data_format),
            Activation('sigmoid'),
        ]
