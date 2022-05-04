from __future__ import absolute_import
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.models import Sequential
from keras.layers import Dropout

from dlabalone.networks.base import Network


class Simple1(Network):
    def __init__(self):
        super().__init__()

    def model(self, input_shape, num_classes, optimizer='adam', dropout_rate=0.1):
        model = Sequential()
        for layer in self._layers(input_shape, dropout_rate):
            model.add(layer)
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def _layers(self, input_shape, dropout_rate):
        return [
            ZeroPadding2D((2, 2), input_shape=input_shape),
            Conv2D(64, (5, 5), padding='valid'),
            Activation('relu'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2)),
            Conv2D(64, (5, 5)),
            Activation('relu'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2)),
            Conv2D(64, (5, 5)),
            Activation('relu'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2)),
            Conv2D(48, (5, 5)),
            Activation('relu'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2)),
            Conv2D(48, (5, 5)),
            Activation('relu'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2)),
            Conv2D(32, (5, 5)),
            Activation('relu'),
            Dropout(rate=dropout_rate),

            ZeroPadding2D((2, 2)),
            Conv2D(32, (5, 5)),
            Activation('relu'),
            Dropout(rate=dropout_rate),

            Flatten(),
        ]
