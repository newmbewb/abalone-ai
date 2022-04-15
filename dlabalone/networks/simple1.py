from __future__ import absolute_import
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.models import Sequential

from dlabalone.networks.base import Network


class Simple1(Network):
    def __init__(self):
        super().__init__()

    def model(self, input_shape, num_classes):
        model = Sequential()
        for layer in self.layers(input_shape):
            model.add(layer)
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def layers(self, input_shape):
        return [
            ZeroPadding2D((2, 2), input_shape=input_shape, data_format='channels_first'),
            Conv2D(64, (5, 5), padding='valid', data_format='channels_first'),
            Activation('relu'),

            ZeroPadding2D((2, 2), data_format='channels_first'),
            Conv2D(64, (5, 5), data_format='channels_first'),
            Activation('relu'),

            ZeroPadding2D((2, 2), data_format='channels_first'),
            Conv2D(64, (5, 5), data_format='channels_first'),
            Activation('relu'),

            ZeroPadding2D((2, 2), data_format='channels_first'),
            Conv2D(48, (5, 5), data_format='channels_first'),
            Activation('relu'),

            ZeroPadding2D((2, 2), data_format='channels_first'),
            Conv2D(48, (5, 5), data_format='channels_first'),
            Activation('relu'),

            ZeroPadding2D((2, 2), data_format='channels_first'),
            Conv2D(32, (5, 5), data_format='channels_first'),
            Activation('relu'),

            ZeroPadding2D((2, 2), data_format='channels_first'),
            Conv2D(32, (5, 5), data_format='channels_first'),
            Activation('relu'),

            Flatten(),
        ]
