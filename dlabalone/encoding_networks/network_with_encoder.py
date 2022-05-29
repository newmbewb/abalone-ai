from __future__ import absolute_import

import functools

import keras
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.models import Sequential
from keras.layers import Dropout
from keras.metrics import RootMeanSquaredError
import tensorflow as tf


class NetworkWithEncoder:
    def __init__(self, encoder_model, base_model):
        self.encoder_model = encoder_model
        self.encoder_model.trainable = False
        self.base_model = base_model

    def model(self):
        config = self.encoder_model.get_config()  # Returns pretty much every information about your model
        input_shape = config["layers"][0]["config"]["batch_input_shape"][1:]
        inputs = tf.keras.Input(shape=input_shape)
        x = self.encoder_model(inputs)
        outputs = self.base_model(x)
        model = tf.keras.Model(inputs, outputs)
        return model


def get_output_shape(model):
    return model.layers[-1].output_shape[1:]
