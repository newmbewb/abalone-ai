import os
import time

from dlabalone.data.encoder_trainset_generator import get_encoder_trainset_filelist, encoder_trainset_generator, \
    generate_encoder_trainset
from dlabalone.encoders.alpha_abalone import AlphaAbaloneEncoder
from dlabalone.encoders.base import get_encoder_by_name
from dlabalone.encoders.fourplane import FourPlaneEncoder
from dlabalone.encoding_networks.simple1 import EncoderNetworkSimple1
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K

from dlabalone.encoding_networks.simple1_concat import EncoderNetworkConcatSimple1
from dlabalone.encoding_networks.simple2 import EncoderNetworkSimple2
from dlabalone.encoding_networks.simple2_concat import EncoderNetworkConcatSimple2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def zero_or_one(x):
    return K.tanh((x - 0.5) * 100000) * 0.5 + 0.5
    # return K.switch(x >= 0.5, x, x)


if __name__ == '__main__':
    batch = 4096
    steps = 10000
    encoder_trainset_path = '../data/encoder_trainset/game_generated'
    # start_time = time.time()
    # generate_encoder_trainset(FourPlaneEncoder(5, None), AlphaAbaloneEncoder(5, None), batch, steps,
    #                           '../data/data_with_value/dataset/', encoder_trainset_path)
    # print(time.time() - start_time)
    train_set, valid_set = get_encoder_trainset_filelist(encoder_trainset_path)
    # encoder_network = EncoderNetworkSimple1()
    encoder_network = EncoderNetworkConcatSimple2()
    input_encoder = get_encoder_by_name('fourplane', 5)
    output_encoder = get_encoder_by_name('alpha_abalone', 5, with_basic_plains=False)
    network_name = f'{encoder_network.name()}_{input_encoder.name()}_{output_encoder.name()}'
    encoder_model = encoder_network.model(input_encoder.shape(), output_encoder.shape(), optimizer='adam')
    ############################
    # encoder_model = load_model(
    #     '../data/checkpoints/EncoderNetworkSimple1_FourPlaneEncoder_channels_last_AlphaAbaloneEncoder_channels_last_epoch_6.h5')
    # encoder_model = load_model(
    #     '../data/checkpoints/EncoderNetworkSimple1_FourPlaneEncoder_channels_last_AlphaAbaloneEncoder_channels_last_epoch_2.h5')
    # encoder_model.summary()
    model = encoder_model


    # ####################################
    # # encoder_model = load_model('../data/checkpoints/EncoderNetworkSimple1_FourPlaneEncoder_channels_last_AlphaAbaloneEncoder_channels_last_epoch_6.h5')
    # config = encoder_model.get_config()  # Returns pretty much every information about your model
    # input_shape = config["layers"][0]["config"]["batch_input_shape"][1:]
    # inputs = tf.keras.Input(shape=input_shape)
    # x = encoder_model(inputs, training=True)
    # outputs = Activation(zero_or_one)(x)
    # model = tf.keras.Model(inputs, outputs)
    # model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # #####################################
    model.summary()
    epochs = 100
    # for x, y in encoder_trainset_generator(train_set):
    #     print(x.shape)
    #     print(y.shape)
    #     break
    print(network_name)
    model.fit_generator(
        generator=encoder_trainset_generator(train_set),
        epochs=epochs,
        steps_per_epoch=len(train_set),
        validation_data=encoder_trainset_generator(valid_set),
        validation_steps=len(valid_set),
        callbacks=[
            ModelCheckpoint(f'../data/checkpoints/{network_name}_epoch_{{epoch}}.h5')
        ]
    )
