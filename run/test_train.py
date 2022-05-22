import multiprocessing

from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD

from dlabalone.data.data_processor import DataGenerator, DataGeneratorMock
from dlabalone.encoders.twoplane import TwoPlaneEncoder
from dlabalone.encoders.threeplane import ThreePlaneEncoder
from dlabalone.encoders.base import get_encoder_by_name
from dlabalone.networks import simple1, alpha_abalone, ac_simple1, ac_light1
from keras.models import Sequential, load_model
from keras.layers.core import Dense
import os
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    multiprocessing.freeze_support()
    mode = 'policy'
    optimizer = SGD(learning_rate=0.1)
    # optimizer = 'adam'
    if mode == 'policy':
        dropout_rate = 0.10
    else:
        dropout_rate = 0.15
    # encoder = get_encoder_by_name('alpha_abalone', 5, mode)
    encoder = get_encoder_by_name('fourplane', 5, mode, data_format="channels_last")
    generator = DataGenerator(encoder, 1024, '../data/data_with_value/dataset', '../data/encoded_data', 0.2)

    model_generator = ac_simple1.ACSimple1(mode, dropout_rate=dropout_rate, data_format="channels_last")
    # model_generator = ac_light1.ACLight1(mode, dropout_rate=dropout_rate, data_format="channels_last")
    model = model_generator.model(encoder.shape(), encoder.num_moves(), optimizer=optimizer)
    # model = load_model('../data/checkpoints/backup/ACSimple1Policy_dropout0.1_AlphaAbaloneEncoder_epoch_14.h5')

    # Make network name
    model.summary()
    network_name = f'{model_generator.name()}_{encoder.name()}'
    print(network_name)
    print(f'optimizer: {optimizer}')
    epochs = 10
    model.fit_generator(
        generator=generator.generate('train'),
        epochs=epochs,
        steps_per_epoch=generator.get_num_steps('train'),
        validation_data=generator.generate('test'),
        validation_steps=generator.get_num_steps('test'),
        callbacks=[
            ModelCheckpoint(f'../data/checkpoints/{network_name}_epoch_{{epoch}}.h5')
        ]
    )
