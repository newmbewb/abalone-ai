import multiprocessing

from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD

from dlabalone.data.data_processor import DataGenerator, DataGeneratorMock
from dlabalone.encoders.twoplane import TwoPlaneEncoder
from dlabalone.encoders.threeplane import ThreePlaneEncoder
from dlabalone.encoders.base import get_encoder_by_name
from dlabalone.networks import simple1, alpha_abalone
from keras.models import Sequential, load_model
from keras.layers.core import Dense
import os
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    multiprocessing.freeze_support()
    mode = 'value'
    encoder = get_encoder_by_name('alpha_abalone', 5, mode)
    generator = DataGenerator(encoder, 1024, '../data/data_with_value/dataset', '../data/encoded_data', 0.2)
    # generator = DataGeneratorMock(encoder, 1024, 100, 25)
    # encoder = get_encoder_by_name('fourplane', 5)
    # generator = DataGenerator(encoder, 4096, '../data/dataset', '../data/encoded_data', 0.2)


    # generator = DataGeneratorMock(encoder, 8192, 100, 25)
    # dataset = tf.data.Dataset.from_generator(lambda: generator.generate('train'), (tf.int8, tf.int8),
    #                                          (encoder.shape(), (encoder.num_moves(), )))
    # dataset = dataset.batch(256)

    model_generator = alpha_abalone.AlphaAbalone(mode)
    # model = model_generator.model(encoder.shape(), encoder.num_moves(), optimizer=SGD(learning_rate=1))
    model = model_generator.model(encoder.shape(), encoder.num_moves(), optimizer='adam')
    # model = load_model('../checkpoints/simple1_twoplane_epoch_10.h5')

    # Make network name
    network_name = f'{model_generator.name()}_{encoder.name()}'
    print(network_name)
    epochs = 100
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
