from keras.callbacks import ModelCheckpoint

from dlabalone.data.data_processor import DataGenerator, DataGeneratorMock
from dlabalone.encoders.twoplane import TwoPlaneEncoder
from dlabalone.networks import simple1
from keras.models import Sequential
from keras.layers.core import Dense
import os
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    encoder = TwoPlaneEncoder(5)
    generator = DataGenerator(encoder, 256, '../../dataset', '../../encoded_data', 0.2)
    # generator = DataGeneratorMock(encoder, 8192, 100, 25)
    # dataset = tf.data.Dataset.from_generator(lambda: generator.generate('train'), (tf.int8, tf.int8),
    #                                          (encoder.shape(), (encoder.num_moves(), )))
    # dataset = dataset.batch(256)

    model_generator = simple1.Simple1()
    model = model_generator.model(encoder.shape(), encoder.num_moves())

    # Make network name
    network_name = f'{model_generator.name()}_{encoder.name()}'
    print(network_name)
    epochs = 1
    model.fit_generator(
        generator=generator.generate('train'),
        epochs=epochs,
        steps_per_epoch=generator.get_num_steps('train'),
        validation_data=generator.generate('test'),
        validation_steps=generator.get_num_steps('test'),
        callbacks=[
            ModelCheckpoint('../checkpoints/simple1_twoplane_epoch_{epoch}.h5')
        ]
    )
