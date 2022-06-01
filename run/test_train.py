import multiprocessing

from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD

from dlabalone.data.data_processor import DataGenerator, DataGeneratorMock
from dlabalone.encoders.twoplane import TwoPlaneEncoder
from dlabalone.encoders.threeplane import ThreePlaneEncoder
from dlabalone.encoders.base import get_encoder_by_name
from dlabalone.encoding_networks.network_with_encoder import NetworkWithEncoder, get_output_shape
from dlabalone.networks import simple1, alpha_abalone, ac_simple1, ac_light1
from keras.models import Sequential, load_model
from keras.layers.core import Dense
import os
import tensorflow as tf

from dlabalone.networks.base import compile_model, prepare_tf_custom_objects

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    # tf.compat.v1.disable_eager_execution()
    multiprocessing.freeze_support()
    mode = 'value'
    # optimizer = SGD(learning_rate=0.1)

    if mode == 'policy':
        # optimizer = 'sgd'
        optimizer = SGD(learning_rate=0.1)
        dropout_rate = 0.3
    elif mode == 'value':
        optimizer = 'adam'
        # optimizer = SGD(learning_rate=0.1)
        dropout_rate = 0.5
    else:
        assert False, "mode should be 'policy' or 'value'"
    # encoder = get_encoder_by_name('alpha_abalone', 5, mode, data_format="channels_last")
    encoder = get_encoder_by_name('fourplane', 5, mode, data_format="channels_last")
    # generator = DataGenerator(encoder, 1024, '../data/data_with_value/dataset', '../data/encoded_data', mode, 0.2,
    #                           max_train_npz=200)
    generator = DataGenerator(
        encoder, 1024, '../data/rl_mcts/dataset', '../data/rl_mcts/encoded_data', mode, 0.2)

    # Simple network
    prepare_tf_custom_objects()
    model_generator = ac_simple1.ACSimple1(mode, dropout_rate=dropout_rate, data_format="channels_last")
    # model_generator = ac_light1.ACLight1(mode, dropout_rate=dropout_rate, data_format="channels_last")
    model = model_generator.model(encoder.shape(), encoder.num_moves(), optimizer=optimizer)
    network_name = f'{model_generator.name()}_{encoder.name()}'


    # # Stack encoder model
    # encoder_model = load_model('../data/checkpoints/models/EncoderNetworkConcatSimple2_FourPlaneEncoder_channels_last_AlphaAbaloneEncoder_channels_last_epoch_1.h5')
    # base_model_generator = ac_simple1.ACSimple1(mode, dropout_rate=dropout_rate, data_format="channels_last")
    # base_model = base_model_generator.model(get_output_shape(encoder_model), encoder.num_moves(), optimizer=optimizer)
    # model_generator = NetworkWithEncoder(encoder_model, base_model)
    # model = model_generator.model()
    # compile_model(model, mode, optimizer)
    # network_name = f'{base_model_generator.name()}_with_encoder_{encoder.name()}'

    # Make network name
    model.summary()
    # network_name = f'{model_generator.name()}_{encoder.name()}'
    print(network_name)
    print(f'optimizer: {optimizer}')
    epochs = 1000
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
