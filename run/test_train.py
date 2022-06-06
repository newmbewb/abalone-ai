import multiprocessing

from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD

from dlabalone.data.data_processor import DataGenerator, DataGeneratorMock
from dlabalone.encoders.twoplane import TwoPlaneEncoder
from dlabalone.encoders.threeplane import ThreePlaneEncoder
from dlabalone.encoders.base import get_encoder_by_name
from dlabalone.encoding_networks.network_with_encoder import NetworkWithEncoder, get_output_shape
from dlabalone.networks import simple1, alpha_abalone, ac_simple1, ac_light1, ac_residual1, ac_simple2, ac_simple3, \
    ac_simple4
from keras.models import Sequential, load_model
from keras.layers.core import Dense
import os
import tensorflow as tf

from dlabalone.networks.base import compile_model, prepare_tf_custom_objects

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    # tf.compat.v1.disable_eager_execution()
    multiprocessing.freeze_support()
    mode = 'policy'

    if mode == 'policy':
        optimizer = SGD(learning_rate=0.1)
        dropout_rate = 0.3
        # optimizer = 'adam'
        # dropout_rate = 0.1
    elif mode == 'value':
        optimizer = 'adam'
        dropout_rate = 0.5
    else:
        assert False, "mode should be 'policy' or 'value'"
    encoder = get_encoder_by_name('fourplane', 5, mode, data_format="channels_last")
    # generator = DataGenerator(encoder, 1024, '../data/data_with_value/dataset', '../data/encoded_data', mode, 0.2,
    #                           max_train_npz=1000)
    generator = DataGenerator(
        encoder, 1024, '../data/rl_mcts/generation01_manual/dataset',
        '../data/rl_mcts/generation01_manual/encoded_data', mode, 0.2)

    # Simple network
    prepare_tf_custom_objects()
    model_generator = ac_simple1.ACSimple1(mode, dropout_rate=dropout_rate, data_format="channels_last")
    # model_generator = ac_simple3.ACSimple3(mode, dropout_rate=dropout_rate, data_format="channels_last")
    # model_generator = ac_residual1.ACResidual1(mode, dropout_rate=dropout_rate)
    model = model_generator.model(encoder.shape(), encoder.num_moves(), optimizer=optimizer)
    # model = load_model('../data/checkpoints/rl_mcts/gen00_policy_ACSimple1Policy_dropout0.3_FourPlaneEncoder_channels_last_epoch_100.h5')
    model = load_model(
        '../data/checkpoints/rl_mcts/gen01_policy_new_only_ACSimple1Policy_dropout0.3_FourPlaneEncoder_channels_last_epoch_42.h5')
    network_name = f'{model_generator.name()}_{encoder.name()}'

    # Make network name
    model.summary()
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
            ModelCheckpoint(f'../data/checkpoints/gen01_policy_new_only_{network_name}_epoch_{{epoch}}.h5')
            # ModelCheckpoint(f'../data/checkpoints/test_{{epoch}}.h5')
        ]
    )
