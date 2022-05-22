import os

from dlabalone.encoders.base import get_encoder_by_name
from dlabalone.networks import ac_simple1
from dlabalone.rl.move_selectors.eps_greedy import EpsilonGreedyMoveSelector
from dlabalone.rl.move_selectors.exponential import ExponentialMoveSelector
from dlabalone.rl.trainer import train_ac, predict_convertor_separated_ac, train_function_separated_ac
from keras.models import load_model
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':
    # model & encoder loader
    def load_models():
        # model_generator = ac_simple1.ACSimple1('policy', dropout_rate=0.1, data_format="channels_last")
        # model_policy = model_generator.model(encoder.shape(), encoder.num_moves())
        model_policy = load_model(
            '../data/checkpoints/ACSimple1Policy_dropout0.1_FourPlaneEncoder_channels_last_epoch_3.h5')
        model_value = load_model(
            '../data/checkpoints/models/ACSimple1Value_dropout0.1_FourPlaneEncoder_channels_last_epoch_100.h5')
        return [model_policy, model_value]

    # model_policy.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])
    # model_value.compile(loss='mean_squared_error', optimizer=SGD(), metrics=['mean_squared_error'])
    encoder = get_encoder_by_name('fourplane', 5, '', data_format='channels_last')

    # Load convertor, train_fn, move_selectors
    predict_convertor = predict_convertor_separated_ac
    train_fn = train_function_separated_ac
    # move_selectors = [EpsilonGreedyMoveSelector(temperature=0.2)]
    move_selectors = [ExponentialMoveSelector(temperature=0.5)]

    # exp_dir & model_directory
    exp_dir = '../data/rl/experience'
    model_directory = '../data/rl/models'

    train_ac(load_models, encoder, predict_convertor, train_fn, exp_dir, model_directory,
             # move_selectors=move_selectors, num_games=128, comparison_game_count=32, populate_games=False,
             # move_selectors=move_selectors, num_games=1, comparison_game_count=2, populate_games=False,
             move_selectors=move_selectors, num_games=32, comparison_game_count=32, populate_games=False,
             compression=None, experience_per_file=32768, batch_size=128)
