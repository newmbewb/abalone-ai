import os

from dlabalone.encoders.base import get_encoder_by_name
from dlabalone.rl.move_selectors.eps_greedy import EpsilonGreedyMoveSelector
from dlabalone.rl.move_selectors.exponential import ExponentialMoveSelector
from dlabalone.rl.trainer import train_ac, predict_convertor_separated_ac, train_function_separated_ac
from keras.models import load_model
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':
    # Load model & encoder
    # with tf.device('/device:CPU:0'):
    model_policy = load_model(
        '../data/checkpoints/ACSimple1/ACSimple1Policy_dropout0.1_AlphaAbaloneEncoder_epoch_10.h5')
    model_value = load_model(
        '../data/checkpoints/ACSimple1/ACSimple1Value_dropout0.2_AlphaAbaloneEncoder_epoch_28.h5')
    # model_policy = load_model(
    #     '../data/rl/models/model_1gen_1536games_0_old.h5')
    # model_value = load_model(
    #     '../data/rl/models/model_1gen_1536games_1_old.h5')

    model_policy.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])
    model_value.compile(loss='mean_squared_error', optimizer=SGD(), metrics=['mean_squared_error'])
    models = [model_policy, model_value]
    encoder = get_encoder_by_name('alpha_abalone', 5, '')

    # Load convertor, train_fn, move_selectors
    predict_convertor = predict_convertor_separated_ac
    train_fn = train_function_separated_ac
    move_selectors = [EpsilonGreedyMoveSelector(temperature=0)]
    # move_selectors = [ExponentialMoveSelector(temperature=0)]

    # exp_dir & model_directory
    exp_dir = '../data/rl/experience'
    model_directory = '../data/rl/models'

    # train_ac(models, encoder, predict_convertor, train_fn, exp_dir, model_directory,
    #          move_selectors=move_selectors, num_games=2, comparison_game_count=64, populate_games=False)
    train_ac(models, encoder, predict_convertor, train_fn, exp_dir, model_directory,
             move_selectors=move_selectors, num_games=256, comparison_game_count=64, populate_games=False)
