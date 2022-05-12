import os
import time

import h5py
import numpy as np
from keras.models import clone_model
from scipy.stats import binom_test

from dlabalone.abltypes import Player
from dlabalone.rl.experience import load_experience
from dlabalone.rl.move_selectors.exponential import ExponentialMoveSelector
from dlabalone.rl.simulate import experience_simulation


def predict_convertor_separated_ac(predict_output_batch_list, index):
    move_probs = predict_output_batch_list[0][index]
    estimated_value = predict_output_batch_list[1][index]
    return move_probs, estimated_value


def train_function_separated_ac(models, encoder, dataset):
    states, actions, rewards, advantages = dataset
    model_policy = models[0]
    model_value = models[1]

    n = states.shape[0]
    num_moves = encoder.num_moves()
    policy_target = np.zeros((n, num_moves))
    value_target = np.zeros((n,))
    for i in range(n):
        action = actions[i]
        reward = rewards[i]
        policy_target[i][action] = advantages[i]
        value_target[i] = reward

    print(states.shape)
    model_policy.fit(states, policy_target, batch_size=n, epochs=1)
    model_value.fit(states, value_target, batch_size=n, epochs=1)


class DataBuffer:
    def __init__(self):
        self.states = None
        self.actions = None
        self.rewards = None
        self.advantages = None

    def append_h5file(self, file):
        with h5py.File(file, 'r') as h5file:
            buffer = load_experience(h5file)
        if self.states is None:
            self.states = buffer.states
            self.actions = buffer.actions
            self.rewards = buffer.rewards
            self.advantages = buffer.advantages
        else:
            self.states = np.append(self.states, buffer.states, axis=0)
            self.actions = np.append(self.actions, buffer.actions, axis=0)
            self.rewards = np.append(self.rewards, buffer.rewards, axis=0)
            self.advantages = np.append(self.advantages, buffer.advantages, axis=0)

    def length(self):
        return self.states.shape[0]

    def get_next_batch(self, batch_size):
        ret = (self.states[:batch_size],
               self.actions[:batch_size],
               self.rewards[:batch_size],
               self.advantages[:batch_size],
               )
        self.states = self.states[batch_size:]
        self.actions = self.actions[batch_size:]
        self.rewards = self.rewards[batch_size:]
        self.advantages = self.advantages[batch_size:]
        return ret


def _train_model(models, encoder, dataset, train_fn):
    # states, actions, rewards, advantages = dataset
    # cloned_models = _clone_models(models)
    # train_fn(cloned_models, encoder, dataset)
    train_fn(models, encoder, dataset)


def _train_on_experience(models, encoder, exp_dir, batch_size, train_fn):
    data_buffer = DataBuffer()
    for file in os.listdir(exp_dir):
        data_buffer.append_h5file(os.path.join(exp_dir, file))
        if data_buffer.length() < batch_size:
            continue
        _train_model(models, encoder, data_buffer.get_next_batch(batch_size), train_fn)

    # Train ggodari things
    if data_buffer.length() > 0:
        _train_model(models, encoder, data_buffer.get_next_batch(batch_size), train_fn)


def _clone_models(models):
    cloned_model_list = []
    for model in models:
        cloned_model = clone_model(model)
        cloned_model.set_weights(model.get_weights())
        cloned_model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
        cloned_model_list.append(cloned_model)
    return cloned_model_list


def _evaluate_model(models, models_prev, encoder, predict_convertor, num_games):
    wins = 0
    losses = 0
    selector = ExponentialMoveSelector(temperature=0)
    stat_list_black = experience_simulation(num_games // 2,
                                            models, models_prev,
                                            encoder, encoder,
                                            selector, selector,
                                            predict_convertor, predict_convertor,
                                            populate_games=False)
    for stat in stat_list_black:
        if stat['winner'] == Player.black:
            wins += 1
        else:
            losses += 1
    stat_list_white = experience_simulation(num_games // 2,
                                            models_prev, models,
                                            encoder, encoder,
                                            selector, selector,
                                            predict_convertor, predict_convertor)
    for stat in stat_list_white:
        if stat['winner'] == Player.white:
            wins += 1
        else:
            losses += 1
    return wins, losses


def train_ac(models, encoder, predict_convertor, train_fn, exp_dir, model_directory,
             max_generation=10000, num_games=1024, comparison_game_count=512, p_value=0.05,
             move_selectors=None, batch_size=1024, model_name='model', populate_games=True):
    if move_selectors is None:
        move_selectors = [ExponentialMoveSelector()]

    move_selector_index = 0
    episode_count = 0
    generation = 1
    total_games = 0
    models_prev = _clone_models(models)
    for _ in range(max_generation):
        cur_move_selector = move_selectors[move_selector_index]
        move_selector_index = (move_selector_index + 1) % len(move_selectors)

        # Simulate black games
        time_start = time.time()
        stat_list = experience_simulation(num_games,
                                          models, models,
                                          encoder, encoder,
                                          cur_move_selector, cur_move_selector,
                                          predict_convertor, predict_convertor,
                                          exp_dir=exp_dir, populate_games=populate_games)
        print(f'Simulation time: {time.time() - time_start} seconds')
        episode_count += 1
        total_games += num_games
        cur_move_selector.temperature_down()

        # Train games
        time_start = time.time()
        _train_on_experience(models, encoder, exp_dir, batch_size, train_fn)
        print(f'Training time: {time.time() - time_start} seconds')
        break

        # Evaluate new model
        time_start = time.time()
        wins, losses = _evaluate_model(models, models_prev, encoder, predict_convertor, comparison_game_count)
        print(f'Evaluating time: {time.time() - time_start} seconds')

        completed_game_count = wins + losses
        win_rate = wins / completed_game_count
        if wins / completed_game_count > 0.5 and binom_test(wins, completed_game_count, 0.5) < p_value:
            print(f'New model accepted (Win rate: {win_rate})! ' +
                  f'Generation {generation}, after {episode_count} episodes. {total_games} games')

            filename = f'{model_name}_{generation}gen_{total_games}games_%d.h5'
            filepath = os.path.join(model_directory, filename)
            for i, model in enumerate(models):
                path = filepath % i
                # https://www.tensorflow.org/guide/keras/save_and_serialize?hl=ko#weights-only_saving_in_savedmodel_format
                # If you want to save as checkpoint: https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko
                model.save(path)
            models_prev = _clone_models(models)

            episode_count = 0
            generation += 1
        else:
            print(f'Win rate: {win_rate}.. Keep learning')



########
# This commented out function is for advanced training, which train only when a explorer wins
# def train_ac(models, models_bsf, encoder, encoder_bsf, predict_convertor, predict_convertor_bsf, exp_dir,
#              max_generation=None, num_games=1024, comparison_game_count=128, next_generation_win_rate=0.55,
#              move_selectors=None):
#     if max_generation is None:
#         max_generation = 10000
#     if move_selectors is None:
#         move_selectors = [ExponentialMoveSelector()]
#
#     move_selector_bsf = ExponentialMoveSelector(3)
#     move_selector_index = 0
#     models_list = [models, models_bsf]
#     encoder_list = [encoder, encoder_bsf]
#     predict_convertor_list = [predict_convertor, predict_convertor_bsf]
#     for _ in range(max_generation):
#         cur_move_selector = move_selectors[move_selector_index]
#         move_selector_index = (move_selector_index + 1) % len(move_selectors)
#
#         # Simulate black games
#         stat_list_black = experience_simulation(num_games // 2,
#                                                 models_list[0], models_list[1],
#                                                 encoder_list[0], encoder_list[1],
#                                                 cur_move_selector, move_selector_bsf,
#                                                 predict_convertor_list[0], predict_convertor_list[1],
#                                                 exp_dir)
#         models_list = models_list[::-1]
#         encoder_list = encoder_list[::-1]
#         predict_convertor_list = predict_convertor_list[::-1]
#         # Simulate white games
#         stat_list_white = experience_simulation(num_games // 2,
#                                                 models_list[0], models_list[1],
#                                                 encoder_list[0], encoder_list[1],
#                                                 cur_move_selector, move_selector_bsf,
#                                                 predict_convertor_list[0], predict_convertor_list[1],
#                                                 exp_dir)
#         models_list = models_list[::-1]
#         encoder_list = encoder_list[::-1]
#         predict_convertor_list = predict_convertor_list[::-1]




