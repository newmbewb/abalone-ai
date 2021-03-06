import math
import os
import random
import time

import h5py
import numpy as np
from keras.models import clone_model
from keras.models import load_model
from scipy.stats import binom_test

from dlabalone.abltypes import Player
from dlabalone.rl.experience import load_experience, ExperienceBuffer
from dlabalone.rl.move_selectors.eps_greedy import EpsilonGreedyMoveSelector
from dlabalone.rl.move_selectors.exponential import ExponentialMoveSelector
from dlabalone.rl.ppo import train_policy
from dlabalone.rl.simulate import experience_simulation


##################################################
# Prediction convertors
def predict_convertor_separated_ac(predict_output_batch_list, index):
    move_probs = predict_output_batch_list[0][index]
    estimated_value = predict_output_batch_list[1][index]
    return move_probs, estimated_value


def predict_convertor_single_model(predict_output_batch_list, index):
    move_probs = predict_output_batch_list[0][index]
    return move_probs, 0


##################################################
# Train functions
def train_function_separated_ac(models, encoder, dataset, history, remove_negative=False):
    states, actions, value, advantages, probabilities = dataset
    model_policy = models[0]
    model_value = models[1]

    # Train value
    n = states.shape[0]
    value_target = value
    history_value = model_value.fit(states, value_target, batch_size=n, epochs=1, verbose=0)

    # Train policy: Remove negative weights first (PPO)
    probabilities = probabilities.astype(np.single)
    train_policy(model_policy, states, actions, probabilities, advantages, encoder.num_moves())

    ##############################
    # # Train policy: Remove negative weights first
    # if remove_negative:
    #     states = states[advantages > 0]
    #     n = states.shape[0]
    #     actions = actions[advantages > 0]
    #     advantages = advantages[advantages > 0]
    #
    # num_moves = encoder.num_moves()
    # policy_target = np.zeros((n, num_moves))
    #
    # sample_weight = advantages
    # for i in range(n):
    #     action = actions[i]
    #     policy_target[i][action] = 1
    #
    # # Normalize
    # m = len(sample_weight) / np.sum(np.abs(sample_weight))
    # sample_weight *= m
    # history_policy = model_policy.fit(states, policy_target, sample_weight=sample_weight, batch_size=n, epochs=1,
    #                                   verbose=0)
    ##############################

    # Save history
    # policy_loss = history.get('policy_loss', [])
    # policy_accuracy = history.get('policy_accuracy', [])
    value_loss = history.get('value_loss', [])
    value_mse = history.get('value_mse', [])

    # policy_loss += history_policy.history['loss']
    # policy_accuracy += history_policy.history['accuracy']
    value_loss += history_value.history['loss']
    value_mse += history_value.history['mean_squared_error']

    # history['policy_loss'] = policy_loss
    # history['policy_accuracy'] = policy_accuracy
    history['value_loss'] = value_loss
    history['value_mse'] = value_mse


class DataBuffer:
    def __init__(self, exp_dir=None, compression=None):
        self.exp_dir = exp_dir
        self.states = None
        self.actions = None
        self.values = None
        self.advantages = None
        self.probabilities = None
        self.savefile_basename = None
        if self.exp_dir:
            self.savefile_basename = os.path.join(self.exp_dir, f'packed_experiences_{random.random()}_%d.h5')
        self.savefile_index = 0
        self.compression = compression

    def consume_h5file(self, file):
        fail = False
        with h5py.File(file, 'r') as h5file:
            try:
                buffer = load_experience(h5file)
            except KeyError:
                fail = True
        if fail:
            os.unlink(file)
            return
        if self.states is None:
            self.states = buffer.states
            self.actions = buffer.actions
            self.values = buffer.values
            self.advantages = buffer.advantages
            self.probabilities = buffer.probabilities
        else:
            self.states = np.append(self.states, buffer.states, axis=0)
            self.actions = np.append(self.actions, buffer.actions, axis=0)
            self.values = np.append(self.values, buffer.values, axis=0)
            self.advantages = np.append(self.advantages, buffer.advantages, axis=0)
            self.probabilities = np.append(self.probabilities, buffer.probabilities, axis=0)
        while True:
            try:
                os.unlink(file)
                break
            except PermissionError:
                print('Failed to delete file due to permission error. retry...')
                time.sleep(1)

    def iter_experience(self, files, experience_count, shuffle=False):
        for file in files:
            self.consume_h5file(file)
            if self.length() < experience_count:
                continue
            if shuffle:
                self.shuffle()
            yield self.get_next_batch(experience_count)
        if self.length() > 0:
            if shuffle:
                self.shuffle()
            yield self.get_next_batch(experience_count)

    def length(self):
        return self.states.shape[0]

    def shuffle(self):
        s = list(range(self.length()))
        random.shuffle(s)

        state_list = []
        action_list = []
        value_list = []
        advantage_list = []
        probability_list = []

        for index in s:
            state_list.append(self.states[index])
            action_list.append(self.actions[index])
            value_list.append(self.values[index])
            advantage_list.append(self.advantages[index])
            probability_list.append(self.probabilities[index])

        self.states = np.array(state_list)
        self.actions = np.array(action_list)
        self.values = np.array(value_list)
        self.advantages = np.array(advantage_list)
        self.probabilities = np.array(probability_list)

    def get_next_batch(self, batch_size):
        ret = (self.states[:batch_size],
               self.actions[:batch_size],
               self.values[:batch_size],
               self.advantages[:batch_size],
               self.probabilities[:batch_size],
               )
        self.states = self.states[batch_size:]
        self.actions = self.actions[batch_size:]
        self.values = self.values[batch_size:]
        self.advantages = self.advantages[batch_size:]
        self.probabilities = self.probabilities[batch_size:]
        return ret

    def save(self, experiences):
        filename = self.savefile_basename % self.savefile_index
        self.savefile_index += 1
        with h5py.File(filename, 'w') as experience_outf:
            exp_buffer = ExperienceBuffer(*experiences)
            exp_buffer.serialize(experience_outf, compression=self.compression)


def train_ac(load_models, encoder, predict_convertor, train_fn, exp_dir, model_directory,
             max_generation=10000, num_games=1024, comparison_game_count=512, p_value=0.05,
             move_selectors=None, batch_size=1024, model_name='model', populate_games=True, experience_per_file=65536,
             compression=None):
    if move_selectors is None:
        move_selectors = [ExponentialMoveSelector()]

    move_selector_index = 0
    episode_count = 0
    generation = 1
    total_games = 0
    models_prev = load_models()
    models = load_models()
    move_selector_greedy = EpsilonGreedyMoveSelector(temperature=0)
    for _ in range(max_generation):
        cur_move_selector = move_selectors[move_selector_index]
        move_selector_index = (move_selector_index + 1) % len(move_selectors)

        # Simulate black games
        print(f'Simulating... ({cur_move_selector})')
        time_start = time.time()
        stat_list = experience_simulation(num_games,
                                          models, models,
                                          encoder, encoder,
                                          cur_move_selector, move_selector_greedy,
                                          predict_convertor, predict_convertor,
                                          exp_dir=exp_dir, populate_games=populate_games,
                                          experience_per_file=experience_per_file, compression=compression)
        print(f'Simulation time: {time.time() - time_start} seconds')
        episode_count += 1
        total_games += num_games
        # cur_move_selector.temperature_down()

        # Shuffle data
        print(f'Shuffling data...')
        time_start = time.time()
        _shuffle_experience(exp_dir, experience_per_file, compression=compression)
        print(f'Shuffling data time: {time.time() - time_start} seconds')

        # Train games
        print('Training...')
        time_start = time.time()
        _train_on_experience(models, encoder, exp_dir, batch_size, train_fn, compression=compression)
        print(f'Training time: {time.time() - time_start} seconds')

        # Evaluate new model
        print('Evaluating...')
        time_start = time.time()
        wins, losses, draws = _evaluate_model(models, models_prev, encoder, predict_convertor, comparison_game_count)
        print(f'Evaluating time: {time.time() - time_start} seconds')
        print(f'Result: Wins/Losses/Draws = {wins}/{losses}/{draws}')

        completed_game_count = wins + losses
        if completed_game_count == 0:
            continue
        win_rate = wins / completed_game_count
        current_p_value = binom_test(wins, completed_game_count, 0.5)
        if wins / completed_game_count > 0.5 and current_p_value < p_value:
            print(f'New model accepted(Win rate: {win_rate*100:.3f} %)! P-value: {current_p_value:.5f}\n' +
                  f'Generation {generation}, after {episode_count} episodes. {total_games} games')

            filename = f'{model_name}_Generation_{generation}__{total_games}games_%d.h5'
            filepath = os.path.join(model_directory, filename)
            model_paths = []
            for i, model in enumerate(models):
                path = filepath % i
                model_paths.append(path)
                # https://www.tensorflow.org/guide/keras/save_and_serialize?hl=ko#weights-only_saving_in_savedmodel_format
                # If you want to save as checkpoint: https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko
                model.save(path)

            def new_load_models_func():
                models = []
                for path in model_paths:
                    model = load_model(path)
                    models.append(model)
                return models
            load_models = new_load_models_func
            models_prev = load_models()
            models = load_models()

            episode_count = 0
            generation += 1
        # elif wins / completed_game_count < 0.5 and current_p_value < p_value:
        #     print(f'Too low win rate! Win rate: {win_rate*100:.3f} % (P-value: {current_p_value:.5f})\n' +
        #           f'Revert to previous model..')
        #     episode_count = 0
        #     models = load_models()
        else:
            print(f'Win rate: {win_rate*100:.3f} % (P-value: {current_p_value:.5f}). Keep learning. ' +
                  f'Game count: {total_games}')


def _get_random_file_list(exp_dir):
    file_list = os.listdir(exp_dir)
    random.shuffle(file_list)
    return map(lambda f: os.path.join(exp_dir, f), file_list)


def _shuffle_experience(exp_dir, experience_per_file, compression=None):
    # Pack experiences into a single file
    data_buffer = DataBuffer(exp_dir=exp_dir, compression=compression)
    iter_count = 0
    file_count = 1000

    while iter_count < math.log(file_count, 2) + 2:
        iter_count += 1
        file_count = 0
        for experiences in data_buffer.iter_experience(_get_random_file_list(exp_dir), experience_per_file * 2,
                                                       shuffle=True):
            # Break experience into two pieces
            experiences1 = tuple()
            experiences2 = tuple()
            for data in experiences:
                experiences1 += (data[:experience_per_file],)
                experiences2 += (data[experience_per_file:],)
            data_buffer.save(experiences1)
            data_buffer.save(experiences2)
            file_count += 2


def _train_on_experience(models, encoder, exp_dir, batch_size, train_fn, compression=None):
    data_buffer = DataBuffer(compression=compression)
    files = map(lambda f: os.path.join(exp_dir, f), os.listdir(exp_dir))
    history = {}
    for experiences in data_buffer.iter_experience(files, batch_size):
        train_fn(models, encoder, experiences, history)
    print('=' * 80)
    for key in history:
        print(f'{key:18s}', end='')
        for value in history[key]:
            print(f' {value:8.4f}', end='')
        print('')
    print('=' * 80)


def _evaluate_model(models, models_prev, encoder, predict_convertor, num_games):
    wins = 0
    losses = 0
    draws = 0
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
        elif stat['winner'] == Player.white:
            losses += 1
        else:
            draws += 1

    stat_list_white = experience_simulation(num_games // 2,
                                            models_prev, models,
                                            encoder, encoder,
                                            selector, selector,
                                            predict_convertor, predict_convertor,
                                            populate_games=False)
    for stat in stat_list_white:
        if stat['winner'] == Player.white:
            wins += 1
        elif stat['winner'] == Player.black:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws


