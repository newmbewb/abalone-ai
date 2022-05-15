import os
import random

import h5py
import numpy as np

__all__ = [
    'ExperienceCollector',
    'ExperienceBuffer',
    'combine_experience',
    'load_experience',
]


class ExperienceSaver:
    def __init__(self, exp_dir, experience_per_file, compression=None):
        self.exp_dir = exp_dir
        self.experience_per_file = experience_per_file
        self.h5file_format = os.path.join(self.exp_dir, f'experiences_{random.random()}_%d.h5')
        self.h5file_index = 0
        self.states = []
        self.actions = []
        self.values = []
        self.advantages = []
        self.compression = compression

    def save_data(self, states, actions, values, advantages):
        self.states += states
        self.actions += actions
        self.values += values
        self.advantages += advantages
        while len(self.states) > self.experience_per_file:
            self.save_as_file(self.experience_per_file)

    def save_as_file(self, length=None):
        if length is None:
            length = len(self.states)
        if length <= 0:
            return
        buffer = ExperienceBuffer(self.states[:length],
                                  self.actions[:length],
                                  self.values[:length],
                                  self.advantages[:length])

        self.states = self.states[length:]
        self.actions = self.actions[length:]
        self.values = self.values[length:]
        self.advantages = self.advantages[length:]

        self.h5file_index += 1
        with h5py.File(self.h5file_format % self.h5file_index, 'w') as experience_outf:
            buffer.serialize(experience_outf, compression=self.compression)

    def __del__(self):
        self.save_as_file()


class ExperienceCollector:
    def __init__(self, saver: ExperienceSaver):
        self.states = []
        self.actions = []
        self.target_values = []
        self.advantages = []

        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []
        self._current_episode_probability = []
        self._current_episode_opp_probability = []
        self.saver = saver

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def record_decision(self, state, action, estimated_value, probability):
        self._current_episode_states.append(state)
        self._current_episode_actions.append(action)
        self._current_episode_estimated_values.append(estimated_value)
        self._current_episode_probability.append(probability)
        self._current_episode_opp_probability.append(1)

    def record_opp_probability(self, probability):
        turn = len(self._current_episode_states) - 1
        if turn >= 0:
            self._current_episode_opp_probability[turn] = probability

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        states = []
        advantage = []
        actions = []
        target_value = []
        last_new_value = reward
        for turn in range(num_states)[::-1]:
            states.append(self._current_episode_states[turn])
            actions.append(self._current_episode_actions[turn])
            prob_opp = self._current_episode_opp_probability[turn]
            pp = self._current_episode_probability[turn] * prob_opp
            estimated_value = self._current_episode_estimated_values[turn]
            new_value = (1 - pp) * estimated_value + pp * last_new_value
            target_value.append(new_value)
            advantage.append(prob_opp * (last_new_value - estimated_value))
            last_new_value = new_value

        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []
        self._current_episode_probability = []
        self._current_episode_opp_probability = []

        self.states += states
        self.actions += actions
        self.target_values += target_value
        self.advantages += advantage

    def save_as_file(self):
        self.saver.save_data(self.states, self.actions, self.target_values, self.advantages)


class ExperienceBuffer:
    def __init__(self, states, actions, values, advantages):
        self.states = states
        self.actions = actions
        self.values = values
        self.advantages = advantages

    def serialize(self, h5file, compression=None):
        h5file.create_group('experience')
        # Compression: None, 'gzip', 'lzf'
        h5file['experience'].create_dataset('states', data=self.states, compression=compression)
        h5file['experience'].create_dataset('actions', data=self.actions)
        h5file['experience'].create_dataset('values', data=self.values)
        h5file['experience'].create_dataset('advantages', data=self.advantages)


def combine_experience(collectors):
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_actions = np.concatenate([np.array(c.actions) for c in collectors])
    combined_values = np.concatenate([np.array(c.values) for c in collectors])
    combined_advantages = np.concatenate([
        np.array(c.advantages) for c in collectors])

    return ExperienceBuffer(
        combined_states,
        combined_actions,
        combined_values,
        combined_advantages)


def load_experience(h5file):
    return ExperienceBuffer(
        states=np.array(h5file['experience']['states']),
        actions=np.array(h5file['experience']['actions']),
        values=np.array(h5file['experience']['values']),
        advantages=np.array(h5file['experience']['advantages']))
