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
    def __init__(self, exp_dir, experience_per_file):
        self.exp_dir = exp_dir
        self.experience_per_file = experience_per_file
        self.h5file_format = os.path.join(self.exp_dir, f'experiences_{random.random()}_%d.h5')
        self.h5file_index = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []

    def save_data(self, states, actions, rewards, advantages):
        self.states += states
        self.actions += actions
        self.rewards += rewards
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
                                  self.rewards[:length],
                                  self.advantages[:length])

        self.states = self.states[length:]
        self.actions = self.actions[length:]
        self.rewards = self.rewards[length:]
        self.advantages = self.advantages[length:]

        self.h5file_index += 1
        with h5py.File(self.h5file_format % self.h5file_index, 'w') as experience_outf:
            buffer.serialize(experience_outf)

    def __del__(self):
        self.save_as_file()


class ExperienceCollector:
    def __init__(self, saver: ExperienceSaver):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []
        self.saver = saver

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def record_decision(self, state, action, estimated_value=0):
        self._current_episode_states.append(state)
        self._current_episode_actions.append(action)
        self._current_episode_estimated_values.append(estimated_value)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.actions += self._current_episode_actions
        self.rewards += [reward for _ in range(num_states)]

        for i in range(num_states):
            advantage = reward - self._current_episode_estimated_values[i]
            self.advantages.append(advantage)

        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def save_as_file(self):
        self.saver.save_data(self.states, self.actions, self.rewards, self.advantages)


class ExperienceBuffer:
    def __init__(self, states, actions, rewards, advantages):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages

    def serialize(self, h5file):
        h5file.create_group('experience')
        compression = 'lzf'  # None, 'gzip', 'lzf'
        h5file['experience'].create_dataset('states', data=self.states, compression=compression)
        h5file['experience'].create_dataset('actions', data=self.actions, compression=compression)
        h5file['experience'].create_dataset('rewards', data=self.rewards, compression=compression)
        h5file['experience'].create_dataset('advantages', data=self.advantages, compression=compression)


def combine_experience(collectors):
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_actions = np.concatenate([np.array(c.actions) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
    combined_advantages = np.concatenate([
        np.array(c.advantages) for c in collectors])

    return ExperienceBuffer(
        combined_states,
        combined_actions,
        combined_rewards,
        combined_advantages)


def load_experience(h5file):
    return ExperienceBuffer(
        states=np.array(h5file['experience']['states']),
        actions=np.array(h5file['experience']['actions']),
        rewards=np.array(h5file['experience']['rewards']),
        advantages=np.array(h5file['experience']['advantages']))
