import time

import minerl
import itertools
import gym
import sys
import tqdm
import numpy as np


# Helper functions
def _check_shape(num_samples, sample_shape, obs):
    if isinstance(obs, list):
        assert (len(obs)) == num_samples
    elif isinstance(obs, np.ndarray):
        assert (obs.shape[0] == num_samples)
        for i in range(len(obs.shape) - 1):
            assert (obs.shape[i + 1] == sample_shape[i])
    else:
        assert(False, "unsupported box data type")


def _check_space(key, space, observation, correct_len):
    if isinstance(space, minerl.spaces.Dict):
        for k, s in space.spaces.items():
            _check_space(k, s, observation[key], correct_len)
    elif isinstance(space, minerl.spaces.MultiDiscrete):
        # print("MultiDiscrete")
        # print(space.shape)
        # print(observation[key])
        _check_shape(correct_len, space.shape, observation[key])
    elif isinstance(space, minerl.spaces.Box):
        # print("Box")
        # print(space.shape)
        # print(observation[key])
        _check_shape(correct_len, space.shape, observation[key])
    elif isinstance(space, minerl.spaces.Discrete):
        # print("Discrete")
        # print(space.shape)
        # print(observation[key])
        _check_shape(correct_len, space.shape, observation[key])
    elif isinstance(space, minerl.spaces.Enum):
        # print("Enum")
        # print(space.shape)
        # print(observation[key])
        _check_shape(correct_len, space.shape, observation[key])
    else:
        assert(False, "Unsupported dict type")


def test_data(environment='MineRLObtainDiamondDense-v0'):
    d = minerl.data.make(environment, num_workers=8)

    # Iterate through batches of data
    counter = tqdm.tqdm()
    for obs, act, rew, nObs, done in d.sarsd_iter(num_epochs=1, max_sequence_len=128):
        correct_len = len(rew)

        for key, space in d.observation_space.spaces.items():
            _check_space(key, space, obs, correct_len)

        for key, space in d.action_space.spaces.items():
            _check_space(key, space, act, correct_len)

        counter.update(correct_len)

    return counter.n / counter.last_print_t if counter.last_print_n > 0 else 0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        rate = test_data(sys.argv[1])
    else:
        rate = test_data()