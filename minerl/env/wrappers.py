from minerl.env import spaces

import copy
import gym
import numpy as np
import imageio
import logging

class FailedToReachTarget(Exception):
    pass


class GridWorldWrapper(gym.Wrapper):
    """A wrapper around MineRLEnv so that it can be treated as a discrete gridworld.
    """
    action_scale = 0.8

    def __init__(self, env, max_inner_steps=25, threshold=0.01, grid_mins=(-2, -1, -2), grid_maxs=(2, -1, 2)):
        super().__init__(env)

        self.grid_mins = grid_mins
        self.grid_maxs = grid_maxs

        self.max_inner_steps = max_inner_steps
        self.threshold = threshold

        self.attacking = 0

        self.action_space = spaces.Dict(spaces={
            "forward": spaces.Discrete(2), 
            "back": spaces.Discrete(2), 
            "left": spaces.Discrete(2), 
            "right": spaces.Discrete(2),
            "jump": spaces.Discrete(2),
            "attack": spaces.Discrete(2),
        })

    def step_in_env(self, action):
        action['attack'] = self.attacking
        return self.env.step(action)

    def step_to_target(self, target_position):
        for _ in range(self.max_inner_steps):
            inner_action = self.get_action_towards_target(self.position, target_position)
            
            obs, reward, done, info = self.step_in_env(inner_action)
            self.position = np.array([obs['XPos'], obs['YPos'], obs['ZPos']])

            if done:
                return obs, reward, done, info

            if np.sqrt(np.sum((self.position - target_position)**2)) < self.threshold:
                return obs, reward, done, info

        dtp = self.discretize_position(target_position)
        dap = self.discretize_position(self.position)
        if not (dtp[0] == dap[0] and dtp[2] == dap[2]):
            logging.debug("Failed to reach discrete target. Running into wall?")

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.position = np.array([obs['XPos'], obs['YPos'], obs['ZPos']])

        # Get to a grid location
        target_position = self.get_target_position(self.position, self.action_space.no_op())

        obs, _, _, _ = self.step_to_target(target_position)

        self.position = np.array([obs['XPos'], obs['YPos'], obs['ZPos']])
        obs['position'] = self.discretize_position(self.position)
        obs['grid_arr'] = self.grid_to_array(obs['grid'])

        return obs

    def step(self, action):
        if action['jump']:
            jump_action = self.env.action_space.no_op()
            jump_action['jump'] = 1
            self.step_in_env(jump_action)

        self.attacking = action['attack']

        target_position = self.get_target_position(self.position, action)

        obs, reward, done, debug_info = self.step_to_target(target_position)

        if not done:
            obs, reward, done, debug_info = self.finish_stepping()

        self.position = np.array([obs['XPos'], obs['YPos'], obs['ZPos']])
        obs['position'] = self.discretize_position(self.position)
        obs['grid_arr'] = self.grid_to_array(obs['grid'])

        return obs, reward, done, debug_info

    def discretize_position(self, position):
        new_p = []
        # print("position:", position)
        for p in position:
            if p.is_integer():
                new_p.append(p - 1)
            else:
                new_p.append(np.floor(p))
        discretized_position = np.array(new_p, dtype=np.int64)
        # print("discretized_position:", discretized_position)
        return discretized_position

    def grid_to_array(self, grid):
        shape = 1 + np.subtract(self.grid_maxs, self.grid_mins)

        arr = np.array(grid).reshape((shape[2], shape[0], shape[1]), order='F')
        arr = np.moveaxis(arr, 0, -1)
        arr = np.rot90(arr, k=2, axes=(0, 2))
        return arr

    def get_target_position(self, position, action):
        action_directions = np.zeros(3)

        x_change = action['left'] - action['right']
        z_change = action['forward'] - action['back']

        action_directions[0] = x_change
        action_directions[2] = z_change

        current_discrete_position = (self.position - 0.5).round() + 0.5
        target_position = current_discrete_position + action_directions

        return target_position

    def finish_stepping(self):
        # Always let settle b/c jumping observations are unreliable
        action = self.env.action_space.no_op()
        for _ in range(5):
            obs, reward, done, debug_info = self.step_in_env(action)

            if done:
                break
        return obs, reward, done, debug_info

    def get_action_towards_target(self, position, target_position):
        action = self.env.action_space.no_op()

        if abs(target_position[0] - position[0]) > self.threshold:
            action['strafe'] = self.action_scale * (position[0] - target_position[0])

        if abs(target_position[2] - position[2]) > self.threshold:
            action['move'] = self.action_scale * (target_position[2] - position[2])

        return action


class DiscretePositionWrapper(gym.Wrapper):

    def __init__(self, env, grid_mins=(-2, -1, -2), grid_maxs=(2, -1, 2)):
        super().__init__(env)

        self.grid_mins = grid_mins
        self.grid_maxs = grid_maxs

    def reset(self):
        obs = self.env.reset()
        self.position = np.array([obs['XPos'], obs['YPos'], obs['ZPos']])
        obs['position'] = self.discretize_position(self.position)
        obs['grid_arr'] = self.grid_to_array(obs['grid'])
        return obs

    def step(self, action):
        obs, reward, done, debug_info = self.env.step(action)
        self.position = np.array([obs['XPos'], obs['YPos'], obs['ZPos']])
        obs['position'] = self.discretize_position(self.position)
        obs['grid_arr'] = self.grid_to_array(obs['grid'])
        return obs, reward, done, debug_info

    def discretize_position(self, position):
        new_p = []
        # print("position:", position)
        for p in position:
            if p.is_integer():
                new_p.append(p - 1)
            else:
                new_p.append(np.floor(p))
        discretized_position = np.array(new_p, dtype=np.int64)
        # print("discretized_position:", discretized_position)
        return discretized_position

    def grid_to_array(self, grid):
        shape = 1 + np.subtract(self.grid_maxs, self.grid_mins)

        arr = np.array(grid).reshape((shape[2], shape[0], shape[1]), order='F')
        arr = np.moveaxis(arr, 0, -1)
        arr = np.rot90(arr, k=2, axes=(0, 2))
        return arr



class VideoWrapper(gym.Wrapper):
    def __init__(self, env, out_path, fps=30):
        super().__init__(env)
        self.out_path = out_path
        self.fps = fps

    def reset(self):
        obs = super().reset()

        self.images = []
        img = super().render()
        self.images.append(img)

        return obs

    def step(self, action):
        obs, reward, done, debug_info = super().step(action)

        img = super().render()
        self.images.append(img)

        return obs, reward, done, debug_info

    def close(self):
        imageio.mimsave(self.out_path, self.images, fps=self.fps)
        print("Wrote out video to {}.".format(self.out_path))
        return super().close()


