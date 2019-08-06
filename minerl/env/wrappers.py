import gym
import numpy as np



class GridWorldWrapper(gym.Wrapper):
    """A wrapper around MineRLEnv so that it can be treated as a discrete gridworld.
    """
    def __init__(self, env, max_inner_steps=25, threshold=0.01):
        super().__init__(env)
        self.env = env
        self.max_inner_steps = max_inner_steps
        self.threshold = threshold

    def step_in_env(self, action, advance_time=False):
        elapsed_steps = self.env._elapsed_steps
        out = self.env.step(action)
        if advance_time:
            self.env._elapsed_steps = elapsed_steps + 1
        else:
            self.env._elapsed_steps = elapsed_steps
        return out

    def step_to_target(self, target_position):
        for _ in range(self.max_inner_steps):
            inner_action = self.get_action_towards_target(self.position, target_position)
            obs, reward, done, info = self.step_in_env(inner_action)
            self.position = np.array([obs['XPos'], obs['YPos'], obs['ZPos']])

            if np.sqrt(np.sum((self.position - target_position)**2)) < self.threshold:
                break

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()

        # Get to a grid location
        self.position = np.array([obs['XPos'], obs['YPos'], obs['ZPos']])
        target_position = self.get_target_position(self.position, self.action_space.noop())

        obs, _, _, _ = self.step_to_target(target_position)

        obs['position'] = self.discretize_position(self.position)
        return obs

    def step(self, action):
        obs, reward, done, info = self.step_in_env(action, advance_time=True)
        self.position = np.array([obs['XPos'], obs['YPos'], obs['ZPos']])

        target_position = self.get_target_position(self.position, action)

        obs, reward, done, debug_info = self.step_to_target(target_position)
        obs['position'] = self.discretize_position(self.position)

        return obs, reward, done, debug_info

    def discretize_position(self, position):
        print('cont position:', position)
        return (position - 0.5).round().astype(np.int64)

    def get_target_position(self, position, action):
        action_directions = np.zeros(3)

        x_change = action['left'] - action['right']
        z_change = action['forward'] - action['back']

        action_directions[0] = x_change
        action_directions[2] = z_change

        current_discrete_position = (self.position - 0.5).round() + 0.5
        target_position = current_discrete_position + action_directions

        return target_position

    def get_action_towards_target(self, position, target_position):
        action = self.action_space.noop()

        if abs(target_position[2] - position[2]) > self.threshold:
            if target_position[2] > position[2]:
                action['forward'] = 1
            else:
                action['back'] = 1

        if abs(target_position[0] - position[0]) > self.threshold:
            if target_position[0] > position[0]:
                action['left'] = 1
            else:
                action['right'] = 1

        return action

