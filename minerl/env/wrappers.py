import tempfile
import gym
import numpy as np
import imageio



class GridWorldWrapper(gym.Wrapper):
    """A wrapper around MineRLEnv so that it can be treated as a discrete gridworld.
    """
    def __init__(self, env, max_inner_steps=25, threshold=0.01, grid_mins=(-2, -1, -2), grid_maxs=(2, -1, 2)):
        super().__init__(env)

        self.grid_mins = grid_mins
        self.grid_maxs = grid_maxs
        env.unwrapped.xml_file = fill_in_env_xml(env.xml_file, {
            'GRID_MIN_X' : grid_mins[2],
            'GRID_MIN_Y' : grid_mins[1],
            'GRID_MIN_Z' : grid_mins[0],
            'GRID_MAX_X' : grid_maxs[2],
            'GRID_MAX_Y' : grid_maxs[1],
            'GRID_MAX_Z' : grid_maxs[0],
        })

        self.max_inner_steps = max_inner_steps
        self.threshold = threshold

    def step_in_env(self, action):
        return self.env.step(action)

    def step_to_target(self, target_position):
        for _ in range(self.max_inner_steps):
            inner_action = self.get_action_towards_target(self.position, target_position)
            obs, reward, done, info = self.step_in_env(inner_action)
            self.position = np.array([obs['XPos'], obs['YPos'], obs['ZPos']])

            if done:
                break

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
        obs['grid_arr'] = self.grid_to_array(obs['grid'])

        return obs

    def step(self, action):
        obs, reward, done, info = self.step_in_env(action)
        self.position = np.array([obs['XPos'], obs['YPos'], obs['ZPos']])

        target_position = self.get_target_position(self.position, action)

        obs, reward, done, debug_info = self.step_to_target(target_position)

        # Always let settle b/c jumping observations are unreliable
        for _ in range(5):
            obs, reward, done, debug_info = self.step_in_env(self.action_space.noop())

        obs['position'] = self.discretize_position(self.position)
        obs['grid_arr'] = self.grid_to_array(obs['grid'])

        return obs, reward, done, debug_info

    def discretize_position(self, position):
        new_p = []
        print("position:", position)
        for p in position:
            if p.is_integer():
                new_p.append(p - 1)
            else:
                new_p.append(np.floor(p))
        discretized_position = np.array(new_p, dtype=np.int64)
        print("discretized_position:", discretized_position)
        return discretized_position

    def grid_to_array(self, grid):
        shape = 1 + np.subtract(self.grid_maxs, self.grid_mins)
        arr = np.array(grid).reshape(shape)
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


def fill_in_env_xml(xml_file, fill_ins):
    with open(xml_file, 'r') as f:
        xml = f.read()
    
    for placeholder, fill_in in fill_ins.items(): 
        xml = xml.replace('$({})'.format(placeholder), str(fill_in))

    new_f = tempfile.NamedTemporaryFile(mode='w', delete=False)
    new_f.write(xml)

    return new_f.name


class VideoWrapper(gym.Wrapper):
    def __init__(self, env, out_path, fps=30, obs_key='pov', viewpoint=0):
        super().__init__(env)
        self.out_path = out_path
        self.fps = fps
        self.obs_key = obs_key
        self.viewpoint = viewpoint

        env.unwrapped.xml_file = fill_in_env_xml(env.xml_file, {
            'VIEWPOINT' : viewpoint
        })

    def reset(self):
        self.images = []
        obs = super().reset()
        self.images.append(obs[self.obs_key])
        return obs

    def step(self, action):
        obs, reward, done, debug_info = super().step(action)
        self.images.append(obs[self.obs_key])

        if done:
            imageio.mimsave(self.out_path, self.images, fps=self.fps)
            print("Wrote out video to {}.".format(self.out_path))

        return obs, reward, done, debug_info


