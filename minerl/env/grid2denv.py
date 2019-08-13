from minerl.env import spaces

import gym
import numpy as np
import scipy.ndimage



class Grid2DEnv(gym.Env):

    EMPTY, WALL, AGENT = 0, 1, 2

    actions = ['forward', 'back', 'left', 'right']
    action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    forbidden = [WALL]

    def __init__(self, layout, init_pos):
        self.layout = layout
        self.init_pos = init_pos
        self.pos = init_pos
        self.height, self.width = np.array(layout).shape

        self.action_space = spaces.Dict(spaces={
            "forward": spaces.Discrete(2), 
            "back": spaces.Discrete(2), 
            "left": spaces.Discrete(2), 
            "right": spaces.Discrete(2)
        })

    def reset(self):
        self.pos = self.init_pos
        return self.get_observation()

    def step(self, action):
        r, c = self.pos
        dr, dc = 0, 0

        for k,v in action.items():
            if v:
                idx = self.actions.index(k)
                adr, adc = self.action_deltas[idx]
                dr += adr
                dc += adc

        if not (0 <= r + dr < self.height):
            return self.get_observation(), 0., False, {}

        if not (0 <= c + dc < self.width):
            return self.get_observation(), 0., False, {}

        if self.layout[r + dr, c + dc] in self.forbidden:
            return self.get_observation(), 0., False, {}

        self.pos = (r + dr, c + dc)

        return self.get_observation(), 0., False, {}

    def get_observation(self):
        return {'position' : self.pos}

    def render(self, mode='human'):
        obs = np.array(self.layout, dtype=np.int64)
        obs[self.pos[0], self.pos[1]] = self.AGENT
        obs = scipy.ndimage.zoom(obs, 20, order=0)

        return obs


