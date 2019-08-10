from agent import AgentWrapper

import numpy as np
import imageio
import matplotlib.pyplot as plt



class GridBuildingAgentWrapper(AgentWrapper):
    def __init__(self, agent, grid_mins=(-2, -1, -2), grid_maxs=(2, -1, 2)):
        AgentWrapper.__init__(self, agent)

        self.grid_mins = grid_mins
        self.grid_maxs = grid_maxs

    def reset(self, obs):
        self.pos_to_ore = {}
        self.min_x, self.min_y, self.min_z = np.inf, np.inf, np.inf
        self.max_x, self.max_y, self.max_z = -np.inf, -np.inf, -np.inf

        self.process_obs(obs)

        return AgentWrapper.reset(self, obs)

    def process_obs(self, obs):
        pos = obs['position']
        arr = obs['grid_arr']

        window_x_min, window_y_min, window_z_min = self.grid_mins
        window_x_max, window_y_max, window_z_max = self.grid_maxs

        indices_i = list(range(self.grid_maxs[0], self.grid_mins[0] - 1, -1))
        indices_j = list(range(self.grid_mins[1], self.grid_maxs[1] + 1))
        indices_k = list(range(self.grid_maxs[2], self.grid_mins[2] - 1, -1))

        for i, zo in enumerate(indices_i):
            for j, yo in enumerate(indices_j):
                for k, xo in enumerate(indices_k):
                    ore = arr[i, j, k]
                    p = (x, y, z) = (pos[0] + xo, pos[1] + yo, pos[2] + zo)
                    if p in self.pos_to_ore:
                        if self.pos_to_ore[p] != ore:
                            print("Inconsistent!")
                            print("Pos:", p)
                            print("Old:", self.pos_to_ore[p])
                            print("New:", ore)
                            import pdb; pdb.set_trace()
                    else:
                        self.pos_to_ore[p] = ore

                    self.min_x = min(self.min_x, x)
                    self.min_y = min(self.min_y, y)
                    self.min_z = min(self.min_z, z)

                    self.max_x = max(self.max_x, x)
                    self.max_y = max(self.max_y, y)
                    self.max_z = max(self.max_z, z)

    def observe(self, obs, reward, done, info):
        self.process_obs(obs)
        return AgentWrapper.observe(self, obs, reward, done, info)

    def finish_episode(self):
        full_grid = np.full((self.max_z - self.min_z + 1, 
            self.max_y - self.min_y + 1, self.max_x - self.min_x + 1), 
            'unk', dtype='object')

        for (x, y, z), ore in self.pos_to_ore.items():
            full_grid[z - self.min_z, y - self.min_y, x - self.min_x] = ore

        full_grid = np.rot90(full_grid, k=2, axes=(0, 2))

        for vertical_plane in range(full_grid.shape[1]):
            draw_vertical_plane(full_grid[:, vertical_plane], 
                'imgs/plane{}.png'.format(vertical_plane))

        return AgentWrapper.finish_episode(self)



colors = {'leaves': (50, 200, 50), 'dirt': (210, 105, 30), 'stone': (100, 100, 100), 'grass': (0, 255, 0), 'unk': (0, 0, 0), 'air': (255, 255, 255), 'tallgrass': (100, 250, 100), 'brown_mushroom': (128, 0, 128), 'double_plant': (123, 200, 75), 'log': (200, 160, 20), 'log2' : (210, 140, 20), 'deadbush': (100, 80, 10), 'water' : (0,0,255), 'sand' : (210,180,140), 'gravel' : (40,40,40), 'clay' : (46,52,60), 'yellow_flower' : (255, 255, 0), 'red_flower' : (255, 0, 0), 'diamond_block' : (185, 242, 255), 'glass' : (255, 185, 242), 'lava' : (207, 16, 32)}
def draw_vertical_plane(grid, outfile):
    plane_img = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            ore = grid[r, c]

            if ore not in colors:
                color_str = input("Color for {}:".format(ore))
                rgb = tuple(eval(color_str))
                colors[ore] = rgb
            plane_img[r, c] = colors[ore]

    imageio.imsave(outfile, plane_img)
    print("Wrote out to {}.".format(outfile))
