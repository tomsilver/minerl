import numpy as np
import imageio
import matplotlib.pyplot as plt



class GridBuildingAgent(object):
    def __init__(self, action_space, grid_mins=(-1, -1, -1), grid_maxs=(1, 1, 1)):
        self.action_space = action_space

        self.grid_mins = grid_mins
        self.grid_maxs = grid_maxs

    def __call__(self, obs):
        action = self.action_space.sample()

        return action

    def reset(self, obs):
        self.pos_to_ore = {}
        self.min_x, self.min_y, self.min_z = np.inf, np.inf, np.inf
        self.max_x, self.max_y, self.max_z = -np.inf, -np.inf, -np.inf

    def observe(self, obs, reward, done, info):
        pos = obs['position']
        grid = tuple(obs['grid'])

        arr = np.array(grid).reshape((tuple(1 + np.subtract(self.grid_maxs, self.grid_mins))))

        print("grid:", grid)

        for i, zo in enumerate(range(self.grid_mins[2], self.grid_maxs[2] + 1)):
            for j, yo in enumerate(range(self.grid_mins[1], self.grid_maxs[1] + 1)):
                for k, xo in enumerate(range(self.grid_mins[0], self.grid_maxs[0] + 1)):
                    ore = arr[i, j, k]
                    p = (x, y, z) = (pos[0] + xo, pos[1] + yo, pos[2] + zo)
                    if p in self.pos_to_ore:
                        if self.pos_to_ore[p] != ore:
                            print("Inconsistent!")
                            print("Pos:", p)
                            print("Old:", self.pos_to_ore[p])
                            print("New:", ore)
                    else:
                        self.pos_to_ore[p] = ore

                    self.min_x = min(self.min_x, x)
                    self.min_y = min(self.min_y, y)
                    self.min_z = min(self.min_z, z)

                    self.max_x = max(self.max_x, x)
                    self.max_y = max(self.max_y, y)
                    self.max_z = max(self.max_z, z)


    def finish_episode(self):
        full_grid = np.full((self.max_x - self.min_x + 1, 
            self.max_y - self.min_y + 1, self.max_z - self.min_z + 1), 
            'unk', dtype='object')

        for (x, y, z), ore in self.pos_to_ore.items():
            full_grid[x - self.min_x, y - self.min_y, z - self.min_z] = ore

        for vertical_plane in range(full_grid.shape[1]):
            draw_vertical_plane(full_grid[:, vertical_plane], 
                'imgs/plane{}.png'.format(vertical_plane))



colors = {'leaves': (50, 200, 50), 'dirt': (210, 105, 30), 'stone': (100, 100, 100), 'grass': (0, 255, 0), 'unk': (0, 0, 0), 'air': (255, 255, 255), 'tallgrass': (100, 250, 100), 'brown_mushroom': (128, 0, 128), 'double_plant': (123, 200, 75), 'log': (200, 160, 20), 'log2' : (210, 140, 20), 'deadbush': (100, 80, 10), 'water' : (0,0,255), 'sand' : (210,180,140), 'gravel' : (40,40,40), 'clay' : (46,52,60), 'yellow_flower' : (255, 255, 0), 'red_flower' : (255, 0, 0)}
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
