from agent import Agent

import numpy as np

class MoveToItemInNeighborhood(Agent):

    rng = np.random.RandomState(0)

    def __init__(self, action_space, item_type, delta=(0, 0)):
        super().__init__(action_space)
        self.item_type = item_type
        self.delta = delta

    def __call__(self, obs):
        grid_arr = obs['grid_arr']

        item_positions = np.argwhere(
            (grid_arr[:, 2] == self.item_type) & \
            (grid_arr[:, 3] == self.item_type) & \
            (grid_arr[:, 4] == self.item_type)
        )
        item_distances = item_positions - np.array([grid_arr.shape[0], grid_arr.shape[2]]) // 2 + np.array(self.delta)

        closest_pos_idx = np.argmin(np.sum(np.abs(item_distances), axis=1))
        target_movement = item_distances[closest_pos_idx]

        print("item_distances:", item_distances)

        # import pdb; pdb.set_trace()
        
        action = self.action_space.no_op()

        if target_movement[0] < 0:
            # action['forward'] = 1
            action['move'] = 1
        elif target_movement[1] == 0 and target_movement[0] > 0:
            if self.rng.random() > 0.5:
                # action['left'] = 1
                action['strafe'] = 1
            else:
                # action['right'] = 1
                action['strafe'] = -1
        elif target_movement[0] > 0:
            # action['back'] = 1
            action['move'] = -1
        elif target_movement[1] < 0:
            # action['left'] = 1
            action['strafe'] = 1
        elif target_movement[1] > 0:
            # action['right'] = 1
            action['strafe'] = -1

        return action
