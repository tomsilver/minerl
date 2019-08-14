from agent import Agent

import numpy as np

class RoombaAgent(Agent):
    rng = np.random.RandomState(0)
    prob_change = 0.1
    stuck_threshold = 3

    def reset_direction(self):
        self.direction = ['forward', 'back', 'left', 'right'][self.rng.choice(4)]
        print("reset direction to", self.direction)
        self.stuck_count_down = self.stuck_threshold

    def reset(self, obs):
        self.reset_direction()
        self.last_position = obs['position']
        self.stuck_count_down = self.stuck_threshold
        return super().reset(obs)

    def __call__(self, obs):

        if self.rng.random() < self.prob_change:
            self.reset_direction()

        elif np.all(obs['position'] == self.last_position):
            if self.stuck_count_down <= 0:
                self.reset_direction()
            else:
                self.stuck_count_down -= 1
        else:
            self.stuck_count_down = self.stuck_threshold

        self.last_position = obs['position']

        action = self.action_space.no_op()
        action[self.direction] = 1
        return action

