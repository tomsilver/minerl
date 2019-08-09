from agent import Agent

import numpy as np
import imageio
import matplotlib.pyplot as plt



class SequentialAgent(Agent):
    def __init__(self, action_space, action_sequence, final_action=None):
        Agent.__init__(self, action_space)

        self.action_sequence = action_sequence
        self.final_action = final_action

    def __call__(self, obs):
        if self.action_idx >= len(self.action_sequence):
            if self.final_action is not None:
                return self.final_action
            raise Exception("Action sequence expired")

        action = self.action_sequence[self.action_idx]
        self.action_idx += 1

        return action

    def reset(self, obs):
        self.action_idx = 0
        return Agent.reset(self, obs)

