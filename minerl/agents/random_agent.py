from agent import Agent

class RandomAgent(Agent):
    def __call__(self, obs):
        return self.action_space.sample()
