class Agent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, obs):
        raise NotImplementedError()

    def reset(self, obs):
        pass

    def observe(self, obs, reward, done, debug_info):
        pass

    def finish_episode(self):
        pass
