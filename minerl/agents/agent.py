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



class AgentWrapper(Agent):
    def __init__(self, agent):
        self.agent = agent

    def __call__(self, obs):
        return self.agent.__call__(obs)

    def reset(self, obs):
        return self.agent.reset(obs)

    def observe(self, obs, reward, done, debug_info):
        return self.agent.observe(obs, reward, done, debug_info)

    def finish_episode(self):
        return self.agent.finish_episode()



