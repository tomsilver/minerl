from agent import Agent

class FSMAgent(Agent):
    def __init__(self, states):
        self.states = states

    def __call__(self, obs):
        agent, done_fn = self.states[self.current_state_idx]
        action = agent(obs)
        done = done_fn(obs)

        if done:
            self.current_state_idx += 1

        return action

    def reset(self, obs):
        self.current_state_idx = 0
        for agent, _ in self.states:
            agent.reset(obs)


class DoneTimer(object):
    def __init__(self, time):
        self.time = time

    def __call__(self, obs):
        self.time -= 1
        return self.time <= 0
