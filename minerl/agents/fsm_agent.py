from agent import Agent

class FSMAgent(Agent):
    def __init__(self, states, repeat=False):
        self.states = states
        self.repeat = repeat

    def __call__(self, obs):
        if self.repeat and self.current_state_idx >= len(self.states):
            self.reset(obs)

        print("fsm state:", self.current_state_idx)

        agent, done_fn = self.states[self.current_state_idx]
        done = done_fn(obs)

        if done:
            self.current_state_idx += 1
            return self.__call__(obs)

        action = agent(obs)

        return action

    def reset(self, obs):
        self.current_state_idx = 0
        for agent, done_fn in self.states:
            agent.reset(obs)

            if hasattr(done_fn, 'reset'):
                done_fn.reset()


class DoneTimer(object):
    def __init__(self, time, finish_fn=None):
        self.time = time
        self.init_time = time
        self.finish_fn = finish_fn

    def __call__(self, obs):
        self.time -= 1
        if self.finish_fn is not None and self.finish_fn(obs):
            return True
        return self.time <= 0

    def reset(self):
        self.time = self.init_time
