from agent import AgentWrapper

import numpy as np



class AlwaysJumpingAgent(AgentWrapper):
    def __call__(self, obs):
        action = AgentWrapper.__call__(self, obs)
        action['jump'] = 1
        return action


class AgentObsWrapper(AgentWrapper):
    def __init__(self, agent, inner_fn, finish_fn=None):
        super().__init__(agent)
        self.inner_fn = inner_fn
        self.finish_fn = finish_fn

    def reset(self, obs):
        super().reset(obs)
        self.inner_fn(obs)

    def observe(self, obs, reward, done, debug_info):
        super().observe(obs, reward, done, debug_info)
        self.inner_fn(obs)

    def finish_episode(self):
        super().finish_episode()
        if self.finish_fn:
            self.finish_fn()


class SafeAgentWrapper(AgentWrapper):
    dangerous_ores = ['lava']
    falling_ores = ['water', 'air']

    def __call__(self, obs):
        action = AgentWrapper.__call__(self, obs)
        next_rel_pos = self.get_next_rel_pos(action, obs)

        # Dangerous ore in front?
        next_ore = obs['grid_arr'][next_rel_pos[0], next_rel_pos[1], next_rel_pos[2]]

        if next_ore in self.dangerous_ores:
            return self.action_space.no_op()

        # Dangerous ore below?
        next_ore_below = obs['grid_arr'][next_rel_pos[0], next_rel_pos[1] - 1, next_rel_pos[2]]

        if next_ore_below in self.dangerous_ores:
            return self.action_space.no_op()

        # Falling into water or air?
        next_ore_below_below = obs['grid_arr'][next_rel_pos[0], next_rel_pos[1] - 2, next_rel_pos[2]]

        if next_ore_below_below in self.falling_ores:
            return self.action_space.no_op()

        # Good to go
        return action

    def get_next_rel_pos(self, action, obs):
        dz = action['back'] - action['forward']
        dx = action['right'] - action['left']

        z, y, x = np.array(obs['grid_arr'].shape, dtype=int) // 2

        return (z + dz, y, x + dx)

