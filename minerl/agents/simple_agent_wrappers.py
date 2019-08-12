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

    def __call__(self, obs):
        action = AgentWrapper.__call__(self, obs)
        next_rel_pos = self.get_next_rel_pos(action, obs)

        above_above_ore = obs['grid_arr'][next_rel_pos[0], next_rel_pos[1] + 2, next_rel_pos[2]]
        above_ore = obs['grid_arr'][next_rel_pos[0], next_rel_pos[1] + 1, next_rel_pos[2]]

        # Dangerous ore in front?
        next_ore = obs['grid_arr'][next_rel_pos[0], next_rel_pos[1], next_rel_pos[2]]

        # Dangerous ore below?
        next_ore_below = obs['grid_arr'][next_rel_pos[0], next_rel_pos[1] - 1, next_rel_pos[2]]

        # Falling into water or air?
        next_ore_below_below = obs['grid_arr'][next_rel_pos[0], next_rel_pos[1] - 2, next_rel_pos[2]]

        if SafeAgentWrapper.is_safe(above_above_ore, above_ore, next_ore, next_ore_below, next_ore_below_below):
            return action
        
        return self.action_space.no_op()

    @staticmethod
    def is_safe(above_above_ore, above_ore, ore, below_ore, below_below_ore):
        if above_ore != 'air':
            return False

        if ore in ['lava', 'water']:
            return False

        if below_ore in ['lava', 'water']:
            return False

        if below_ore in ['water', 'air'] and below_below_ore in ['water', 'air']:
            return False

        return True


    def get_next_rel_pos(self, action, obs):
        dz = action['back'] - action['forward']
        dx = action['right'] - action['left']

        z, y, x = np.array(obs['grid_arr'].shape, dtype=int) // 2

        return (z + dz, y, x + dx)

