from agent import AgentWrapper


class AlwaysJumpingAgent(AgentWrapper):
    def __call__(self, obs):
        action = AgentWrapper.__call__(self, obs)
        action['jump'] = 1
        return action

class SafeAgentWrapper(AgentWrapper):
    pass
    # dangerous_ores = ['lava']
    # falling_ores = ['water', 'air']

    # def __call__(self, obs):
    #     action = AgentWrapper.__call__(self, obs)
    #     next_rel_pos = self.get_next_rel_pos(action, obs)

    #     # Dangerous ore in front?
    #     next_ore = obs['grid'][next_rel_pos[0], next_rel_pos[1], next_rel_pos[2]]

    #     if next_ore in self.dangerous_ores:
    #         return self.action_space.noop()

    #     # Dangerous ore below?
    #     next_ore_below = obs['grid'][next_rel_pos[0], next_rel_pos[1] - 1, next_rel_pos[2]]

    #     if next_ore_below in self.dangerous_ores:
    #         return self.action_space.noop()

    #     # Falling into water or air?
    #     next_ore_below_below = obs['grid'][next_rel_pos[0], next_rel_pos[1] - 2, next_rel_pos[2]]

    #     if next_ore_below_below in self.falling_ores:
    #         return self.action_space.noop()

    #     # Good to go
    #     return action

    # def get_next_rel_pos(self, action, obs):
    #     dz = action['back'] - action['forward']
    #     dx = action['right'] - action['left']

    #     TODO...
