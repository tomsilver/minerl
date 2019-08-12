from agent import Agent
from algorithms import Planner

import numpy as np


class SearchAgent(Agent):

    def __call__(self, obs):
        pos_to_ore = self.grid_agent.pos_to_ore
        action_list = ['forward', 'back', 'left', 'right']
        action_effects = [(0, 0, 1),  (0, 0, -1), (1, 0, 0), (-1, 0, 0)]

        def model(state, action):
            possible_next_state = np.add(state, action_effects[action_list.index(action)])

            if tuple(possible_next_state) in pos_to_ore and pos_to_ore[tuple(possible_next_state)] == 'air':
                return possible_next_state

            return state

        def goal_check(state, goal):
            return np.all(state == goal)

        def heuristic(state):
            return 0

        init_state = np.array(obs['position'])
        goal = self.goal

        # import pdb; pdb.set_trace()

        planner = Planner(model, action_list, goal_check, heuristic)
        out = planner.plan(init_state, goal)

        # import pdb; pdb.set_trace()

        action_str = out.plan[0]
        action = self.action_space.no_op()
        action[action_str] = 1
        return action

    def set_goal(self, goal):
        self.goal = np.array(goal)

    def subscribe_to_grid_agent(self, grid_agent):
        self.grid_agent = grid_agent
