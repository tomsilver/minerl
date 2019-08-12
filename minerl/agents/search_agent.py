from agent import Agent
from algorithms import Planner

import numpy as np


class SearchAgent(Agent):

    def __call__(self, obs):
        pos_to_ore = self.grid_agent.pos_to_ore
        action_list = ['forward', 'back', 'left', 'right']
        action_effects = [
            [(0, -1, 1), (0, 0, 1), (0, 1, 1)],  
            [(0, -1, -1), (0, 0, -1), (0, 1, -1)], 
            [(1, -1, 0), (1, 0, 0), (1, 1, 0)], 
            [(-1, -1, 0), (-1, 0, 0), (-1, 1, 0)],
        ]

        def model(state, action):
            for action_effect in action_effects[action_list.index(action)]:
                possible_next_state = np.add(state, action_effect)

                if tuple(possible_next_state) in pos_to_ore and pos_to_ore[tuple(possible_next_state)] == 'air':
                    return possible_next_state

            return state

        def heuristic(state):
            return 0

        init_state = np.array(obs['position'])
        goal = self.goal

        # import pdb; pdb.set_trace()

        planner = Planner(model, action_list, self.goal_check, heuristic)
        out = planner.plan(init_state, goal)

        # import pdb; pdb.set_trace()

        action_str = out.plan[0]
        action = self.action_space.no_op()
        action[action_str] = 1
        return action

    def goal_check(self, state, goal):
        return np.all(state == goal)

    def set_goal(self, goal):
        self.goal = np.array(goal)

    def subscribe_to_grid_agent(self, grid_agent):
        self.grid_agent = grid_agent


class ExploringSearchAgent(SearchAgent):
    def reset(self, obs):
        self.visited_positions = { tuple(obs['position']) }
        self.goal = None
        return super().reset(obs)

    def goal_check(self, state, goal):
        if tuple(state) not in self.visited_positions:
            print("Found goal", goal)
            return True
        return False

    def __call__(self, obs):
        self.visited_positions.add(tuple(obs['position']))
        return super().__call__(obs)

