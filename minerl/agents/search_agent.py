from agent import Agent
from algorithms import Planner

import numpy as np


class SearchAgent(Agent):

    rng = np.random.RandomState(0)

    def __call__(self, obs):
        pos_to_ore = self.grid_agent.pos_to_ore

        self.xz_positions = {(x, z) for x, y, z in pos_to_ore}

        action_list = ['forward', 'back', 'left', 'right']
        action_effects = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
        ]

        def model(state, action):
            action_effect = action_effects[action_list.index(action)]
            possible_next_state = np.add(state, action_effect)

            if tuple(possible_next_state) in self.xz_positions:
                return possible_next_state

            return state

        def heuristic(state):
            return 0

        init_state = np.array([obs['position'][0], obs['position'][2]])
        goal = self.goal

        seed = self.rng.randint(100000)
        planner = Planner(model, action_list, self.goal_check, heuristic, seed=seed)

        try:
            out = planner.plan(init_state, goal)
            action_str = out.plan[0]
            action = self.action_space.no_op()
            action[action_str] = 1
            return action
        except:
            print("Warning: planning problem, returning noop.")

        return self.action_space.no_op()

    def goal_check(self, state, goal):
        return np.all(state == goal)

    def set_goal(self, goal):
        self.goal = np.array(goal)

    def subscribe_to_grid_agent(self, grid_agent):
        self.grid_agent = grid_agent


class ExploringSearchAgent(SearchAgent):
    goal = None

    def reset(self, obs):
        self.last_position = tuple(obs['position'])
        self.goals_to_avoid = set()

    def goal_check(self, state, goal):
        if tuple(state) not in self.xz_positions:
            return False

        if tuple(state) in self.goals_to_avoid:
            return False

        for action_effect in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            possible_next_state = np.add(state, action_effect)
            if tuple(possible_next_state) not in self.xz_positions:
                self.last_goal = tuple(state)
                return True

        return False

    def observe(self, obs, reward, done, debug_info):
        if tuple(obs['position']) == self.last_position:
            print("Stuck! Avoiding goal.")
            self.goals_to_avoid.add(self.last_goal)
        else:
            self.last_position = tuple(obs['position'])


