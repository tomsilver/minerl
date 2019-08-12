from agent import Agent
from algorithms import Planner
from simple_agent_wrappers import SafeAgentWrapper

import numpy as np

def ore_is_solid(ore):
    if 'grass' in ore:
        return True

    if 'ore' in ore:
        return True

    if 'stone' in ore:
        return True

    if 'dirt' in ore:
        return True

    return False


class SearchAgent(Agent):

    rng = np.random.RandomState(0)

    def __call__(self, obs):
        pos_to_ore = self.grid_agent.pos_to_ore
        action_list = ['forward', 'back', 'left', 'right']
        # action_effects = [
        #     [(0, -1, 1), (0, 0, 1), (0, 1, 1)],  
        #     [(0, -1, -1), (0, 0, -1), (0, 1, -1)], 
        #     [(1, -1, 0), (1, 0, 0), (1, 1, 0)], 
        #     [(-1, -1, 0), (-1, 0, 0), (-1, 1, 0)],
        # ]
        action_effects = [
            (0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0),
        ]

        def model(state, action):
            action_effect = action_effects[action_list.index(action)]
            possible_next_state = np.add(state, action_effect)

            if tuple(possible_next_state) in pos_to_ore:
                ore = pos_to_ore[tuple(possible_next_state)]
            else:
                return state

            if tuple(possible_next_state + [0, 2, 0]) in pos_to_ore:
                above_above_ore = pos_to_ore[tuple(possible_next_state + [0, 2, 0])]
            else:
                above_above_ore = None

            if tuple(possible_next_state + [0, 1, 0]) in pos_to_ore:
                above_ore = pos_to_ore[tuple(possible_next_state + [0, 1, 0])]
            else:
                above_ore = None

            if tuple(possible_next_state + [0, -1, 0]) in pos_to_ore:
                below_ore = pos_to_ore[tuple(possible_next_state + [0, -1, 0])]
            else:
                below_ore = None

            if tuple(possible_next_state + [0, -2, 0]) in pos_to_ore:
                below_below_ore = pos_to_ore[tuple(possible_next_state + [0, -2, 0])]
            else:
                below_below_ore = None

            if SafeAgentWrapper.is_safe(above_above_ore, above_ore, ore, below_ore, below_below_ore):
                
                if not ore_is_solid(below_ore) and not ore_is_solid(ore):
                    # print("predicting a step down")
                    return np.add(possible_next_state, [0, -1, 0])

                if not ore_is_solid(ore):
                    # print("predicting flat movement")
                    return possible_next_state

                # print("predicting a move up, ore is", ore)
                return np.add(possible_next_state, [0, 1, 0])

            return state

        def heuristic(state):
            return 0

        init_state = np.array(obs['position'])
        goal = self.goal

        # import pdb; pdb.set_trace()

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

    def goal_check(self, state, goal):
        if tuple(state) not in self.grid_agent.pos_to_ore:
            return False

        for action_effect in [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0)]:
            possible_next_state = np.add(state, action_effect)
            if tuple(possible_next_state) not in self.grid_agent.pos_to_ore:
                return True

        return False


