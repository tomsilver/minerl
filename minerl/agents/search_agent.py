from agent import Agent
from algorithms import Planner

import numpy as np



class SearchAgent(Agent):
    # TODO bring back?
    pass



class ExploringSearchAgent(Agent):

    rng = np.random.RandomState(0)
    max_countdown = 2
    all_directions = ['forward', 'back', 'left', 'right']

    def reset(self, obs):
        position = tuple(obs['position'])
        
        self.position_to_action_to_neighbor = {position : {}}
        self.visited_positions = set([position])
        self.last_position = position
        self.last_action = None
        self.internal_state = ('expanding', self.all_directions[0], self.max_countdown)

        return super().reset(obs)

    def __call__(self, obs):

        position = tuple(obs['position'])
        position_changed = (position != self.last_position)
        last_position = self.last_position
        self.last_position = position

        print("internal_state:", self.internal_state, "position:", position, "last position:", self.last_position)

        if self.internal_state[0] == 'expanding':
            direction, countdown = self.internal_state[1:]
            
            if position_changed:
                self.add_node(last_position, position, direction)
                next_direction = self.opposite_direction(direction)
                self.internal_state = ('retracting', next_direction, self.max_countdown)
                return self.finish_action(next_direction)

            elif countdown > 0:
                self.internal_state = ('expanding', direction, countdown - 1)
                return self.finish_action(direction)
            
            elif direction == self.all_directions[-1]:
                self.internal_state = ('moving_to_goal',)
                action = self.plan_action_to_goal(position)
                return self.finish_action(action)

            else:
                direction_idx = self.all_directions.index(direction)
                next_direction = self.all_directions[direction_idx + 1]
                self.internal_state = ('expanding', next_direction, self.max_countdown)
                return self.finish_action(next_direction)

        elif self.internal_state[0] == 'retracting':
            direction, countdown = self.internal_state[1:]

            if not position_changed:
                if countdown == 0:
                    # import pdb; pdb.set_trace()
                    # raise Exception("Failed to retract.")
                    print("Warning: failed to retract. Expanding from this new position.")
                    direction = self.all_directions[0]
                    self.internal_state = ('expanding', direction, self.max_countdown)
                    self.visited_positions.add(position)
                    return self.finish_action(direction)

                self.internal_state = ('retracting', direction, countdown - 1)
                return self.finish_action(direction)

            if position not in self.visited_positions:
                import pdb; pdb.set_trace()
                raise Exception("Retracted incorrectly")

            expanding_direction = self.opposite_direction(direction)

            if expanding_direction == self.all_directions[-1]:
                self.internal_state = ('moving_to_goal',)
                action = self.plan_action_to_goal(position)
                return self.finish_action(action)

            else:
                direction_idx = self.all_directions.index(expanding_direction)
                next_direction = self.all_directions[direction_idx + 1]
                self.internal_state = ('expanding', next_direction, self.max_countdown)
                return self.finish_action(next_direction)

        else:
            assert self.internal_state[0] == 'moving_to_goal'

            if self.last_action is not None and position_changed:
                self.add_node(last_position, position, self.last_action)

            if self.is_goal_position(position):
                direction = self.all_directions[0]
                self.internal_state = ('expanding', direction, self.max_countdown)
                self.visited_positions.add(position)
                return self.finish_action(direction)

            action = self.plan_action_to_goal(position)
            return self.finish_action(action)

    def add_node(self, last_position, position, direction):
        self.position_to_action_to_neighbor[last_position][direction] = position

        if position not in self.position_to_action_to_neighbor:
            self.position_to_action_to_neighbor[position] = {}

        opposite_direction = self.opposite_direction(direction)

        self.position_to_action_to_neighbor[position][opposite_direction] = last_position

    def opposite_direction(self, direction):
        return {
            'forward' : 'back',
            'back' : 'forward',
            'left' : 'right',
            'right' : 'left'
        }[direction]

    def finish_action(self, action_str):
        self.last_action = action_str
        action = self.action_space.no_op()
        action[action_str] = 1
        return action

    def is_goal_position(self, position):
        return position not in self.visited_positions

    def plan_action_to_goal(self, position):
        init_state = position

        def model(state, action):
            if state in self.position_to_action_to_neighbor:
                if action in self.position_to_action_to_neighbor[state]:
                    return self.position_to_action_to_neighbor[state][action]
            return state

        action_list = self.all_directions
        goal_check = lambda s, g : self.is_goal_position(s)
        heuristic = lambda s : 0.
        seed = self.rng.randint(1000000)

        planner = Planner(model, action_list, goal_check, heuristic, seed=seed)

        # try:
        out = planner.plan(init_state)
        action_str = out.plan[0]
        return action_str
        # except:
        #     print("Warning: planning problem, returning noop.")

        # return self.action_space.no_op()








