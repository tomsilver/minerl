from agent import Agent
from algorithms import Planner

import numpy as np



class SearchAgent(Agent):
    # TODO bring back?
    pass



class ExploringSearchAgent(Agent):

    TODO TEST THIS IN GRID WORLD NOT MINECRAFT

    rng = np.random.RandomState(0)
    max_countdown = 3
    all_directions = ['forward', 'back', 'left', 'right']

    def reset(self, obs):
        position = tuple(obs['position'])
        
        self.position_to_action_to_neighbor = {position : {}}
        self.visited_positions = set([position])
        self.last_position = position
        self.last_action = None
        self.internal_state = ('expanding', self.all_directions[0], self.max_countdown)

        return super().reset(obs)

    def observe(self, obs, reward, done, debug_info):

        position = tuple(obs['position'])
        self.visited_positions.add(position)
        self.last_position = position

        return super().observe(obs, reward, done, debug_info)

    def __call__(self, obs):
        position = tuple(obs['position'])

        if self.internal_state[0] == 'expanding':
            direction, countdown = self.internal_state[1:]
            
            if position != self.last_position:
                self.add_node(position, direction)
                next_direction = self.opposite_direction(direction)
                self.internal_state = ('retracting', next_direction, self.max_countdown)
                return self.finish_action(next_direction)

            elif countdown > 0:
                self.internal_state = ('expanding', direction, countdown - 1)
                return self.finish_action(direction)
            
            elif direction == self.all_directions[-1]:
                self.internal_state = ('moving_to_goal',)
                action = self.plan_action_to_goal()
                return self.finish_action(action)

            else:
                direction_idx = self.all_directions.index(direction)
                next_direction = self.all_directions[direction_idx + 1]
                self.internal_state = ('expanding', next_direction, self.max_countdown)
                return self.finish_action(next_direction)

        elif self.internal_state[0] == 'retracting':
            direction, countdown = self.internal_state[1:]

            if position == self.last_position:
                if countdown == 0:
                    import pdb; pdb.set_trace()
                    raise Exception("Failed to retract.")

                self.internal_state = ('retracting', direction, countdown - 1)
                return self.finish_action(direction)

            if position not in self.visited_positions:
                import pdb; pdb.set_trace()
                raise Exception("Retracted incorrectly")

            expanding_direction = self.opposite_direction(direction)

            if expanding_direction == self.all_directions[-1]:
                self.internal_state = ('moving_to_goal',)
                action = self.plan_action_to_goal()
                return self.finish_action(action)

            else:
                direction_idx = self.all_directions.index(expanding_direction)
                next_direction = self.all_directions[direction_idx + 1]
                self.internal_state = ('expanding', next_direction, self.max_countdown)
                return self.finish_action(next_direction)

        else:
            assert self.internal_state[0] == 'moving_to_goal'

            if self.is_goal_position(position):
                direction = self.all_directions[0]
                self.internal_state = ('expanding', direction, self.max_countdown)
                return self.finish_action(direction)

            action = self.plan_action_to_goal()
            return self.finish_action(action)

    def add_node(self, position, direction):
        TODO

    def opposite_direction(self, direction):
        TODO

    def finish_action(self, action):
        TODO

    def is_goal_position(self, position):
        TODO

    def plan_action_to_goal(self):
        TODO








