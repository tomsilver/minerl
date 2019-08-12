import numpy as np
import heapq as hq


class Planner(object):
    """A* planner.
    """
    def __init__(self, model, action_list, goal_check, heuristic, seed=0):
        self.model = model
        self.action_list = action_list
        self.goal_check = goal_check
        self.heuristic = heuristic
        self.rng = np.random.RandomState(seed)

    def plan(self, init_state, goal):
        """A* search. Returns a PlannerOutput object.
        """
        pqueue = []
        visited = set()
        root = SearchTreeEntry(init_state,
                               state_sequence=[],
                               action_sequence=[],
                               cost=0,
                               cost_sequence=[0])
        hq.heappush(pqueue, (self.heuristic(init_state), 0., root))
        while len(pqueue) > 0:
            _, _, entry = hq.heappop(pqueue)
            # print("popped", entry.state, entry.action_sequence)
            if self.goal_check(entry.state, goal):
                entry.state_sequence.append(entry.state)
                cost_to_go = (entry.cost_sequence[-1]-np.array(entry.cost_sequence))
                cost_to_go_pred = [self.heuristic(s) for s in entry.state_sequence]
                if any(cost_to_go_pred > cost_to_go):
                    raise Exception("Heuristic is not admissible")
                return PlannerOutput(entry.action_sequence, entry.cost, entry.state_sequence)
            if not self._in_set(entry.state, visited):
                self._add_to_set(entry.state, visited)
                for action in self.action_list:
                    cost = self._action_cost(action)
                    next_state = self.model(entry.state, action)
                    # print("adding to heap:", next_state, action)
                    new_entry = SearchTreeEntry(
                        next_state,
                        entry.state_sequence+[entry.state],
                        entry.action_sequence+[action],
                        entry.cost+cost,
                        entry.cost_sequence+[entry.cost+cost])
                    
                    hq.heappush(pqueue, (entry.cost+cost+self.heuristic(next_state), self.rng.random(), new_entry))

        # import pdb; pdb.set_trace()
        raise Exception("No path found")

    @staticmethod
    def _add_to_set(state, state_set):
        state_tuple = tuple(state.copy()) #tuple(map(tuple, state.copy()))
        state_set.add(state_tuple)

    @staticmethod
    def _in_set(state, state_set):
        state_tuple = tuple(state.copy()) #tuple(map(tuple, state.copy()))
        return state_tuple in state_set

    @staticmethod
    def _action_cost(_action):
        return 1


class SearchTreeEntry:
    """Entry in search tree for A*.
    """
    def __init__(self, state, state_sequence, action_sequence,
                 cost, cost_sequence):
        self.state = state
        self.state_sequence = state_sequence
        self.action_sequence = action_sequence
        self.cost = cost
        self.cost_sequence = cost_sequence


class PlannerOutput:
    """Output of planner. If the plan has length n, then state_traj has
    length n+1. state_traj[i] is the state before running the action
    given by plan[i].
    """
    def __init__(self, plan, cost, state_traj):
        self.plan = plan
        self.cost = cost
        self.state_traj = state_traj

