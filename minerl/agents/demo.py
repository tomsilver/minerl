from grid_building_agent import GridBuildingAgentWrapper, build_full_grid, grid_colors
from simple_agent_wrappers import AlwaysJumpingAgent, SafeAgentWrapper, AgentObsWrapper
from random_agent import RandomAgent
from sequential_agent import SequentialAgent
from fsm_agent import DoneTimer, FSMAgent
from roomba_agent import RoombaAgent
from search_agent import SearchAgent, ExploringSearchAgent
from minerl.env.wrappers import GridWorldWrapper, VideoWrapper
from utils import run_single_episode, fill_in_xml
from algorithms import Planner
from minerl.env.grid2denv import Grid2DEnv

from gym.wrappers import TimeLimit

import matplotlib.pyplot as plt
import gym
import minerl
import numpy as np
import random
import os
import itertools
import imageio
import logging
import coloredlogs
import pickle
coloredlogs.install(logging.INFO)

from mpl_toolkits.mplot3d import Axes3D


def flatworld_demo():
    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    max_episode_steps = 20
    grid_mins = (-2, -1, -2)
    grid_maxs = (2, -1, 2)

    env = gym.make('MineRLFlatGrid-v0')
    env = VideoWrapper(env, 'imgs/flatworld_inner.gif', fps=30)
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/flatworld_outer.gif', fps=30)

    env.unwrapped.xml_file = fill_in_xml(env.xml_file, {
        'GRID_MIN_X' : grid_mins[2],
        'GRID_MIN_Y' : grid_mins[1],
        'GRID_MIN_Z' : grid_mins[0],
        'GRID_MAX_X' : grid_maxs[2],
        'GRID_MAX_Y' : grid_maxs[1],
        'GRID_MAX_Z' : grid_maxs[0],
    })

    env.seed(seed)

    action_strs = ['forward'] * 20
    action_sequence = []
    for action_str in action_strs:
        action = env.action_space.no_op()
        action[action_str] = 1
        action_sequence.append(action)
    final_action = env.action_space.no_op()

    agent = SequentialAgent(env.action_space, action_sequence, final_action=final_action)
    agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    run_single_episode(env, agent)

def grid_unit_test():
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    grid_mins = (-2, -1, -2)
    grid_maxs = (2, -1, 2)
    viewpoint = 1
    max_episode_steps = 20

    env = gym.make('MineRLGridUnitTest-v0')
    env = VideoWrapper(env, 'imgs/inner_grid_unit_test.mp4', fps=30)
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/grid_unit_test.gif', fps=3)

    env.unwrapped.xml_file = fill_in_xml(env.xml_file, {
        'GRID_MIN_X' : grid_mins[2],
        'GRID_MIN_Y' : grid_mins[1],
        'GRID_MIN_Z' : grid_mins[0],
        'GRID_MAX_X' : grid_maxs[2],
        'GRID_MAX_Y' : grid_maxs[1],
        'GRID_MAX_Z' : grid_maxs[0],
        'VIEWPOINT' : viewpoint,
    })

    env.seed(seed)

    action_strs = ['back', 'left', 'forward', 'forward', 'right', 'right', 'back', 'back', 'left']
    action_sequence = []
    for action_str in action_strs:
        action = env.action_space.no_op()
        action[action_str] = 1
        action_sequence.append(action)
    final_action = env.action_space.no_op()

    agent = SequentialAgent(env.action_space, action_sequence, final_action=final_action)
    agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    run_single_episode(env, agent)

def stairs_unit_test():
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    grid_mins = (-2, -1, -2)
    grid_maxs = (2, -1, 2)
    viewpoint = 1
    max_episode_steps = 20

    env = gym.make('MineRLStairsUnitTest-v0')
    env = VideoWrapper(env, 'imgs/inner_stairs_unit_test.gif', fps=30)
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/stairs_unit_test.gif', fps=30)

    env.unwrapped.xml_file = fill_in_xml(env.xml_file, {
        'GRID_MIN_X' : grid_mins[2],
        'GRID_MIN_Y' : grid_mins[1],
        'GRID_MIN_Z' : grid_mins[0],
        'GRID_MAX_X' : grid_maxs[2],
        'GRID_MAX_Y' : grid_maxs[1],
        'GRID_MAX_Z' : grid_maxs[0],
        'VIEWPOINT' : viewpoint,
    })

    env.seed(seed)

    action_strs = ['forward'] * 10 + ['back'] * 10
    action_sequence = []
    for action_str in action_strs:
        action = env.action_space.no_op()
        action[action_str] = 1
        action_sequence.append(action)
    final_action = env.action_space.no_op()

    agent = SequentialAgent(env.action_space, action_sequence, final_action=final_action)
    # agent = AlwaysJumpingAgent(agent)
    agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    run_single_episode(env, agent)

def safety_unit_test():
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    grid_mins = (-2, -2, -2)
    grid_maxs = (2, 2, 2)
    viewpoint = 2
    max_episode_steps = 20

    env = gym.make('MineRLSafetyUnitTest-v0')
    env = VideoWrapper(env, 'imgs/inner_safety_unit_test.mp4', fps=30)
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/safety_unit_test.gif', fps=3)

    env.unwrapped.xml_file = fill_in_xml(env.xml_file, {
        'GRID_MIN_X' : grid_mins[2],
        'GRID_MIN_Y' : grid_mins[1],
        'GRID_MIN_Z' : grid_mins[0],
        'GRID_MAX_X' : grid_maxs[2],
        'GRID_MAX_Y' : grid_maxs[1],
        'GRID_MAX_Z' : grid_maxs[0],
        'VIEWPOINT' : viewpoint,
    })

    env.seed(seed)

    action_strs = ['forward'] * 10
    action_sequence = []
    for action_str in action_strs:
        action = env.action_space.no_op()
        action[action_str] = 1
        action_sequence.append(action)
    final_action = env.action_space.no_op()

    agent = SequentialAgent(env.action_space, action_sequence, final_action=final_action)
    agent = SafeAgentWrapper(agent)
    agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    run_single_episode(env, agent)

def safety_unit_test2():
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    grid_mins = (-2, -2, -2)
    grid_maxs = (2, 2, 2)
    viewpoint = 2
    max_episode_steps = 20

    env = gym.make('MineRLSafetyUnitTest2-v0')
    env = VideoWrapper(env, 'imgs/inner_safety_unit_test2.mp4', fps=30)
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/safety_unit_test2.gif', fps=30)

    env.unwrapped.xml_file = fill_in_xml(env.xml_file, {
        'GRID_MIN_X' : grid_mins[2],
        'GRID_MIN_Y' : grid_mins[1],
        'GRID_MIN_Z' : grid_mins[0],
        'GRID_MAX_X' : grid_maxs[2],
        'GRID_MAX_Y' : grid_maxs[1],
        'GRID_MAX_Z' : grid_maxs[0],
        'VIEWPOINT' : viewpoint,
    })

    env.seed(seed)

    action_strs = ['left'] * 10
    action_sequence = []
    for action_str in action_strs:
        action = env.action_space.no_op()
        action[action_str] = 1
        action_sequence.append(action)
    final_action = env.action_space.no_op()

    agent = SequentialAgent(env.action_space, action_sequence, final_action=final_action)
    agent = SafeAgentWrapper(agent)
    agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    run_single_episode(env, agent)

def grid_unit_test2():
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    grid_mins = (-2, -2, -2)
    grid_maxs = (2, 2, 2)
    viewpoint = 1
    max_episode_steps = 100

    env = gym.make('MineRLBumpyRoomTest-v0')
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/grid_unit_test2.gif', fps=3)

    env.unwrapped.xml_file = fill_in_xml(env.xml_file, {
        'GRID_MIN_X' : grid_mins[2],
        'GRID_MIN_Y' : grid_mins[1],
        'GRID_MIN_Z' : grid_mins[0],
        'GRID_MAX_X' : grid_maxs[2],
        'GRID_MAX_Y' : grid_maxs[1],
        'GRID_MAX_Z' : grid_maxs[0],
        'VIEWPOINT' : viewpoint,
    })

    env.seed(seed)

    agent = RandomAgent(env.action_space)
    agent = AlwaysJumpingAgent(agent)
    # agent = SafeAgentWrapper(agent)
    agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    run_single_episode(env, agent)

def grid_unit_test():
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    grid_mins = (-2, -1, -2)
    grid_maxs = (2, -1, 2)
    viewpoint = 1
    max_episode_steps = 20

    env = gym.make('MineRLGridUnitTest-v0')
    env = VideoWrapper(env, 'imgs/inner_grid_unit_test.mp4', fps=30)
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/grid_unit_test.gif', fps=3)

    env.unwrapped.xml_file = fill_in_xml(env.xml_file, {
        'GRID_MIN_X' : grid_mins[2],
        'GRID_MIN_Y' : grid_mins[1],
        'GRID_MIN_Z' : grid_mins[0],
        'GRID_MAX_X' : grid_maxs[2],
        'GRID_MAX_Y' : grid_maxs[1],
        'GRID_MAX_Z' : grid_maxs[0],
        'VIEWPOINT' : viewpoint,
    })

    env.seed(seed)

    action_strs = ['back', 'left', 'forward', 'forward', 'right', 'right', 'back', 'back', 'left']
    action_sequence = []
    for action_str in action_strs:
        action = env.action_space.no_op()
        action[action_str] = 1
        action_sequence.append(action)
    final_action = env.action_space.no_op()

    agent = SequentialAgent(env.action_space, action_sequence, final_action=final_action)
    agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    run_single_episode(env, agent)


def foraging_test():
    seed = 3
    np.random.seed(seed)
    random.seed(seed)

    grid_mins = (-2, -2, -2)
    grid_maxs = (2, 2, 2)
    viewpoint = 1
    max_episode_steps = 250

    env = gym.make('MineRLForaging-v0')
    env = VideoWrapper(env, 'imgs/inner_foraging_test.mp4', fps=30)
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/foraging_test.gif', fps=30)

    env.unwrapped.xml_file = fill_in_xml(env.xml_file, {
        'GRID_MIN_X' : grid_mins[2],
        'GRID_MIN_Y' : grid_mins[1],
        'GRID_MIN_Z' : grid_mins[0],
        'GRID_MAX_X' : grid_maxs[2],
        'GRID_MAX_Y' : grid_maxs[1],
        'GRID_MAX_Z' : grid_maxs[0],
        'VIEWPOINT' : viewpoint,
    })

    env.seed(seed)

    agent = RandomAgent(env.action_space)
    agent = AlwaysJumpingAgent(agent)
    agent = SafeAgentWrapper(agent)
    agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    def finish_fn():
        with open('foraging_pos_to_ore.pkl', 'wb') as f:
            pickle.dump(agent.pos_to_ore, f)

    agent = AgentObsWrapper(agent, lambda x : True, finish_fn)

    run_single_episode(env, agent)

def fsm_maze_test():
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    grid_mins = (-2, -2, -2)
    grid_maxs = (2, 2, 2)
    viewpoint = 1
    max_episode_steps = 20

    env = gym.make('MineRLMazeTest-v0')
    env = VideoWrapper(env, 'imgs/inner_maze_test.mp4', fps=30)
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/maze_test.gif', fps=30)

    env.unwrapped.xml_file = fill_in_xml(env.xml_file, {
        'GRID_MIN_X' : grid_mins[2],
        'GRID_MIN_Y' : grid_mins[1],
        'GRID_MIN_Z' : grid_mins[0],
        'GRID_MAX_X' : grid_maxs[2],
        'GRID_MAX_Y' : grid_maxs[1],
        'GRID_MAX_Z' : grid_maxs[0],
        'VIEWPOINT' : viewpoint,
    })

    env.seed(seed)

    action_strs = ['right', 'right', 'right', 'right', 'forward', 'forward', 'forward', 
                    'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 
                    'back', 'back', 'back', 'right', 'right', 'right']
    action_sequence = []
    for action_str in action_strs:
        action = env.action_space.no_op()
        action[action_str] = 1
        action_sequence.append(action)
    final_action = env.action_space.no_op()

    action_sequence0, action_sequence1 = action_sequence[:11], action_sequence[11:]

    agent0 = SequentialAgent(env.action_space, action_sequence0, final_action=final_action)
    agent1 = SequentialAgent(env.action_space, action_sequence1, final_action=final_action)
    done0 = DoneTimer(len(action_sequence0))
    done1 = lambda obs : False
    agent = FSMAgent([(agent0, done0), (agent1, done1)])
    agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    run_single_episode(env, agent)

def pure_search_test():
    maze2d = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 2, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    ])

    action_list = ['up', 'down', 'left', 'right']
    action_effects = [(-1, 0),  (1, 0), (0, -1), (0, 1)]

    def model(state, action):
        possible_next_state = state + action_effects[action_list.index(action)]

        if possible_next_state[0] < 0 or possible_next_state[1] < 0:
            return state

        if possible_next_state[0] >= maze2d.shape[0] or possible_next_state[1] >= maze2d.shape[1]:
            return state

        if maze2d[possible_next_state[0], possible_next_state[1]] == 1:
            return state

        return possible_next_state

    def goal_check(state, goal):
        return np.all(state == goal)

    def heuristic(state):
        return 0

    init_state = np.array([0, 0])
    goal = np.argwhere(maze2d == 2)[0]

    planner = Planner(model, action_list, goal_check, heuristic)
    out = planner.plan(init_state, goal)
    print("Plan:", out.plan)

def search_maze_test():
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    grid_mins = (-1, -1, -1)
    grid_maxs = (1, 1, 1)
    viewpoint = 1
    max_episode_steps = 20

    env = gym.make('MineRLMazeTest-v0')
    env = VideoWrapper(env, 'imgs/inner_fsm_maze_test.mp4', fps=30)
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/fsm_maze_test.gif', fps=30)

    env.unwrapped.xml_file = fill_in_xml(env.xml_file, {
        'GRID_MIN_X' : grid_mins[2],
        'GRID_MIN_Y' : grid_mins[1],
        'GRID_MIN_Z' : grid_mins[0],
        'GRID_MAX_X' : grid_maxs[2],
        'GRID_MAX_Y' : grid_maxs[1],
        'GRID_MAX_Z' : grid_maxs[0],
        'VIEWPOINT' : viewpoint,
    })

    env.seed(seed)

    action_strs = ['right', 'right', 'right', 'right', 'forward', 'forward', 'forward', 
                    'left', 'left', 'left', 'left']
    action_sequence = []
    for action_str in action_strs:
        action = env.action_space.no_op()
        action[action_str] = 1
        action_sequence.append(action)
    final_action = env.action_space.no_op()

    agent0 = SequentialAgent(env.action_space, action_sequence, final_action=final_action)
    done0 = DoneTimer(len(action_sequence))
    agent1 = SearchAgent(env.action_space)
    done1 = lambda obs : False
    agent = FSMAgent([(agent0, done0), (agent1, done1)])
    agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    agent1.set_goal((0, 2, 0))
    # agent1.subscribe_to_grid_agent(agent)

    run_single_episode(env, agent)

def search_ascending_maze_test():
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    grid_mins = (-1, -1, -1)
    grid_maxs = (1, 1, 1)
    viewpoint = 1
    max_episode_steps = 20

    env = gym.make('MineRLAscendingMazeTest-v0')
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/ascending_maze_test.gif', fps=30)

    env.unwrapped.xml_file = fill_in_xml(env.xml_file, {
        'GRID_MIN_X' : grid_mins[2],
        'GRID_MIN_Y' : grid_mins[1],
        'GRID_MIN_Z' : grid_mins[0],
        'GRID_MAX_X' : grid_maxs[2],
        'GRID_MAX_Y' : grid_maxs[1],
        'GRID_MAX_Z' : grid_maxs[0],
        'VIEWPOINT' : viewpoint,
    })

    env.seed(seed)

    action_strs = ['right', 'right', 'right', 'right', 'forward', 'forward', 'forward', 'forward', 
                    'left', 'left', 'left', 'left']
    # action_strs = ['right', 'right', 'right', 'right', 'forward', 'forward', 'forward', 'forward',
    #             'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 
    #             'back', 'back', 'back', 'right', 'right', 'right']
    action_sequence = []
    for action_str in action_strs:
        action = env.action_space.no_op()
        action[action_str] = 1
        action_sequence.append(action)
    final_action = env.action_space.no_op()

    # action_sequence0, action_sequence1 = action_sequence[:11], action_sequence[11:]

    # agent0 = SequentialAgent(env.action_space, action_sequence0, final_action=final_action)
    # agent1 = SequentialAgent(env.action_space, action_sequence1, final_action=final_action)
    # done0 = DoneTimer(len(action_sequence0))
    # done1 = lambda obs : False
    # agent = FSMAgent([(agent0, done0), (agent1, done1)])
    # agent = AlwaysJumpingAgent(agent)
    # agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    agent0 = SequentialAgent(env.action_space, action_sequence, final_action=final_action)
    done0 = DoneTimer(len(action_sequence))
    agent1 = SearchAgent(env.action_space)
    done1 = lambda obs : False
    agent = FSMAgent([(agent0, done0), (agent1, done1)])
    agent = AlwaysJumpingAgent(agent)
    agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    agent1.set_goal((0, 2, 0))
    # agent1.subscribe_to_grid_agent(agent)

    run_single_episode(env, agent)


def open_room_test(agent_type='exploring'):
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    grid_mins = (-1, -1, -1)
    grid_maxs = (1, 1, 1)
    viewpoint = 1
    max_episode_steps = 100

    env = gym.make('MineRLOpenRoomTest-v0')
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/open_room_test_{}.gif'.format(agent_type), fps=3)

    env.unwrapped.xml_file = fill_in_xml(env.xml_file, {
        'GRID_MIN_X' : grid_mins[2],
        'GRID_MIN_Y' : grid_mins[1],
        'GRID_MIN_Z' : grid_mins[0],
        'GRID_MAX_X' : grid_maxs[2],
        'GRID_MAX_Y' : grid_maxs[1],
        'GRID_MAX_Z' : grid_maxs[0],
        'VIEWPOINT' : viewpoint,
    })

    env.seed(seed)

    base_dir = 'imgs/{}_agent'.format(agent_type)
    if agent_type == 'random':
        agent = RandomAgent(env.action_space)
        agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    elif agent_type == 'roomba':
        agent = RoombaAgent(env.action_space)
        agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)
    
    elif agent_type == 'exploring':
        search_agent = ExploringSearchAgent(env.action_space)
        agent = GridBuildingAgentWrapper(search_agent, grid_mins=grid_mins, grid_maxs=grid_maxs)
        # search_agent.subscribe_to_grid_agent(agent)

    else:
        raise NotImplementedError()

    time_counter = itertools.count()
    filenames = []

    max_bounds = (-5, 5, 1, 3, -5, 5)

    def inner_fn(obs):
        outdir = os.path.join(base_dir, '{}'.format(next(time_counter)))
        filenames.append(os.path.join(outdir, 'plane1.png'))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        agent.draw_grid_images(outdir=outdir, max_bounds=max_bounds)

    def finish_fn():
        images = [imageio.imread(f) for f in filenames]
        outfile = os.path.join(base_dir, '{}.gif'.format(agent_type))
        imageio.mimsave(outfile, images, fps=3)
        print("Wrote out video to {}.".format(outfile))

    final_agent = AgentObsWrapper(agent, inner_fn, finish_fn)

    run_single_episode(env, final_agent)


def bumpy_room_test(agent_type='exploring'):
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    grid_mins = (-1, -1, -1)
    grid_maxs = (1, 1, 1)
    viewpoint = 1
    max_episode_steps = 100

    env = gym.make('MineRLBumpyRoomTest-v0')
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/bumpy_room_test_{}.gif'.format(agent_type), fps=3)

    env.unwrapped.xml_file = fill_in_xml(env.xml_file, {
        'GRID_MIN_X' : grid_mins[2],
        'GRID_MIN_Y' : grid_mins[1],
        'GRID_MIN_Z' : grid_mins[0],
        'GRID_MAX_X' : grid_maxs[2],
        'GRID_MAX_Y' : grid_maxs[1],
        'GRID_MAX_Z' : grid_maxs[0],
        'VIEWPOINT' : viewpoint,
    })

    env.seed(seed)

    base_dir = 'imgs/{}_agent'.format(agent_type)
    if agent_type == 'random':
        agent = RandomAgent(env.action_space)
        agent = AlwaysJumpingAgent(agent)
        agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    elif agent_type == 'roomba':
        agent = RoombaAgent(env.action_space)
        agent = AlwaysJumpingAgent(agent)
        agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)
    
    elif agent_type == 'exploring':
        agent = ExploringSearchAgent(env.action_space)
        agent = AlwaysJumpingAgent(agent)
        agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)
        # search_agent.subscribe_to_grid_agent(agent)

    else:
        raise NotImplementedError()

    time_counter = itertools.count()
    filenames = []

    max_bounds = (-5, 5, 1, 6, -5, 5)

    def inner_fn(obs):
        outdir = os.path.join(base_dir, '{}'.format(next(time_counter)))
        filenames.append(os.path.join(outdir, 'plane1.png'))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        agent.draw_grid_images(outdir=outdir, max_bounds=max_bounds)

    def finish_fn():
        images = [imageio.imread(f) for f in filenames]
        outfile = os.path.join(base_dir, '{}.gif'.format(agent_type))
        imageio.mimsave(outfile, images, fps=3)
        print("Wrote out video to {}.".format(outfile))

    final_agent = AgentObsWrapper(agent, inner_fn, finish_fn)

    run_single_episode(env, final_agent)

def forage_explore(agent_type='random', seed=4):
    np.random.seed(seed)
    random.seed(seed)

    grid_mins = (-2, -2, -2)
    grid_maxs = (2, 2, 2)
    viewpoint = 1
    max_episode_steps = 250

    env = gym.make('MineRLForaging-v0')
    env = VideoWrapper(env, 'imgs/inner_forage_explore_test.mp4', fps=30)
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/forage_explore_test.gif', fps=30)

    env.unwrapped.xml_file = fill_in_xml(env.xml_file, {
        'GRID_MIN_X' : grid_mins[2],
        'GRID_MIN_Y' : grid_mins[1],
        'GRID_MIN_Z' : grid_mins[0],
        'GRID_MAX_X' : grid_maxs[2],
        'GRID_MAX_Y' : grid_maxs[1],
        'GRID_MAX_Z' : grid_maxs[0],
        'VIEWPOINT' : viewpoint,
    })

    env.seed(seed)

    if agent_type == 'random':
        agent = RandomAgent(env.action_space)
        agent = AlwaysJumpingAgent(agent)
        agent = SafeAgentWrapper(agent)
        agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    elif agent_type == 'roomba':
        agent = RoombaAgent(env.action_space)
        agent = AlwaysJumpingAgent(agent)
        agent = SafeAgentWrapper(agent)
        agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)
    
    elif agent_type == 'exploring':
        agent = ExploringSearchAgent(env.action_space)
        agent = AlwaysJumpingAgent(agent)
        agent = SafeAgentWrapper(agent)
        agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    def finish_fn():
        with open('foraging_{}_pos_to_ore.pkl'.format(agent_type), 'wb') as f:
            pickle.dump(agent.pos_to_ore, f)

    agent = AgentObsWrapper(agent, lambda x : True, finish_fn)
    run_single_episode(env, agent)


def visualize_3d_test():
    seed = 0
    np.random.seed(seed)
    random.seed(seed)

    grid_mins = (-2, -2, -2)
    grid_maxs = (2, 2, 2)
    viewpoint = 1
    max_episode_steps = 2500

    env = gym.make('MineRLForaging-v0')
    env = VideoWrapper(env, 'imgs/inner_foraging_test.mp4', fps=30)
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/foraging_test.mp4', fps=30)

    env.unwrapped.xml_file = fill_in_xml(env.xml_file, {
        'GRID_MIN_X' : grid_mins[2],
        'GRID_MIN_Y' : grid_mins[1],
        'GRID_MIN_Z' : grid_mins[0],
        'GRID_MAX_X' : grid_maxs[2],
        'GRID_MAX_Y' : grid_maxs[1],
        'GRID_MAX_Z' : grid_maxs[0],
        'VIEWPOINT' : viewpoint,
    })

    env.seed(seed)

    search_agent = ExploringSearchAgent(env.action_space)
    search_agent = AlwaysJumpingAgent(search_agent)
    search_agent = SafeAgentWrapper(search_agent)
    agent = GridBuildingAgentWrapper(search_agent, grid_mins=grid_mins, grid_maxs=grid_maxs)
    safe_agent = SafeAgentWrapper(agent)
    # search_agent.subscribe_to_grid_agent(agent)

    def finish_fn():
        with open('explore_pos_to_ore.pkl', 'wb') as f:
            pickle.dump(agent.pos_to_ore, f)

    final_agent = AgentObsWrapper(safe_agent, lambda x : True, finish_fn)

    run_single_episode(env, final_agent)

def visualize_3d_from_pkl(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        pos_to_ore = pickle.load(f)

    full_grid = build_full_grid(pos_to_ore)
    full_grid = np.swapaxes(full_grid, 1, 2)

    colors = np.moveaxis(np.vectorize(grid_colors.get)(full_grid), 0, -1)
    colors = colors.astype(np.float32) / 255.
    voxels = (full_grid != 'unk') & (full_grid != 'air')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=colors) #, edgecolor='k')
    ax.axis('off')

    max_range = np.array([voxels.shape]).max() / 2.0
    mid_x = (voxels.shape[0]) * 0.5
    mid_y = (voxels.shape[1]) * 0.5
    mid_z = (voxels.shape[2]) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    base_dir = 'imgs/visualize_3d'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    angles = np.linspace([0., 0.], [30., 30.], num=20).tolist()
    angles += np.linspace([30., 30.], [30., 360.], num=60).tolist()

    filenames = []

    for i, (elev, azim) in enumerate(angles):
        ax.view_init(elev=elev, azim=azim)
        filename = os.path.join(base_dir, "view%d.png" % i)
        plt.savefig(filename, dpi=250)
        filenames.append(filename)

    images = [imageio.imread(f) for f in filenames]
    outfile = os.path.join(base_dir, 'out.gif')
    imageio.mimsave(outfile, images, fps=5)
    print("Wrote out video to {}.".format(outfile))

def grid_2d_env_test():
    layout = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    ])
    init_pos = (0, 0)
    max_episode_steps = 500

    env = Grid2DEnv(layout, init_pos)
    env = VideoWrapper(env, 'imgs/grid2denv.gif', fps=30)
    env = TimeLimit(env, max_episode_steps)

    # agent = RandomAgent(env.action_space)
    agent = ExploringSearchAgent(env.action_space)

    run_single_episode(env, agent, verbose=True)

def wood_unit_test():
    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    max_episode_steps = 20
    grid_mins = (-2, -1, -2)
    grid_maxs = (2, -1, 2)
    viewpoint = 1

    env = gym.make('MineRLWoodUnitTest-v0')
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/wood_unit_test.gif', fps=10)

    env.unwrapped.xml_file = fill_in_xml(env.xml_file, {
        'GRID_MIN_X' : grid_mins[2],
        'GRID_MIN_Y' : grid_mins[1],
        'GRID_MIN_Z' : grid_mins[0],
        'GRID_MAX_X' : grid_maxs[2],
        'GRID_MAX_Y' : grid_maxs[1],
        'GRID_MAX_Z' : grid_maxs[0],
        'VIEWPOINT' : viewpoint,
    })

    env.seed(seed)

    action_strs = ['forward'] * 5 + ['attack'] * 5
    action_sequence = []
    for action_str in action_strs:
        action = env.action_space.no_op()
        action[action_str] = 1
        action_sequence.append(action)
    final_action = env.action_space.no_op()

    action_sequence[10]['crouch'] = 1
    action_sequence[10]['attack'] = 1

    agent = SequentialAgent(env.action_space, action_sequence, final_action=final_action)
    agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    run_single_episode(env, agent)



if __name__ == "__main__":
    # flatworld_demo()
    # grid_unit_test()
    # stairs_unit_test()
    # grid_unit_test2()
    # safety_unit_test()
    # safety_unit_test2()
    # safety_unit_test3()
    # foraging_test()
    # visualize_3d_from_pkl('foraging_exploring_pos_to_ore.pkl')
    # fsm_maze_test()
    # pure_search_test()
    # search_maze_test()
    # search_ascending_maze_test()
    # open_room_test(agent_type='roomba')
    # bumpy_room_test(agent_type='exploring')
    # forage_explore(agent_type='exploring', seed=24)
    # visualize_3d_test()
    # grid_2d_env_test()
    wood_unit_test()

