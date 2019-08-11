from grid_building_agent import GridBuildingAgentWrapper
from simple_agent_wrappers import AlwaysJumpingAgent, SafeAgentWrapper
from random_agent import RandomAgent
from sequential_agent import SequentialAgent
from minerl.env.wrappers import GridWorldWrapper, VideoWrapper
from utils import run_single_episode, fill_in_xml

from gym.wrappers import TimeLimit

import gym
import minerl
import numpy as np
import random
import logging
import coloredlogs
coloredlogs.install(logging.INFO)

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
    env = VideoWrapper(env, 'imgs/grid_unit_test.gif', fps=30)

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
    env = VideoWrapper(env, 'imgs/inner_stairs_unit_test.mp4', fps=30)
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
    agent = AlwaysJumpingAgent(agent)
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
    env = VideoWrapper(env, 'imgs/safety_unit_test.gif', fps=30)

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

def safety_unit_test3():
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    grid_mins = (-2, -2, -2)
    grid_maxs = (2, 2, 2)
    viewpoint = 2
    max_episode_steps = 20

    env = gym.make('MineRLSafetyUnitTest3-v0')
    env = VideoWrapper(env, 'imgs/inner_safety_unit_test3.mp4', fps=30)
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/safety_unit_test3.gif', fps=30)

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

def foraging_test():
    seed = 1
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

    run_single_episode(env, agent)

if __name__ == "__main__":
    # grid_unit_test()
    # stairs_unit_test()
    # safety_unit_test()
    # safety_unit_test2()
    # safety_unit_test3()
    foraging_test()
