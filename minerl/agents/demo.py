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

seed = 3
np.random.seed(seed)
random.seed(seed)

grid_mins = (-2, -2, -2)
grid_maxs = (2, 2, 2)
viewpoint = 1
max_episode_steps = 250

if __name__ == "__main__":
    # env = gym.make('MineRLGridUnitTest-v0')
    env = gym.make('MineRLForaging-v0')
    # env = gym.make('MineRLStairsUnitTest-v0')
    # env = gym.make('MineRLSafetyUnitTest-v0')
    # env = gym.make('MineRLSafetyUnitTest2-v0')
    # env = gym.make('MineRLSafetyUnitTest3-v0')
    env = GridWorldWrapper(env, grid_mins=grid_mins, grid_maxs=grid_maxs)
    env = TimeLimit(env, max_episode_steps)
    env = VideoWrapper(env, 'imgs/demo_pov.gif', fps=30)

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

    # action_strs = ['back', 'left', 'forward', 'forward', 'right', 'right', 'back', 'back', 'left']
    action_strs = ['forward'] * 6 # + ['back'] * 6
    # action_strs = ['left'] * 10 # + ['back'] * 6
    action_sequence = []
    for action_str in action_strs:
        action = env.action_space.noop()
        action[action_str] = 1
        action_sequence.append(action)
    final_action = env.action_space.noop()

    agent = RandomAgent(env.action_space)
    # agent = SequentialAgent(env.action_space, action_sequence, final_action=final_action)
    agent = AlwaysJumpingAgent(agent)
    agent = SafeAgentWrapper(agent)
    agent = GridBuildingAgentWrapper(agent, grid_mins=grid_mins, grid_maxs=grid_maxs)

    rewards = run_single_episode(env, agent)
