from grid_building_agent import GridBuildingAgentWrapper
from simple_agent_wrappers import AlwaysJumpingAgent, SafeAgentWrapper
from random_agent import RandomAgent
from sequential_agent import SequentialAgent
from minerl.env.wrappers import GridWorldWrapper

from utils import run_single_episode

import gym
import minerl
import numpy as np
import random

seed = 0
np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    # env = gym.make('MineRLGridUnitTest-v0')
    # env = gym.make('MineRLForaging-v0')
    # env = gym.make('MineRLStairsUnitTest-v0')
    env = gym.make('MineRLSafetyUnitTest-v0')
    env = GridWorldWrapper(env)

    env.seed(seed)

    # action_strs = ['back', 'left', 'forward', 'forward', 'right', 'right', 'back', 'back', 'left']
    action_strs = ['forward'] * 6 # + ['back'] * 6
    action_sequence = []
    for action_str in action_strs:
        action = env.action_space.noop()
        action[action_str] = 1
        action_sequence.append(action)
    final_action = env.action_space.noop()

    # agent = RandomAgent(env.action_space)
    agent = SequentialAgent(env.action_space, action_sequence, final_action=final_action)
    # agent = AlwaysJumpingAgent(agent)
    agent = SafeAgentWrapper(agent)
    agent = GridBuildingAgentWrapper(agent)

    video_out_path = 'imgs/foraging_demo.gif'
    max_num_steps = 500

    rewards = run_single_episode(env, agent, record_video=True, 
        video_out_path=video_out_path, max_num_steps=max_num_steps)
    print("Wrote out video to {}.".format(video_out_path))
