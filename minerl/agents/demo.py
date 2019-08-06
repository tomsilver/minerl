from grid_building_agent import GridBuildingAgent
from random_agent import RandomAgent
from minerl.env.wrappers import GridWorldWrapper

from utils import run_single_episode

import gym
import minerl
import numpy as np
import random

seed = 1
np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    env = gym.make('MineRLForagingGrid-v0')
    env = GridWorldWrapper(env)

    env.seed(seed)

    # agent = RandomAgent(env.action_space)
    agent = GridBuildingAgent(env.action_space)

    video_out_path = 'imgs/foraging_demo.gif'
    max_num_steps = 500

    rewards = run_single_episode(env, agent, record_video=True, 
        video_out_path=video_out_path, max_num_steps=max_num_steps)
    print("Wrote out video to {}.".format(video_out_path))
