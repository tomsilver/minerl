import imageio
import numpy as np

def run_single_episode(env, agent):
    video_frames = []
    rewards = []

    obs = env.reset()
    agent.reset(obs)

    total_reward = 0.
    
    while True:
        action = agent(obs)
        obs, reward, done, debug_info = env.step(action)
        agent.observe(obs, reward, done, debug_info)

        total_reward += reward
        rewards.append(reward)

        if done:
            break

    agent.finish_episode()
    env.close()

    return rewards
