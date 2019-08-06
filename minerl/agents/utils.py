import imageio
import numpy as np

def run_single_episode(env, agent, record_video=False, video_out_path=None, max_num_steps=100, fps=30):
    video_frames = []
    rewards = []

    obs = env.reset()
    agent.reset(obs)

    total_reward = 0.

    if record_video:
        img = env.render()
        video_frames.append(img)
    
    for t in range(max_num_steps):
        action = agent(obs)
        obs, reward, done, debug_info = env.step(action)
        agent.observe(obs, reward, done, debug_info)

        total_reward += reward
        rewards.append(reward)

        if record_video:
            img = env.render()
            video_frames.append(img)

        if done:
            break

    if record_video:
        imageio.mimsave(video_out_path, video_frames, fps=fps)

    agent.finish_episode()
    # env.close()

    return rewards
