import imageio
import numpy as np
import tempfile

def run_single_episode(env, agent, verbose=False):
    video_frames = []
    rewards = []

    obs = env.reset()
    agent.reset(obs)

    if verbose:
        print("Reset obs:", obs)

    total_reward = 0.
    
    while True:
        action = agent(obs)
        obs, reward, done, debug_info = env.step(action)
        agent.observe(obs, reward, done, debug_info)

        if verbose:
            print("Action:", action)
            print("Obs:", obs)

        total_reward += reward
        rewards.append(reward)

        if done:
            break

    agent.finish_episode()
    env.close()

    return rewards

def fill_in_xml(xml_file, fill_ins):
    with open(xml_file, 'r') as f:
        xml = f.read()
    
    for placeholder, fill_in in fill_ins.items(): 
        xml = xml.replace('$({})'.format(placeholder), str(fill_in))

    new_f = tempfile.NamedTemporaryFile(mode='w', delete=False)
    new_f.write(xml)

    return new_f.name
