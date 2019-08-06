# Simple env test.
import logging

import matplotlib.pyplot as plt

import gym
import minerl

from minerl.env.wrappers import GridWorldWrapper

import numpy as np

import coloredlogs
coloredlogs.install(logging.INFO)



def main():
    """
    Tests running a simple environment.
    """
    all_positions = []

    env = gym.make('MineRLFlatGrid-v0')
    env = GridWorldWrapper(env)

    env.seed(420)
    obs = env.reset()
    position = np.array([obs['XPos'], obs['YPos'], obs['ZPos']])
    print("position:", position.round())
    all_positions.append(position)
    done = False
    while not done:
        action = env.action_space.noop()
        action['forward'] = 1
        action['left'] = 1
        obs, reward, done, info = env.step(action)
        position = np.array([obs['XPos'], obs['YPos'], obs['ZPos']])
        print("position:", position.round())
        all_positions.append(position)

    ts = np.arange(len(all_positions))
    xs, ys, zs = np.transpose(all_positions)
    dxs = np.subtract(xs[1:], xs[:-1])
    dys = np.subtract(ys[1:], ys[:-1])
    dzs = np.subtract(zs[1:], zs[:-1])
    ddxs = np.subtract(dxs[1:], dxs[:-1])
    ddys = np.subtract(dys[1:], dys[:-1])
    ddzs = np.subtract(dzs[1:], dzs[:-1])

    fig, axes = plt.subplots(3, 3, sharex=True)
    axes[0][0].plot(ts, xs)
    axes[0][1].plot(ts, ys)
    axes[0][2].plot(ts, zs)

    axes[0][0].set_title("XPos")
    axes[0][1].set_title("YPos")
    axes[0][2].set_title("ZPos")

    axes[1][0].plot(ts[1:], dxs)
    axes[1][1].plot(ts[1:], dys)
    axes[1][2].plot(ts[1:], dzs)

    axes[1][0].set_title("Delta XPos")
    axes[1][1].set_title("Delta YPos")
    axes[1][2].set_title("Delta ZPos")

    axes[2][0].plot(ts[2:], ddxs)
    axes[2][1].plot(ts[2:], ddys)
    axes[2][2].plot(ts[2:], ddzs)

    axes[2][0].set_title("Delta Delta XPos")
    axes[2][1].set_title("Delta Delta YPos")
    axes[2][2].set_title("Delta Delta ZPos")

    plt.tight_layout()
    plt.show()


    print("Demo complete.")

if __name__ == "__main__":
    main()
