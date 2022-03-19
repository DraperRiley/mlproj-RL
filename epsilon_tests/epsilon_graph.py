import math

import gym
import numpy as np
import matplotlib.pyplot as plt

# instantiate environment
ENVIRONMENT1 = 'MountainCar-v0'
env = gym.make(ENVIRONMENT1)
env._max_episode_steps = 1000


def main():

    epsilon = 0.9
    epsilon_val = []
    episodes = []
    num_episodes = 10000
    decay = 0.99
    min_epsilon = 0.01
    decreasing_decay = 0.001
    epsilon_start = 0.9

    # Exponential decay
    epsilon = 0.9
    episodes = []
    epsilon_val = []
    for episode in range(num_episodes):
        epsilon = max(min_epsilon, epsilon_start*math.e**(-decreasing_decay*episode))
        episodes.append(episode + 1)
        epsilon_val.append(epsilon)

    plt.plot(episodes, epsilon_val, label='EXPONENTIAL')

    # Step decay
    epsilon = 0.9
    episodes = []
    epsilon_val = []
    for episode in range(num_episodes):
        if (episode + 1) % 20 == 0:
            epsilon *= decay
        episodes.append(episode + 1)
        epsilon_val.append(epsilon)

    plt.plot(episodes, epsilon_val, label='STEP')

    # Static epsilon
    epsilon = 0.2
    episodes = []
    epsilon_val = []
    for episode in range(num_episodes):
        episodes.append(episode + 1)
        epsilon_val.append(epsilon)

    plt.plot(episodes, epsilon_val, label='STATIC')

    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Epsilon value')
    plt.title('Epsilon over {f} episodes'.format(f=num_episodes))
    plt.savefig('epsilon_over_time')

    return 0


if __name__ == '__main__':
    main()