import math
import gym
import numpy as np
import matplotlib.pyplot as plt

# instantiate environment
ENVIRONMENT1 = 'MountainCar-v0'
env = gym.make(ENVIRONMENT1)
env._max_episode_steps = 1000


def main():

    mean_arr = np.load('agg1_mean_rewards.npy')
    mean = sum(mean_arr)/10000
    print(mean)
    print(stddev(mean, mean_arr))

    mean_arr = np.load('agg2_mean_rewards.npy')
    mean = sum(mean_arr)/10000
    print(mean)
    print(stddev(mean, mean_arr))

    return 0

def stddev(mean, rewards):
    sum = 0
    for x in rewards:
        sum += abs((x-mean))**2
    result = math.sqrt(sum/10000)
    return result

if __name__ == '__main__':
    main()