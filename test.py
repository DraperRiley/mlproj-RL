import gym
import pandas as pd
import tensorboard as tb
import numpy as np
import math


def main():

    mean_rewards = np.load('./epsilon_tests/exponential_epsilon_mean.npy')
    mean = sum(mean_rewards)/10000
    print(mean)
    print(stddev(mean, mean_rewards))

    mean_rewards = np.load('./epsilon_tests/step_epsilon_mean.npy')
    mean = sum(mean_rewards)/10000
    print(mean)
    print(stddev(mean, mean_rewards))

    mean_rewards = np.load('./epsilon_tests/static_epsilon_mean.npy')
    mean = sum(mean_rewards)/10000
    print(mean)
    print(stddev(mean, mean_rewards))


def stddev(mean, rewards):
    sum = 0
    for x in rewards:
        sum += abs((x-mean))**2
    result = math.sqrt(sum/10000)
    return result


if __name__ == '__main__':
    main()