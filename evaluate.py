import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# instantiate environment
ENVIRONMENT1 = 'MountainCar-v0'
env = gym.make(ENVIRONMENT1)
env._max_episode_steps = 1000

def main():

    sarsa_dir = './mountain_car_sarsa_dir/mountain_car_sarsa/mountain_car_sarsa_{f}_episodes.npy'
    sarsamax_dir = './mountain_car_sarsamax_dir/mountain_car_sarsamax/mountain_car_sarsamax_{f}_episodes.npy'
    sarsalambda_dir = './mountain_car_sarsalambda_dir/mountain_car_sarsalambda/mountain_car_sarsalambda_{f}_episodes.npy'

    increment = 1000
    num_eps = 50
    count = increment
    EPS = 10000

    x = []
    y = []

    while count <= EPS:

        q_table = np.load(sarsa_dir.format(f=count))
        score = 0

        for i in range(num_eps):

            obs = env.reset()
            done = False

            while not done:

                # Get current state
                curr_x, curr_y = discrete_state(obs[0], obs[1])
                #env.render()
                action = np.argmax(q_table[curr_x][curr_y])
                obs, reward, done, info = env.step(action)
                score += reward

        average = score / num_eps
        x.append(count)
        y.append(average)
        count += increment

    plt.plot(x, y, label = 'SARSA')
    count = increment
    x = []
    y = []

    while count <= EPS:

        q_table = np.load(sarsamax_dir.format(f=count))
        score = 0

        for i in range(num_eps):

            obs = env.reset()
            done = False

            while not done:

                # Get current state
                curr_x, curr_y = discrete_state(obs[0], obs[1])
                #env.render()
                action = np.argmax(q_table[curr_x][curr_y])
                obs, reward, done, info = env.step(action)
                score += reward

        average = score / num_eps
        x.append(count)
        y.append(average)
        count += increment

    plt.plot(x, y, label = 'SARSAMAX')
    count = increment
    x = []
    y = []

    while count <= EPS:

        q_table = np.load(sarsalambda_dir.format(f=count))
        score = 0

        for i in range(num_eps):

            obs = env.reset()
            done = False

            while not done:

                # Get current state
                curr_x, curr_y = discrete_state(obs[0], obs[1])
                #env.render()
                action = np.argmax(q_table[curr_x][curr_y])
                obs, reward, done, info = env.step(action)
                score += reward

        average = score / num_eps
        x.append(count)
        y.append(average)
        count += increment

    plt.plot(x, y, label = 'SARSALAMBDA')
    count = increment

    plt.legend()
    plt.xlabel('Episodes trained')
    plt.ylabel('Average total reward')
    plt.title('Average reward over ' + str(num_eps) + ' episodes')
    plt.savefig('learning_evaluation')

    return 0


def discrete_state(val1, val2):

    result_x = round(val1, 1)
    result_x = result_x/0.1 + 12

    result_y = round(val2, 2)
    result_y = result_y/0.01 + 7

    return int(result_x), int(result_y)


if __name__ == '__main__':
    main()