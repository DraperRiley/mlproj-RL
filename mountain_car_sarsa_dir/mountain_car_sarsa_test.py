import gym
import numpy as np
import matplotlib.pyplot as plt

# instantiate environment
ENVIRONMENT1 = 'MountainCar-v0'
env = gym.make(ENVIRONMENT1)
env._max_episode_steps = 500

"""
Observations:
Static epsilon seems to perform the best
Linear decay performs okay
epsilon = epsilon*0.9 performs okay
"""

def main():

    q_table = np.load('./mountain_car_sarsa/mountain_car_sarsa_10000_episodes.npy')
    #print(q_table)

    score = 0
    total_score = 0
    num_eps = 100

    for i in range(num_eps):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            # Get current state
            curr_x, curr_y = discrete_state(obs[0], obs[1])
            #env.render()
            action = np.argmax(q_table[curr_x][curr_y])
            obs, reward, done, info = env.step(action)
            score += reward
        total_score += score

    avg_reward = total_score / num_eps

    print('Average reward {f}'.format(f=avg_reward))

    return 0

def discrete_state(val1, val2):

    result_x = round(val1, 1)
    result_x = result_x/0.1 + 12

    result_y = round(val2, 2)
    result_y = result_y/0.01 + 7

    return int(result_x), int(result_y)


if __name__ == '__main__':
    main()