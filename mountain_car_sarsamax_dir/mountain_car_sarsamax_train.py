import math

import gym
import numpy as np
import matplotlib.pyplot as plt

# instantiate environment
ENVIRONMENT1 = 'MountainCar-v0'
env = gym.make(ENVIRONMENT1)
env._max_episode_steps = 1000

def main():

    # PARAMS
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 0.9  # For greedy epsilon exploration
    epsilon_start = 0.9
    min_epsilon = 0.01
    num_episodes = 10000
    decay = 0.99
    decreasing_decay = 0.001

    # print(env.observation_space)
    # print(env.observation_space.high)
    # print(env.observation_space.low)

    # FOR X VALUE DISCRETIZATION: ROUND TO NEAREST 0.1, DIVIDE BY 0.1, ADD 12 therefore -1.2 is state 0
    # FOR FORCE VALUE, ROUND TO NEAREST 0.01, DIVIDE BY 0.01, ADD 7

    q_table = np.zeros((18, 14, 3))  # AN 18 X 14 X 3 Matrix denoting the state action space for environment

    # np.argmax(q_table[0][0]) The following can find an argmax for an action
    # np.random.binomial(1,epsilon,1) == 1  The following can be used for greedy epsilon exploration

    scores = []
    episodes = []
    num_eps = 0

    for episode in range(num_episodes):

        score = 0
        obs = env.reset()
        done = False

        # Initialize S
        curr_x, curr_y = discrete_state(obs[0], obs[1])

        while not done:

            # Predict action using Q table
            action = predict(curr_x, curr_y, q_table, epsilon)
            obs, reward, done, info = env.step(action)

            score += reward

            # Update Q table
            future_x, future_y = discrete_state(obs[0], obs[1])
            #max_a = np.argmax(q_table[future_x][future_y])
            curr_q = q_table[curr_x][curr_y][action]
            q_table[curr_x][curr_y][action] = curr_q + alpha * (reward + gamma * max(q_table[future_x][future_y]) - curr_q)

            curr_x = future_x
            curr_y = future_y

        scores.append(score)
        episodes.append(episode+1)

        #epsilon = max(min_epsilon, np.exp(-decreasing_decay*episode))

        #epsilon = max(epsilon_start*math.e**(-decreasing_decay*episode), min_epsilon)

        #if (episode + 1) % 20 == 0:
        #    epsilon *= decay

        if (episode + 1) % 20 == 0:
            epsilon *= decay

        if (episode + 1) % 100 == 0:
            np.save('./mountain_car_sarsamax/mountain_car_sarsamax_{f}_episodes.npy'.format(f=str(len(episodes))), q_table)
            num_eps = episode + 1

    mean_rewards = np.zeros(num_episodes)
    for r in range(num_episodes):
        mean_rewards[r] = np.mean(scores[max(0, r-50):(r+1)])

    plt.plot(episodes, mean_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.title('Total reward over ' + str(len(episodes)) + ' episodes')
    #plt.show()
    plt.savefig('mountaincar_reward_sarsamax_{f}'.format(f=num_eps))

    np.save('./mountain_car_sarsamax/mountain_car_sarsamax_{f}_episodes_mean_rewards.npy'.format(f=str(num_episodes)), mean_rewards)

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
    print(avg_reward)

    env.close()


def discrete_state(val1, val2):

    result_x = round(val1, 1)
    result_x = result_x/0.1 + 12

    result_y = round(val2, 2)
    result_y = result_y/0.01 + 7

    return int(result_x), int(result_y)


def predict(statex, statey, q_table, epsilon):
    if np.random.binomial(1, epsilon, 1) == 1:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[statex][statey])


if __name__ == '__main__':
    main()