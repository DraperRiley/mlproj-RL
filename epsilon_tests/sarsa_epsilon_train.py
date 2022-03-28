import gym
import numpy as np
import matplotlib.pyplot as plt
import math

# instantiate environment
ENVIRONMENT1 = 'MountainCar-v0'
env = gym.make(ENVIRONMENT1)
env._max_episode_steps = 1000

def main():

    """
    Test various methods of epsilon decay using SARSA
    """

    # PARAMS
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 0.9  # For greedy epsilon exploration
    num_episodes = 10000
    decay = 0.99
    min_epsilon = 0.01
    decreasing_decay = 0.001
    epsilon_start = 0.9

    # Exponential decay epsilon
    q_table = np.zeros((18, 14, 3))
    scores = []
    episodes = []
    num_eps = 0

    for episode in range(num_episodes):

        score = 0
        obs = env.reset()
        done = False
        curr_x, curr_y = discrete_state(obs[0], obs[1])
        action = predict(curr_x, curr_y, q_table, epsilon)

        while not done:

            # Take action A
            obs, reward, done, info = env.step(action)
            score += reward

            # Get state S'
            future_x, future_y = discrete_state(obs[0], obs[1])

            # Get A'
            action_prime = predict(future_x, future_y, q_table, epsilon)

            # Update Q-Table
            curr_q = q_table[curr_x][curr_y][action]
            q_table[curr_x][curr_y][action] = curr_q + alpha*(reward + gamma*q_table[future_x][future_y][action_prime] - curr_q)

            # S = S', A = A'
            curr_x = future_x
            curr_y = future_y
            action = action_prime

        scores.append(score)
        episodes.append(episode+1)

        epsilon = max(min_epsilon, epsilon_start*math.e**(-decreasing_decay*episode))

    mean_rewards = np.zeros(num_episodes)
    for r in range(num_episodes):
        mean_rewards[r] = np.mean(scores[max(0, r-50):(r+1)])

    plt.plot(episodes, mean_rewards, label='EXPONENTIAL')
    np.save('exponential_epsilon_mean.npy', mean_rewards)

    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.title('Total reward over ' + str(len(episodes)) + ' episodes using SARSA')

    # Step decay
    q_table = np.zeros((18, 14, 3))
    scores = []
    episodes = []
    num_eps = 0
    epsilon = 0.9

    for episode in range(num_episodes):

        score = 0
        obs = env.reset()
        done = False
        curr_x, curr_y = discrete_state(obs[0], obs[1])
        action = predict(curr_x, curr_y, q_table, epsilon)

        while not done:

            # Take action A
            obs, reward, done, info = env.step(action)
            score += reward

            # Get state S'
            future_x, future_y = discrete_state(obs[0], obs[1])

            # Get A'
            action_prime = predict(future_x, future_y, q_table, epsilon)

            # Update Q-Table
            curr_q = q_table[curr_x][curr_y][action]
            q_table[curr_x][curr_y][action] = curr_q + alpha*(reward + gamma*q_table[future_x][future_y][action_prime] - curr_q)

            # S = S', A = A'
            curr_x = future_x
            curr_y = future_y
            action = action_prime

        scores.append(score)
        episodes.append(episode+1)

        if (episode + 1) % 20 == 0:
            epsilon *= decay

    mean_rewards = np.zeros(num_episodes)
    for r in range(num_episodes):
        mean_rewards[r] = np.mean(scores[max(0, r-50):(r+1)])

    plt.plot(episodes, mean_rewards, label='STEP')
    np.save('step_epsilon_mean.npy', mean_rewards)

    # Static epsilon
    q_table = np.zeros((18, 14, 3))
    scores = []
    episodes = []
    num_eps = 0
    epsilon = 0.2

    for episode in range(num_episodes):

        score = 0
        obs = env.reset()
        done = False
        curr_x, curr_y = discrete_state(obs[0], obs[1])
        action = predict(curr_x, curr_y, q_table, epsilon)

        while not done:

            # Take action A
            obs, reward, done, info = env.step(action)
            score += reward

            # Get state S'
            future_x, future_y = discrete_state(obs[0], obs[1])

            # Get A'
            action_prime = predict(future_x, future_y, q_table, epsilon)

            # Update Q-Table
            curr_q = q_table[curr_x][curr_y][action]
            q_table[curr_x][curr_y][action] = curr_q + alpha*(reward + gamma*q_table[future_x][future_y][action_prime] - curr_q)

            # S = S', A = A'
            curr_x = future_x
            curr_y = future_y
            action = action_prime

        scores.append(score)
        episodes.append(episode+1)

    mean_rewards = np.zeros(num_episodes)
    for r in range(num_episodes):
        mean_rewards[r] = np.mean(scores[max(0, r-50):(r+1)])

    plt.plot(episodes, mean_rewards, label='STATIC')
    np.save('static_epsilon_mean.npy', mean_rewards)
    plt.legend()
    plt.savefig('mountain_car_sarsa_varying_epsilon')

    return 0


# Make state discrete
def discrete_state(val1, val2):

    result_x = round(val1, 1)
    result_x = result_x/0.1 + 12

    result_y = round(val2, 2)
    result_y = result_y/0.01 + 7

    return int(result_x), int(result_y)


# Predict an action using epsilon greedy
def predict(statex, statey, q_table, epsilon):
    if np.random.binomial(1, epsilon, 1) == 1:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[statex][statey])


if __name__ == '__main__':
    main()