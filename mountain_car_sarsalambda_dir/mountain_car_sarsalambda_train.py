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
    num_episodes = 10000
    decay = 0.99
    min_epsilon = 0.01
    decreasing_decay = 0.001
    epsilon_start = 0.9
    lambda_val = 0.6

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
        curr_x, curr_y = discrete_state(obs[0], obs[1])
        action = predict(curr_x, curr_y, q_table, epsilon)

        # Initialize e
        e = np.zeros((18, 14, 3))

        while not done:

            # Take action A
            obs, reward, done, info = env.step(action)
            score += reward

            # Get state S'
            future_x, future_y = discrete_state(obs[0], obs[1])

            # Get A'
            action_prime = predict(future_x, future_y, q_table, epsilon)

            # Calculate delta
            delta = reward + gamma*q_table[future_x][future_y][action_prime] - q_table[curr_x][curr_y][action]

            # Update e table
            e[curr_x][curr_y][action] += 1

            alpha_delta = alpha * delta
            gamma_lambda = gamma * lambda_val

            for idx in np.ndindex(q_table.shape):
                q_table[idx[0]][idx[1]][idx[2]] += alpha_delta * e[idx[0]][idx[1]][idx[2]]
                e[idx[0]][idx[1]][idx[2]] = gamma_lambda * e[idx[0]][idx[1]][idx[2]]

            # S = S', A = A'
            curr_x = future_x
            curr_y = future_y
            action = action_prime


        scores.append(score)
        episodes.append(episode+1)

        if (episode + 1) % 20 == 0:
            epsilon *= decay

        #epsilon = epsilon - 1/num_episodes

        if (episode + 1) % 20 == 0:
            epsilon *= decay

        if (episode + 1) % 100 == 0:
            np.save('./mountain_car_sarsalambda/mountain_car_sarsalambda_{f}_episodes.npy'.format(f=str(len(episodes))), q_table)
            num_eps = episode + 1

    mean_rewards = np.zeros(num_episodes)
    for r in range(num_episodes):
        mean_rewards[r] = np.mean(scores[max(0, r-50):(r+1)])

    plt.plot(episodes, mean_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.title('Total reward over ' + str(len(episodes)) + ' episodes')
    #plt.show()
    plt.savefig('mountaincar_reward_sarsalambda_{f}'.format(f=num_eps))

    np.save('./mountain_car_sarsalambda/mountain_car_sarsalambda_{f}_epsisodes_mean_rewards'.format(f=str(num_episodes)), mean_rewards)

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