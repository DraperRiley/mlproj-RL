import numpy as np
import matplotlib.pyplot as plt

def main():

    episodes = []
    for i in range(10000):
        episodes.append(i+1)

    mean_rewards = np.load('./mountain_car_sarsa_dir/mountain_car_sarsa/mountain_car_sarsa_10000_episodes_mean_rewards.npy')

    plt.plot(episodes, mean_rewards, label='SARSA')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.title('Total reward over ' + str(len(episodes)) + ' episodes')

    mean_rewards = np.load('./mountain_car_sarsalambda_dir/mountain_car_sarsalambda/mountain_car_sarsalambda_10000_epsisodes_mean_rewards.npy')

    plt.plot(episodes, mean_rewards, label='SARSALAMBDA')

    mean_rewards = np.load('./mountain_car_sarsamax_dir/mountain_car_sarsamax/mountain_car_sarsamax_10000_episodes_mean_rewards.npy')

    plt.plot(episodes, mean_rewards, label='SARSAMAX')

    plt.legend()
    plt.savefig('learning_results')

    return 0

if __name__ == '__main__':
    main()