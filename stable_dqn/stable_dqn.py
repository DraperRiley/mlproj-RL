import gym
import numpy
import time
from stable_baselines3 import DQN

# instantiate environment
ENVIRONMENT1 = 'MountainCar-v0'
env = gym.make(ENVIRONMENT1)
env._max_episode_steps = 1000

def main():

    try:
        model = DQN.load('mountain_car_dqn')
    except:
        model = DQN('MlpPolicy', env, verbose=1, tensorboard_log='./dqn_mountain_car')

    model.learn(total_timesteps=1000000)

    done = False

    for i_episode in range(1):
        observation = env.reset()
        while not done:
            #time.sleep(0.1)
            #print(observation)
            action, _state = model.predict(observation)
            #action = env.action_space.sample()
            #print(action)
            observation, reward, done, info = env.step(action)
            env.render() #This method does not work with Atari games, causes errors
    env.close()

    return 0


if __name__ == '__main__':
    main()