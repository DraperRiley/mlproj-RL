import gym
import numpy
import time
from stable_baselines3 import DQN
from os.path import exists
import glob

def main():
    ENVIRONMENT1 = 'ALE/Breakout-ram-v5'
    ENVIRONMENT2 = 'CartPole-v0'
    ENVIRONMENT3 = 'Breakout-ram-v4'
    ENVIRONMENT4 = 'SpaceInvaders-v4'

    env = gym.make(ENVIRONMENT3)

    #if glob.glob0('breakout-dqn','breakout-dqn[0-9]*'):
    #model = DQN.load('cartPole_dqn')


    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)

    done = False

    for i_episode in range(1):
        observation = env.reset()
        while not done:
            time.sleep(0.1)
            #print(observation)
            action, _state = model.predict(observation)
            #print(_state)
            #action = env.action_space.sample()
            #print(action)
            observation, reward, done, info = env.step(action)
            env.render() #This method does not work with Atari games, causes errors
    env.close()

if __name__ == '__main__':
    main()