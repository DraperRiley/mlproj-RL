import gym
import numpy
import time
from stable_baselines3 import DQN
from os.path import exists
import glob

def main():
    ENVIRONMENT1 = 'ALE/Breakout-ram-v5'
    ENVIRONMENT2 = 'CartPole-v0'
    ENVIRONMENT3 = 'Breakout-v4'
    ENVIRONMENT4 = 'SpaceInvaders-v4'

    env = gym.make(ENVIRONMENT3)

    #if glob.glob0('breakout-dqn','breakout-dqn[0-9]*'):
    #model = DQN.load('cartPole_dqn')


    try:
        model = DQN.load('spaceinvaders-dqn/spaceinvaders-dqn-100000')
        model.set_env(env)
    except:
        print("no model")
        model = DQN('MlpPolicy', env, verbose=1)


    for i in range(10):
        model.learn(total_timesteps=100000)
        timestep = (i+1) * 100000
        model.save('breakout-dqn/breakout-dqn-step' + str(timestep))

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