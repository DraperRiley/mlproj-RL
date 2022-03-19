import gym
import numpy
import time
from stable_baselines3 import DQN

def main():
    ENVIRONMENT1 = 'ALE/Breakout-ram-v5'
    ENVIRONMENT2 = 'CartPole-v0'
    ENVIRONMENT3 = 'Breakout-ram-v4'

    env = gym.make(ENVIRONMENT2)

    try:
        model = DQN.load('cartPole_dqn', env)
    except:
        print("no model")

    done = False
    score = 0

    for i_episode in range(1):
        observation = env.reset()
        while not done:

            #time.sleep(0.1)
            #print(observation)
            action, _state = model.predict(observation)
            #print(_state)
            #action = env.action_space.sample()
            #print(action)
            observation, reward, done, info = env.step(action)
            score += reward
            env.render() #This method does not work with Atari games, causes errors
    env.close()
    print(score)

if __name__ == '__main__':
    main()