import gym
import numpy
import time
from stable_baselines3 import DQN

# instantiate environment
ENVIRONMENT1 = 'MountainCar-v0'
env = gym.make(ENVIRONMENT1)
env._max_episode_steps = 1000

def main():
    ENVIRONMENT1 = 'ALE/Breakout-ram-v5'
    ENVIRONMENT2 = 'CartPole-v0'
    ENVIRONMENT3 = 'Breakout-ram-v4'

    #env = gym.make(ENVIRONMENT2)

    try:
        model = DQN.load('cartPole_dqn')
        model.set_env(env)
    except:
        print("No model found")
        model = DQN('MlpPolicy', env, verbose=1)

    model.learn(total_timesteps=100000)
    model.save('cartPole_dqn')

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

if __name__ == '__main__':
    main()