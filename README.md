# Reinforcement Learning

The following details the process of training a model to tackle the
mountain car problem as described in Sutton & Barto's book *Reinforcement Learning*.

<br></br>

# The Mountain Car Problem

The problem begins with a car positioned deep within a valley. The goal is to reach the top of the hill to the right, however, the car's engine is not powerful enough to drive up the hill. Therefore, the car must use the hill to its left in order to gain enough momentum to reach the rightmost hill.

<br></br>

# Environment

This project makes use of [OpenAI's Gym toolkit](https://gym.openai.com/) which provides an environment for training reinforcement learning models.

<br></br>

# Algorithms

This project uses SARSA, Q-Learning(SARSAMAX), and SARSA($\lambda$) as described in Sutton & Barto's book *Reinforcement Learning* to train models to maximize score. 

## SARSA

$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1}+ \gamma Q(S_{t+1},A_{t+1}-Q(S_t,A_t)]$