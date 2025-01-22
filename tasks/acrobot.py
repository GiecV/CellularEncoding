from core.phenotype import Phenotype
from core.nn_cont import NNFromGraph

import gym
import torch

env = gym.make('Acrobot-v1')
torch.set_num_threads(1)

def compute_fitness(individual, n=5):
    """
    Compute the fitness of an individual in the Acrobot environment.

    The fitness is measured by the mean reward obtained over a number of trials, 
    where each trial consists of running the individual's neural network in 
    the Acrobot-v1 environment.

    :param individual: The individual whose fitness is to be computed. 
                       It is expected to be an object that can be converted 
                       into a `Phenotype` for neural network evaluation.
    :type individual: Any (typically an individual in an evolutionary algorithm)

    :return: The mean reward obtained over a number of trials.
    :rtype: float
    """
    p = Phenotype(individual)
    nn = NNFromGraph(p, inputs=6, outputs=3)

    total_reward = 0
    max_steps = 500
    trials = 3

    if nn.r == 0:
        return 0

    for _ in range(trials):
        obs = env.reset()  # Reset the environment
        obs = obs[0]
        done = False
        for _ in range(max_steps):  # Simulate the environment
            if done:
                break
            # Get the action from the individual
            action = nn.forward(torch.tensor(obs, dtype=torch.float32))
            action = torch.argmax(action).item()
            obs, reward, done, truncated, info = env.step(
                action)  # Perform the action
            total_reward += reward

    return total_reward / trials  # Mean of the reward on the trials
