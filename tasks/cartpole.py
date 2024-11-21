from core.phenotype import Phenotype
from core.neural_network_from_graph import NNFromGraph

import gym
import torch

env = gym.make('CartPole-v1')
torch.set_num_threads(1)


def compute_fitness(individual):
    """
    Compute the fitness of an individual in the CartPole environment.

    The fitness is measured by the mean reward obtained over a number of trials, 
    where each trial consists of running the individual's neural network in 
    the CartPole-v1 environment.

    :param individual: The individual whose fitness is to be computed. 
                       It is expected to be an object that can be converted 
                       into a `Phenotype` for neural network evaluation.
    :type individual: Any (typically an individual in an evolutionary algorithm)

    :return: The mean reward obtained over a number of trials.
    :rtype: float
    """
    p = Phenotype(individual)
    nn = NNFromGraph(p, inputs=4, outputs=1)

    total_reward = 0
    max_steps = 200
    trials = 5

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
            action = nn.forward(torch.tensor(obs, dtype=torch.float32)).item()
            obs, reward, done, truncated, info = env.step(
                action)  # Perform the action
            total_reward += reward

    return total_reward / trials  # Mean of the reward on the trials


def compute_fitness_growth_penalty(individual, penalty=0):
    """
    Compute the fitness of an individual with a growth penalty.

    This function evaluates the fitness of an individual by first calculating 
    its original fitness using `compute_fitness`. Then, it applies a penalty 
    based on the number of nodes in the individual's genotype. The penalty 
    helps to discourage overly complex solutions.

    :param individual: The individual whose fitness is to be computed.
    :type individual: Any (typically an individual in an evolutionary algorithm)
    :param penalty: The penalty factor for the number of nodes. Default is 0, 
                    meaning no penalty is applied.
    :type penalty: float, optional

    :return: The penalized fitness value.
    :rtype: float
    """
    fitness = compute_fitness(individual)

    nodes = individual.get_number_of_nodes()

    return fitness - penalty * nodes  # Penalize the number of nodes
