from core.phenotype import Phenotype
from core.neural_network_from_graph import NNFromGraph

import gym
import torch

env = gym.make('CartPole-v1')
torch.set_num_threads(1)


def compute_fitness(individual):
    """
    Compute the fitness of an individual in the CartPole environment.

    Args:
        individual: The individual whose fitness is to be computed.

    Returns:
        float: The mean reward obtained over a number of trials.
    """
    p = Phenotype(individual)
    nn = NNFromGraph(p, inputs=4, outputs=1)

    total_reward = 0
    max_steps = 200
    trials = 5

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

    Args:
        individual: The individual whose fitness is to be computed.
        penalty (float, optional): The penalty factor for the number of nodes. Default is 0.

    Returns:
        float: The penalized fitness value.
    """
    fitness = compute_fitness(individual)

    nodes = individual.get_number_of_nodes()

    return fitness - penalty * nodes  # Penalize the number of nodes
