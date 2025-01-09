import torch
from core.phenotype import Phenotype
from core.sig_nn import NNFromGraph
import gym
import numpy as np

torch.set_num_threads(1)

num_episodes = 5
max_timesteps = 1000


def compute_fitness(individual, n=2):
    """
    Evaluate the fitness of an individual in the walking task.

    :param individual: The individual whose fitness is to be evaluated.
                       Expected to be a graph convertible to a Phenotype.
    :return: The fitness score, as the distance traveled.
    """
    # Create the Phenotype and neural network from the individual
    p = Phenotype(individual)
    # Adjusted inputs and outputs for the ant
    nn = NNFromGraph(p, inputs=27, outputs=8)

    if nn.r == 0:  # Check if the neural network is functional
        print(f'r is 0')
        return 0

    # Initialize the Ant environment
    env = gym.make('Ant-v4')

    total_distance = 0

    for episode in range(num_episodes):
        obs = env.reset()
        obs = obs[0]
        for t in range(max_timesteps):
            print(f'Episode {episode}: {t}')
            # Get the neural network output
            action = nn.forward(torch.tensor(
                obs, dtype=torch.float32)).detach().numpy()

            # Step the environment with the neural network output
            obs, reward, terminated, truncated, info = env.step(action)

            # Calculate the distance traveled
            position = env.env.data.qpos[0]
            total_distance += position

            if terminated or truncated:
                break

    env.close()
    return total_distance / num_episodes  # Average distance traveled per episode
