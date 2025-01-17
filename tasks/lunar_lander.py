import torch
from core.phenotype_cont import Phenotype
from core.nn_cont import NNFromGraph
import gym

torch.set_num_threads(1)

max_timesteps = 500
episodes = 3


def compute_fitness(individual, n=2):
    """
    Evaluate the fitness of an individual in the Lunar Lander task.

    :param individual: The individual whose fitness is to be evaluated.
                       Expected to be a graph convertible to a Phenotype.
    :return: The fitness score, as the total reward.
    """
    # Create the Phenotype and neural network from the individual
    p = Phenotype(individual)
    # Adjusted inputs and outputs for the lunar lander
    nn = NNFromGraph(p, inputs=8, outputs=4)

    if nn.r == 0:  # Check if the neural network is functional
        # print(f'r is 0')
        return -1000

    # Initialize the Lunar Lander environment
    env = gym.make('LunarLander-v2')

    total_reward = 0

    for episode in range(episodes):
        obs = env.reset()
        obs = obs[0]
        for t in range(max_timesteps):
            # Get the neural network output
            action = nn.forward(torch.tensor(
                obs, dtype=torch.float32)).detach().numpy()

            action = action.argmax()

            # Step the environment with the neural network output
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

    env.close()
    return total_reward/episodes  # Average reward per episode
