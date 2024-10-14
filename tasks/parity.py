import torch
import itertools

from core.phenotype import Phenotype
from core.neural_network_from_graph import NNFromGraph


def compute_fitness(individual, n=4):

    p = Phenotype(individual)
    nn = NNFromGraph(p)

    correct_attempts = 0
    total_attempts = 2**n

    # Generate all possible combinations of n binary inputs
    for combination in itertools.product([0, 1], repeat=n):
        input_data = torch.tensor(combination, dtype=torch.float32)

        # Get the output from the neural network
        output = nn(input_data)

        # Compute the expected parity of the input combination
        expected_parity = sum(combination) % 2

        # Compare the network's output parity with the expected parity
        if output == expected_parity:
            correct_attempts += 1

    return correct_attempts / total_attempts
