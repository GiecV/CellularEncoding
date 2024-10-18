import torch
import itertools

from core.phenotype import Phenotype
from core.neural_network_from_graph import NNFromGraph
from sklearn.metrics import normalized_mutual_info_score


def compute_fitness(individual, n=2):

    p = Phenotype(individual)
    nn = NNFromGraph(p, inputs=n, outputs=1)

    if nn.r == 0:
        return 0

    outputs = []
    targets = []

    # Generate all possible combinations of n binary inputs
    for combination in itertools.product([0, 1], repeat=n):
        input_data = torch.tensor(combination, dtype=torch.float32)

        # Get the output from the neural network
        output = nn(input_data)
        outputs.append(output.item())

        # Compute the expected parity of the input combination
        expected_parity = sum(combination) % 2
        targets.append(expected_parity)

    return normalized_mutual_info_score(outputs, targets)
