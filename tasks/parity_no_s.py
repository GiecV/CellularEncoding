import torch
import itertools

from core.phenotype_no_s import Phenotype
from core.neural_network_from_graph import NNFromGraph
from sklearn.metrics import normalized_mutual_info_score

torch.set_num_threads(1)


def compute_fitness(individual, n=2):
    """
    Compute the fitness of an individual in the parity problem.

    The fitness is evaluated by computing the normalized mutual information score 
    between the outputs of a neural network and the expected parity targets 
    for all possible binary input combinations. The network is evaluated with 
    `n` binary inputs, and the fitness is a measure of how well the network 
    predicts the parity of the inputs.

    :param individual: The individual whose fitness is to be computed. 
                       It is expected to be an object that can be converted 
                       into a `Phenotype` for neural network evaluation.
    :type individual: Any (typically an individual in an evolutionary algorithm)
    :param n: The number of binary inputs to the neural network. Default is 2.
    :type n: int, optional

    :return: The normalized mutual information score between the network's 
             outputs and the expected targets.
    :rtype: float
    """
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


def compute_fitness_up_to_n(individual, n=2):
    """
    Compute the fitness of an individual in the parity problem.

    The fitness is evaluated by computing the normalized mutual information score 
    between the outputs of a neural network and the expected parity targets 
    for all possible binary input combinations. The network is evaluated with 
    `n` binary inputs, and the fitness is a measure of how well the network 
    predicts the parity of the inputs.

    :param individual: The individual whose fitness is to be computed. 
                       It is expected to be an object that can be converted 
                       into a `Phenotype` for neural network evaluation.
    :type individual: Any (typically an individual in an evolutionary algorithm)
    :param n: The number of binary inputs to the neural network. Default is 2.
    :type n: int, optional

    :return: The normalized mutual information score between the network's 
             outputs and the expected targets.
    :rtype: float
    """

    outputs = []
    targets = []

    for inputs in range(2, n+1):

        p = Phenotype(individual)
        nn = NNFromGraph(p, inputs=inputs, outputs=1)

        if nn.r == 0:
            return 0

        # Generate all possible combinations of n binary inputs
        for combination in itertools.product([0, 1], repeat=inputs):
            input_data = torch.tensor(combination, dtype=torch.float32)

            # Get the output from the neural network
            output = nn(input_data)
            outputs.append(output.item())

            # Compute the expected parity of the input combination
            expected_parity = sum(combination) % 2
            targets.append(expected_parity)

    return normalized_mutual_info_score(outputs, targets)
