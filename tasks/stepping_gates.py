import torch
import itertools
from core.phenotype import Phenotype
from core.neural_network_from_graph import NNFromGraph

torch.set_num_threads(1)


def compute_fitness(individual, max_gate=3, print_output=False):
    """
    Evaluate the fitness of an individual in the stepping gates task.

    :param individual: The individual whose fitness is to be evaluated.
                       Expected to be a graph convertible to a Phenotype.
    :param max_gate: The maximum gate to evaluate (1-indexed). Default is 3.
    :return: The fitness score, as the proportion of correct outputs.
    """
    # Create the Phenotype and neural network from the individual
    p = Phenotype(individual)
    nn = NNFromGraph(p, inputs=8, outputs=1)

    if nn.r == 0:  # Check if the neural network is functional
        return 0

    # Define the gate functions
    def ex(inputs):
        return int(any(inputs))

    def multiplexer(inputs):
        return inputs[1] if inputs[0] else inputs[2]

    def nand_gate(inputs):
        return int(not (inputs[0] and inputs[1]))

    def not_gate(inputs):
        return int(not inputs[0])

    def and_gate(inputs):
        return int(inputs[0] and inputs[1])

    def or_gate(inputs):
        return int(inputs[0] or inputs[1])

    def xor_gate(inputs):
        return int(inputs[0] != inputs[1])

    # gates = [copy_1, copy_2, nand_gate, not_gate, and_gate, or_gate, xor_gate]
    gates = [(multiplexer, 3), (nand_gate, 2), (not_gate, 2),
             (and_gate, 2), (or_gate, 2), (xor_gate, 2)]
    if max_gate > len(gates):
        raise ValueError("max_gate exceeds the number of available gates")

    # Generate control bits and evaluate each gate
    control_bits = [list(format(i, '04b')) for i in range(max_gate)]
    control_bits = [[int(bit) for bit in cb] for cb in control_bits]

    total_tests = 0
    correct_outputs = 0

    for control, gate in zip(control_bits, gates[:max_gate]):
        gate_function, input_size = gate
        # Generate all combinations of 4 input bits
        for inputs in itertools.product([0, 1], repeat=input_size):
            inputs = inputs + (-1,) * (4 - input_size)\
                # Full input to the neural network: control bits + inputs
            full_input = torch.tensor(
                control + list(inputs), dtype=torch.float32)

            # Get the network's output
            output = nn(full_input).item()
            predicted_output = round(output)  # Assuming binary output

            # Compute the expected output from the gate function
            expected_output = gate_function(inputs)

            # Update the score
            if predicted_output == expected_output:
                correct_outputs += 1
            total_tests += 1

            if print_output:
                print(f'Control: {control} Input: {inputs}')
                print(f'Expected: {expected_output}' +
                      f'Predicted: {predicted_output}')

    # Compute the fitness as the proportion of correct outputs
    fitness = correct_outputs / total_tests
    return fitness


def compute_fitness_just_n(individual, gate_index=1, print_output=False):
    """
    Evaluate the fitness of an individual in the stepping gates task.

    :param individual: The individual whose fitness is to be evaluated.
                       Expected to be a graph convertible to a Phenotype.
    :param gate_index: The index of the gate to evaluate (0-indexed).
    :return: The fitness score, as the proportion of correct outputs.
    """
    # Create the Phenotype and neural network from the individual
    p = Phenotype(individual)
    nn = NNFromGraph(p, inputs=8, outputs=1)

    if nn.r == 0:  # Check if the neural network is functional
        return 0

    gate_index -= 1  # Convert to 0-indexed

    # Define the gate functions
    def ex(inputs):
        return int(any(inputs))

    def multiplexer(inputs):
        return inputs[1] if inputs[0] else inputs[2]

    def nand_gate(inputs):
        return int(not (inputs[0] and inputs[1]))

    def not_gate(inputs):
        return int(not inputs[0])

    def and_gate(inputs):
        return int(inputs[0] and inputs[1])

    def or_gate(inputs):
        return int(inputs[0] or inputs[1])

    def xor_gate(inputs):
        return int(inputs[0] != inputs[1])
    
    def xax_gate(inputs):
        return int(inputs[0] and inputs[1] and inputs[2])
    
    def incr_gate(inputs):
        return 0

    # gates = [(multiplexer, 3), (nand_gate, 2), (not_gate, 2),
    #          (and_gate, 2), (or_gate, 2), (xor_gate, 2)]
    gates = [(nand_gate, 2), (not_gate, 2), (and_gate, 2), (xor_gate, 2),(xax_gate, 3), (incr_gate, 4), (decr_gate, 4), (shiftleft, 4)]
    if gate_index >= len(gates):
        raise ValueError("gate_index exceeds the number of available gates")

    # Generate control bits for the specified gate
    control_bits = [list(format(gate_index, '04b'))]
    control_bits = [[int(bit) for bit in cb] for cb in control_bits]

    total_tests = 0
    correct_outputs = 0

    gate_function, input_size = gates[gate_index]
    # Generate all combinations of input bits for the specified gate
    for inputs in itertools.product([0, 1], repeat=input_size):
        inputs = inputs + (-1,) * (4 - input_size)
        # Full input to the neural network: control bits + inputs
        full_input = torch.tensor(
            control_bits[0] + list(inputs), dtype=torch.float32)

        # Get the network's output
        output = nn(full_input).item()
        predicted_output = round(output)  # Assuming binary output

        # Compute the expected output from the gate function
        expected_output = gate_function(inputs)

        # Update the score
        if predicted_output == expected_output:
            correct_outputs += 1
        total_tests += 1

        if print_output:
            print(f'Control: {control_bits[0]} Input: {inputs}')
            print(f'Expected: {expected_output} Predicted: {predicted_output}')

    # Compute the fitness as the proportion of correct outputs
    fitness = correct_outputs / total_tests
    return fitness
