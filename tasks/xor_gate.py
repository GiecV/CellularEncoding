from sklearn.metrics import mutual_info_score
import torch


def compute_fitness_information(individual):
    def target_function(x, y): return x ^ y

    inputs = [[x, y] for x in range(2) for y in range(2)]
    targets = [target_function(x, y) for x, y in inputs]
    outputs = [individual.forward(torch.tensor(
        [x, y]).float()).item() for x, y in inputs]

    outputs = [1 if x > 0.5 else 0 for x in outputs]

    mutual_info = mutual_info_score(targets, outputs)
    target_info = mutual_info_score(targets, targets)

    fitness = 0.85 * mutual_info / target_info + 0.15 * individual.t
    return fitness


def compute_each_input_combination(individual):

    inputs = [[x, y] for x in range(2) for y in range(2)]
    outputs = [individual.forward(torch.tensor(
        [x, y]).float()).item() for x, y in inputs]

    for i, input in enumerate(inputs):
        print(f'f({input})={outputs[i]}')
