from sklearn.metrics import mutual_info_score
from cellular_encoding.genome import Genome
import torch


def target_function(x, y): return x ^ y


def compute_fitness_information(individual):

    inputs = [[x, y] for x in range(2) for y in range(2)]
    targets = [target_function(x, y) for x, y in inputs]
    outputs = [individual.forward(torch.tensor(
        [x, y]).float()).item() for x, y in inputs]

    outputs = [1 if x > 0.5 else 0 for x in outputs]

    mutual_info = mutual_info_score(targets, outputs)
    target_info = mutual_info_score(targets, targets)

    fitness = 0.85 * mutual_info / target_info + 0.15 * individual.t
    # fitness = individual.t
    return fitness


def compute_each_input_combination(individual):

    inputs = [[x, y] for x in range(2) for y in range(2)]
    outputs = [individual.forward(torch.tensor(
        [x, y]).float()).item() for x, y in inputs]

    for i, input in enumerate(inputs):
        print(f'f({input})={outputs[i]}')


def compute_fitness_target(individual, print_info=False):

    target = Genome()
    root = target.get_tree(0).root
    target.change_symbol(0, root + 0, 's')
    target.change_symbol(0, root + 1, 'p')
    target.change_symbol(0, root + 2, 'w')
    target.change_symbol(0, root + 3, 't')
    target.change_symbol(0, root + 5, '-')
    target_nodes = [node.tag for node in target.get_tree(0).all_nodes_itr()]

    common_tags = 0
    common_successors = 0
    individual_tree = individual.get_tree(0)
    target_tree = target.get_tree(0)

    if individual is None:
        individual = target

    depth1 = individual_tree.depth()
    depth2 = target_tree.depth()

    n_nodes1 = len(list(individual_tree.all_nodes_itr()))
    n_nodes2 = len(list(target_tree.all_nodes_itr()))

    tags1 = [node.tag for node in individual_tree.all_nodes()]
    tags2 = [node.tag for node in target_tree.all_nodes()]
    tag_difference = abs(len(tags1) - len(tags2))
    target_tags = len(tags2)
    for tag in tags1:
        if tag in tags2:
            tags2.remove(tag)
            common_tags += 1

    successors1 = []
    for node in individual_tree.all_nodes_itr():
        successors = node.successors(individual_tree.identifier)
        successors_of_node = []
        for successor in successors:
            successors_of_node.append(individual_tree.get_node(successor).tag)
        successors1.append(successors_of_node)
    successors2 = []
    for node in target_tree.all_nodes_itr():
        successors = node.successors(target_tree.identifier)
        successors_of_node = []
        for successor in successors:
            successors_of_node.append(target_tree.get_node(successor).tag)
        successors2.append(successors_of_node)

    successors_difference = abs(len(successors1) - len(successors2))
    target_successors = len(successors2)

    for successor in successors1:
        if successor in successors2:
            successors2.remove(successor)
            common_successors += 1

    depth_similarity = 1 - abs(depth1 - depth2) / max(depth1, depth2)
    nodes_similarity = 1 - abs(n_nodes1 - n_nodes2) / max(n_nodes1, n_nodes2)
    tags_similarity = (common_tags - tag_difference) / target_tags
    successors_similarity = (common_successors - successors_difference) / target_successors
    fitness = (depth_similarity + nodes_similarity + tags_similarity + successors_similarity) / 4

    if print_info:
        print(f"Depth similarity: {depth_similarity}")
        print(f"Nodes similarity: {nodes_similarity}")
        print(f"Tags similarity: {tags_similarity}")
        print(f"Successors similarity: {successors_similarity}")
        print(f"Fitness: {fitness}")

    return fitness
