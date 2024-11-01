import json
from treelib import Tree
from core.genome import Genome
from utils.visualizer import Visualizer


def get_lineage(file_name, gens_to_save=5):

    print('a')

    def traverse_generations(data, generation_idx, individual_idx, genomes):

        individual = data[generation_idx]['individuals'][individual_idx]
        genomes.append({'generation': generation_idx,
                        'genome': individual['genome']})

        parents = individual['parents']
        if parents is not None and generation_idx > generations - gens_to_save:
            traverse_generations(
                data, generation_idx - 1, parents[0], genomes)
            traverse_generations(
                data, generation_idx - 1, parents[1], genomes)

    print('b')

    with open(file_name, 'r') as file:
        data = json.load(file)
    data = data[0]
    log = data['log']

    genomes = []
    generations = len(log)
    traverse_generations(log, generations - 1, 0, genomes)

    return genomes


if __name__ == "__main__":

    genomes = get_lineage('logs/3iprovapickle.json', 4)
    json.dump(genomes, open('logs/prova.json', 'w'), indent=4)

    with open('logs/prova.json', 'r') as file:
        genomes_json = json.load(file)

    # for entry in genomes_json:
    #     genome = Genome()
    #     genome.from_json_pickle(entry)
    #     genome.print()
    visualizer = Visualizer()
    visualizer.print_lineage(genomes_json, save=True)
