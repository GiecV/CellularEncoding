from cellular_encoding.evolution import Evolution
from cellular_encoding.phenotype import Phenotype
from cellular_encoding.neural_network_from_graph import NNFromGraph
import sys
import os
import cProfile
import pstats
import copy
import torch

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def run():

    os.system('clear')
    evolution = Evolution(population_size=400, generations=30, mutation_rate=0.05)
    best_individual = evolution.evolve()
    os.system('clear')

    p = Phenotype(best_individual)
    nn = NNFromGraph(p)
    print(nn.forward(torch.tensor([0, 0]).float()))
    print(nn.forward(torch.tensor([0, 1]).float()))
    print(nn.forward(torch.tensor([1, 0]).float()))
    print(nn.forward(torch.tensor([1, 1]).float()))


if __name__ == "__main__":

    run()

    # cProfile.run('run()', 'output')
    # p = pstats.Stats('output')
    # p.sort_stats('cumulative').print_stats(10)
