from cellular_encoding.evolution import Evolution
from cellular_encoding.phenotype import Phenotype
from cellular_encoding.neural_network_from_graph import NNFromGraph
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def run():

    os.system('clear')
    evolution = Evolution(population_size=100, generations=30, mutation_rate=0.05, depopulation_rate=0.01)
    best_individual = evolution.evolve()
    os.system('clear')

    evolution.plot_fitness_history()

    p = Phenotype(best_individual)
    nn = NNFromGraph(p, inputs=4, outputs=1)
    nn.phenotype.print()

    # print(nn.forward(torch.tensor([0, 0]).float()))
    # print(nn.forward(torch.tensor([0, 1]).float()))
    # print(nn.forward(torch.tensor([1, 0]).float()))
    # print(nn.forward(torch.tensor([1, 1]).float()))

    # print(nn.forward(torch.tensor([0, 0, 0, 0]).float()))
    # print(nn.forward(torch.tensor([0, 0, 0, 1]).float()))
    # print(nn.forward(torch.tensor([0, 0, 1, 0]).float()))
    # print(nn.forward(torch.tensor([0, 0, 1, 1]).float()))
    # print(nn.forward(torch.tensor([0, 1, 0, 0]).float()))
    # print(nn.forward(torch.tensor([0, 1, 0, 1]).float()))
    # print(nn.forward(torch.tensor([0, 1, 1, 0]).float()))
    # print(nn.forward(torch.tensor([0, 1, 1, 1]).float()))
    # print(nn.forward(torch.tensor([1, 0, 0, 0]).float()))
    # print(nn.forward(torch.tensor([1, 0, 0, 1]).float()))
    # print(nn.forward(torch.tensor([1, 0, 1, 0]).float()))
    # print(nn.forward(torch.tensor([1, 0, 1, 1]).float()))
    # print(nn.forward(torch.tensor([1, 1, 0, 0]).float()))
    # print(nn.forward(torch.tensor([1, 1, 0, 1]).float()))
    # print(nn.forward(torch.tensor([1, 1, 1, 0]).float()))
    # print(nn.forward(torch.tensor([1, 1, 1, 1]).float()))


if __name__ == "__main__":

    run()
