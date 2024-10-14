from core.evolution import Evolution
from core.phenotype import Phenotype
from core.neural_network_from_graph import NNFromGraph
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def run():

    os.system('clear')
    evolution = Evolution(population_size=400, generations=100, mutation_rate=0.05)
    best_individual = evolution.evolve()
    os.system('clear')

    evolution.plot_fitness_history()

    p = Phenotype(best_individual)
    nn = NNFromGraph(p, inputs=4, outputs=1)
    nn.phenotype.print()

    # for individual, fitness in evolution.innovative_individuals:
    #     print(f'Fitness: {fitness}')
    #     p = Phenotype(individual)
    #     nn = NNFromGraph(p, inputs=4, outputs=1)
    #     nn.phenotype.print()


if __name__ == "__main__":

    run()
