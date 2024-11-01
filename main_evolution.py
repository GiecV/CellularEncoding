from core.evolution import Evolution
from utils.visualizer import Visualizer
import sys
import os


# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def run():

    os.system('clear')
    inputs = 3
    evolution = Evolution(population_size=1000, generations=300, mutation_rate=0.05, inputs=inputs,
                          info=True)

    best_individual = evolution.evolve()
    os.system('clear')

    visualizer = Visualizer(inputs=inputs, outputs=1)
    visualizer.plot_fitness_history(evolution.fitness_history)
    visualizer.print_innovative_networks(evolution.innovative_individuals, save=False)


if __name__ == "__main__":

    run()
