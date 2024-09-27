from cellular_encoding.evolution2 import Evolution
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":

    evolution = Evolution(num_islands=4, island_size=10,
                          generations=3, inputs=4, outputs=1)

    best_individual, best_fitness = evolution.evolve()

    print(f'The best individual has the following genome:')
    best_individual.phenotype.genome.print()
    try:
        best_individual.phenotype.print()
    except:
        print('Cannot print the phenotype')
        best_individual.phenotype.print_no_position()

    print(f'The best individual has a fitness of {best_fitness}')
