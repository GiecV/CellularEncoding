from cellular_encoding.evolution import Evolution
import sys
import os
import cProfile
import pstats

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def run():
    evolution = Evolution(num_islands=4, island_size=5,
                          generations=10, inputs=4, outputs=1)

    best_individual, best_fitness = evolution.evolve()

    print(f'The best individual has the following genome:')
    best_individual.phenotype.genome.print()
    try:
        best_individual.phenotype.print()
    except:
        print('Cannot print the phenotype')
        best_individual.phenotype.print_no_position()

    print(f'The best individual has a fitness of {best_fitness}')


if __name__ == "__main__":

    run()

    # cProfile.run('run()', 'output')
    # p = pstats.Stats('output')
    # p.sort_stats('cumulative').print_stats(10)
