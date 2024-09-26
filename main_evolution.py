from evolution2 import Evolution

if __name__ == "__main__":

    evolution = Evolution(num_islands=4, island_size=5,
                          generations=5, inputs=4, outputs=1)

    best_individual, best_fitness = evolution.evolve()

    print(f'The best individual has the following genome:')
    best_individual.phenotype.genome.print()
    try:
        best_individual.phenotype.print()
    except:
        print('Cannot print the phenotype')
        best_individual.phenotype.print_no_position()

    print(f'The best individual has a fitness of {best_fitness}')
