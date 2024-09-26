from evolution2 import Evolution

if __name__ == "__main__":
    evolution = Evolution(num_islands=4, island_size=5,
                          generations=100, inputs=2, outputs=1)

    best_individual, best_fitness = evolution.evolve()

    # print(f'These are all the final individuals:')
    evolution.display_individuals()

    print(f'The best individual has the following genome:')
    best_individual.phenotype.genome.print()
    try:
        best_individual.phenotype.print()
    except:
        print('Cannot print the phenotype')
        best_individual.phenotype.print_no_position()
    print(f'The best individual has a fitness of {best_fitness}')
    evolution.compute_each_input_combination(best_individual)

    # nn1 = evolution.create_individual()
    # nn2 = evolution.create_individual()

    # nn1.phenotype.genome.print()
    # nn2.phenotype.genome.print()

    # offspring = evolution.crossover(nn1, nn2)
    # evolution.mutate(offspring)

    # offspring.phenotype.genome.print()
