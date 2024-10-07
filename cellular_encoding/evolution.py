import random
import time
import gym
import torch
import copy
from treelib import Tree
from cellular_encoding.genome import Genome
from utils.counter import GlobalCounter
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import warnings
# from tasks.xor_gate import compute_fitness
from tasks.cartpole import compute_fitness


warnings.filterwarnings("ignore")
cpus = 12
torch.set_num_interop_threads(1)


class Evolution:

    # * Initialize the evolution
    def __init__(self, population_size=400, generations=200, exchange_rate=0.01, mutation_rate=0.005, depopulation_rate=0.01):
        self.population_size = population_size
        self.generations = generations
        self.exchange_rate = exchange_rate
        self.mutation_rate = mutation_rate
        self.fitness_history = []
        self.fitness_grids = {}
        self.depopulation_rate = depopulation_rate

        self.population = self.initialize_population()
        self.fitness_scores = [None] * self.population_size

    def initialize_population(self):
        population = []

        for _ in range(self.population_size):
            population.append(self.create_individual())

        return population

    # * Create a new individual
    def create_individual(self):
        genome = Genome()  # Create a new genome
        symbols = copy.deepcopy(Genome.SYMBOLS)
        symbols.remove('n2')
        symbols.remove('n1')

        for n in range(genome.get_levels()):
            symbol = random.choice(symbols)  # Choose a random symbol for the first node
            root = genome.get_tree(n).root
            genome.change_symbol(level=n, node_id=root, symbol=symbol)

        return genome

    # * Evolve the population
    def evolve(self):

        for generation in range(self.generations):
            start_time = time.time()  # Start the simulation timer
            print(f"Generation {generation + 1}/{self.generations}")
            offspring = self.get_offspring()
            new_population = self.population + offspring
            self.population, self.fitness_scores = self.select_best(new_population)
            # self.population_size -= max(1, int(self.population_size * self.depopulation_rate))
            # print(f'Population size: {self.population_size}')
            print(f'{time.time() - start_time} s')

        return self.population[0]

    def get_offspring(self):
        offspring = []

        for parent1 in self.population:
            parent2 = random.choice(self.population)
            while parent1 == parent2:
                parent2 = random.choice(self.population)
            child = self.crossover(parent1, parent2)
            child.update_ids()
            offspring.append(child)
        for individual in offspring:
            individual = self.mutate(individual)

        return offspring

    def select_best(self, population):

        with ProcessPoolExecutor(cpus) as executor:
            fitness_list = executor.map(compute_fitness, population)
        fitness_list = list(fitness_list)
        # fitness_list = [compute_fitness(individual) for individual in population]
        individuals_and_fitness = list(zip(population, fitness_list))
        individuals_and_fitness.sort(key=lambda x: x[1], reverse=True)
        best_individuals = [individual for individual, fitness in individuals_and_fitness[:self.population_size]]
        best_fitness_scores = [fitness for individual, fitness in individuals_and_fitness[:self.population_size]]

        self.fitness_history.append(best_fitness_scores[0])
        return best_individuals, best_fitness_scores

    # * Perform crossover between two parents
    def crossover(self, parent1, parent2):
        trees = []
        g = Genome()

        for level in range(g.get_levels()):
            tree1 = Tree(tree=parent1.get_tree(level), deep=True)
            tree2 = Tree(tree=parent2.get_tree(level), deep=True)
            cutpoint1 = random.choice(list(tree1.all_nodes_itr())).identifier
            # Choose random cutpoints
            cutpoint2 = random.choice(list(tree2.all_nodes_itr())).identifier
            tree2 = tree2.subtree(cutpoint2)  # Get the bottom of the tree
            # Get the parent of the cutpoint for attaching the other subtree
            parent = tree1.parent(cutpoint1)
            if parent is not None:
                root = tree2.root
                tree2.get_node(root).parent = parent.identifier
                tree1.remove_node(cutpoint1)  # Get the top of the tree
                # Paste the two trees together
                tree1.paste(parent.identifier, tree2)
                tree = Tree(tree=tree1, deep=True)  # Save the tree
            else:
                # If the top of the tree is empty, then just copy the bottom part
                tree = Tree(tree=tree2, deep=True)
            trees.append(tree)  # Save the tree in the genome

        g = Genome(trees)

        return g

    # * Mutate a symbol of the individual
    def mutate(self, genome):
        new_genome = copy.deepcopy(genome)
        i = 0

        for tree in new_genome.get_trees():
            jumping_symbols = copy.deepcopy(Genome.JUMPING_SYMBOLS)
            if i == 1:
                jumping_symbols.remove('n2')
            elif i == 2:
                jumping_symbols.remove('n1')
                jumping_symbols.remove('n2')
            nodes = list(tree.all_nodes_itr())
            for node in nodes:
                if random.random() < self.mutation_rate:  # There is a small chance to mutate a symbol
                    if node.tag == 'e':
                        new_symbol = random.choice(
                            Genome.DIVISION_SYMBOLS + Genome.OPERATIONAL_SYMBOLS)
                        new_genome.change_symbol(
                            level=i, node_id=node.identifier, symbol=new_symbol)
                    else:
                        if node.tag in Genome.DIVISION_SYMBOLS:  # Randomly choose a division symbol
                            new_symbol = random.choice(
                                Genome.DIVISION_SYMBOLS)
                        else:  # Randomly choose an unary symbol
                            new_symbol = random.choice(
                                # jumping_symbols
                                Genome.OPERATIONAL_SYMBOLS
                            )
                        tree.update_node(nid=node.identifier, tag=new_symbol)
            i += 1

        return new_genome

    # * Plot the best fitness in every generation
    def plot_fitness_history(self):
        generations = list(range(1, len(self.fitness_history) + 1))
        plt.plot(generations, self.fitness_history, marker='o')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness History Over Generations')
        plt.grid(True)
        plt.show()
