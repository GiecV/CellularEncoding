import random
import time
from treelib import Tree
from core.genome import Genome
from concurrent.futures import ProcessPoolExecutor
import warnings
from tasks.parity import compute_fitness

warnings.filterwarnings("ignore")
cpus = 128


class Evolution:

    def __init__(self, population_size=1000, generations=300, mutation_rate=0.05, inputs=2, population=None):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.fitness_history = []
        self.innovative_individuals = []
        self.fitness_grids = {}
        self.inputs = inputs
        self.fitness_scores = [None] * self.population_size
        self.logs = []

        self.population = population or self.initialize_population()

    # * Create the initial population
    def initialize_population(self):
        return [self.create_individual() for _ in range(self.population_size)]

    # * Create an individual with random initial tag
    def create_individual(self):
        genome = Genome()
        symbols = Genome.SYMBOLS
        for n in range(genome.get_levels()):
            symbol = random.choice(symbols)
            root = genome.get_tree(n).root
            genome.change_symbol(level=n, node_id=root, symbol=symbol)

        return genome

    # * Evolve the population controlling generations. Return the best individual
    def evolve(self, info=True, max_time=2100):

        best_score = float('-inf')
        start_time = time.time()

        for generation in range(self.generations):
            generation_start_time = time.time()
            print(f"Generation {generation + 1}/{self.generations}")
            offspring = self.get_offspring()
            new_population = self.population + offspring
            self.population, self.fitness_scores = self.select_best(
                new_population)
            if self.fitness_scores[0] > best_score:
                self.innovative_individuals.append(
                    (self.population[0], self.fitness_scores[0]))
                best_score = self.fitness_scores[0]

            self.generation_time = time.time() - generation_start_time

            if info:
                self.logs.append({
                    'best_score': best_score,
                    'generation_time': self.generation_time,
                    'individuals': [individual.json_pickle() for individual in self.population],
                    'fitness_scores': self.fitness_scores
                })

            print(f'{self.generation_time} s')

            # if self.fitness_scores[0] == 1:
            #     break
            print('Elapsed time:', (time.time() - start_time) / 60, 'min')
            if time.time() - start_time > max_time:
                break

        return self.population[0], generation + 1

    # * Make each individual reproduce with a random partner
    def get_offspring(self):
        offspring = []
        for parent1 in self.population:
            parent2 = random.choice(self.population)
            index1 = self.population.index(parent1)
            while parent1 == parent2:
                parent2 = random.choice(self.population)
            index2 = self.population.index(parent2)
            child1, child2 = self.crossover(parent1, parent2, [index1, index2])
            child1.update_ids()
            child2.update_ids()
            offspring.extend((child1, child2))
        return [self.mutate(individual) for individual in offspring]

    # * Select the best `self.population_size` individuals, discard the others
    def select_best(self, population):
        ns = [self.inputs for _ in range(len(population))]
        with ProcessPoolExecutor(cpus) as executor:
            fitness_list = list(executor.map(compute_fitness, population, ns))
        individuals_and_fitness = sorted(
            zip(population, fitness_list), key=lambda x: x[1], reverse=True)
        best_individuals = [individual for individual,
                            _ in individuals_and_fitness[:self.population_size]]
        best_fitness_scores = [
            fitness for _, fitness in individuals_and_fitness[:self.population_size]]

        self.fitness_history.append(best_fitness_scores[0])
        return best_individuals, best_fitness_scores

    def edit_population_size(self, acceptable_time=30):
        if self.generation_time > acceptable_time:
            self.population_size = max(
                self.min_population_size, int(self.population_size * .9))
        elif self.generation_time < acceptable_time:
            self.population_size = min(
                self.max_population_size, int(self.population_size * 1.1))

    # * Perform crossover between two parents
    def crossover(self, parent1, parent2, parents_indexes):
        child1, child2 = self.get_children(parent1, parent2, parents_indexes)

        return child1, child2

    # * Create two children starting from two parents
    def get_children(self, parent1, parent2, parents_indexes):
        trees = []
        trees2 = []
        g = Genome()

        for level in range(g.get_levels()):
            tree1 = Tree(tree=parent1.get_tree(level), deep=True)
            tree2 = Tree(tree=parent2.get_tree(level), deep=True)
            cutpoint1 = random.choice(list(tree1.all_nodes_itr())).identifier
            cutpoint2 = random.choice(list(tree2.all_nodes_itr())).identifier
            tree2 = tree2.subtree(cutpoint2)
            parent = tree1.parent(cutpoint1)
            if parent is not None:
                root = tree2.root
                tree2.get_node(root).parent = parent.identifier  # type: ignore
                tree1.remove_node(cutpoint1)
                tree1.paste(parent.identifier, tree2)
                tree = Tree(tree=tree1, deep=True)
            else:
                tree = Tree(tree=tree2, deep=True)
            trees.append(tree)

            tree1 = Tree(tree=parent1.get_tree(level), deep=True)
            tree2 = Tree(tree=parent2.get_tree(level), deep=True)
            subtree1 = tree1.subtree(cutpoint1)
            parent2_node = tree2.parent(cutpoint2)
            if parent2_node is not None:
                root = subtree1.root
                subtree1.get_node(
                    root).parent = parent2_node.identifier  # type: ignore
                tree2.remove_node(cutpoint2)
                tree2.paste(parent2_node.identifier, subtree1)
                new_tree2 = Tree(tree=tree2, deep=True)
            else:
                new_tree2 = Tree(tree=subtree1, deep=True)
            trees2.append(new_tree2)

        return Genome(trees, parents_indexes), Genome(trees2, parents_indexes)

    # * Mutate symbols of the individual
    def mutate(self, genome):
        for i, tree in enumerate(genome.get_trees()):
            nodes = list(tree.all_nodes_itr())
            for node in nodes:
                if random.random() < self.mutation_rate:
                    if node.tag in ['e', 'n']:
                        new_symbol = random.choice(Genome.SYMBOLS)
                        genome.change_symbol(
                            level=i, node_id=node.identifier, symbol=new_symbol)
                    else:
                        new_symbol = (
                            random.choice(Genome.DIVISION_SYMBOLS)
                            if node.tag in Genome.DIVISION_SYMBOLS
                            else random.choice(Genome.OPERATIONAL_SYMBOLS)
                        )
                        tree.update_node(nid=node.identifier, tag=new_symbol)
        return genome
