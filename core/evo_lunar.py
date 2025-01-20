import random
import time
from treelib import Tree
from core.genome import Genome
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import warnings
import torch
import os
from tasks.lunar_lander import compute_fitness as compute_fitness

warnings.filterwarnings("ignore")
cpus = multiprocessing.cpu_count()
torch.set_num_threads(1)


class Evolution:
    """
    Manages the evolution of a population of genomes over multiple generations.

    This class implements an evolutionary algorithm that evolves a population of genomes through selection, 
    crossover, and mutation. It tracks fitness scores, maintains a history of fitness, and allows for the 
    generation of offspring based on the best individuals in the population.

    :param population_size: The size of the population (default is 1000).
    :param generations: The number of generations to evolve (default is 300).
    :param mutation_rate: The rate of mutation for individuals (default is 0.05).
    :param inputs: The number of inputs for the genomes (default is 2).
    :param population: An optional initial population (default is None).

    :ivar fitness_history: A history of the best fitness scores over generations.
    :ivar innovative_individuals: A list of individuals that achieved the best fitness scores.
    :ivar fitness_grids: A dictionary to store fitness grids.
    :ivar fitness_scores: A list of fitness scores for the current population.
    :ivar logs: A log of the evolution process.
    :ivar lineage: A record of the lineage of the best individuals.
    :ivar fitness_function: The function used to compute the fitness of individuals.
    """

    def __init__(self, population_size: int = 1000, generations: int = 300, mutation_rate: float = 0.05, inputs: int = 2, population: list = None):
        """
        Initialize the Evolution class with specified parameters.

        This constructor sets up the parameters for the evolutionary process, including the size of the population, 
        the number of generations to evolve, and the mutation rate. It also initializes various attributes to track 
        the fitness history, innovative individuals, and the current population.

        :param population_size: The size of the population (default is 1000).
        :param generations: The number of generations to evolve (default is 300).
        :param mutation_rate: The rate of mutation for individuals (default is 0.05).
        :param inputs: The number of inputs for the genomes (default is 2).
        :param population: An optional initial population (default is None).

        :return: None
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.fitness_function = compute_fitness
        self.fitness_history = []
        self.innovative_individuals = []
        self.fitness_grids = {}
        self.inputs = inputs
        self.fitness_scores = [None] * self.population_size
        self.logs = []
        self.saved_individuals = []

        self.population = population or self.initialize_population()

    def resume_evolution(self, state_file: str):
        """
        Resume the evolution process from a saved state file.

        This method loads the state of the evolution process from a specified file and continues the evolution 
        from where it left off. It updates the population, fitness scores, and other relevant attributes.

        :param state_file: The path to the file containing the saved state.

        :return: None
        """
        if not os.path.exists(state_file):
            raise FileNotFoundError(f"State file {state_file} not found.")

        state = torch.load(state_file)
        self.population = state['population']
        self.fitness_scores = state['fitness_scores']
        self.logs = state['logs']
        self.innovative_individuals = state['innovative_individuals']
        self.fitness_history = [log['best_score'] for log in self.logs]
        self.generations -= state['generation']

    def initialize_population(self):
        """
        Create and initialize the population of individuals.

        This method generates a list of individuals for the population by calling the method to create each individual. 
        It ensures that the population is ready for the evolutionary process.

        :return: A list containing the initialized individuals in the population.
        """
        return [self.create_individual() for _ in range(self.population_size)]

    def create_individual(self):
        """
        Generate a new individual with a randomly initialized genome.

        This method creates a new instance of a genome and assigns random symbols to the root nodes of each tree 
        within the genome. It ensures that each individual in the population starts with a unique genetic configuration.

        :return: The newly created individual represented as a genome.
        """
        genome = Genome()
        symbols = genome.SYMBOLS
        for n in range(genome.get_levels()):
            symbol = random.choice(symbols)
            root = genome.get_tree(n).root
            genome.change_symbol(level=n, node_id=root, symbol=symbol)

        return genome

    def evolve(self, info: bool = True, stop: bool = True, index = 0):
        """
        Evolve the population over a specified number of generations.

        This method manages the evolutionary process by generating offspring, selecting the best individuals, 
        and tracking the best fitness scores over generations. It also logs information about the evolution 
        process and stops if the maximum allowed time is exceeded.

        :param info: Whether to log detailed information about each generation (default is True).
        :param stop: Whether to stop if the best fitness score is 1 (default is True).

        :return: A tuple containing the best individual from the final population and the number of generations completed.
        """
        best_score = float('-inf')
        start_time = time.time()

        if self.generations <= 0:
            return self.population[0], 0

        for generation in range(self.generations):
            generation_start_time = time.time()
            print(f'Run {index}/10')
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
                    # 'individuals': [individual.json_pickle() for individual in self.population],
                    # 'fitness_scores': self.fitness_scores
                })

            print(f'{self.generation_time} s')
            print(f'Best fitness: {self.fitness_scores[0]}')

            if stop and self.fitness_scores[0] == 1:
                break

            state_folder = 'state'
            if not os.path.exists(state_folder):
                os.makedirs(state_folder)
            state_file = os.path.join(state_folder, f'state_{index}.pkl')
            with open(state_file, 'wb') as f:
                torch.save({
                    'generation': generation,
                    'population': self.population,
                    'fitness_scores': self.fitness_scores,
                    'best_score': best_score,
                    'logs': self.logs,
                    'innovative_individuals': self.innovative_individuals
                }, f)

            print('Elapsed time:', (time.time() - start_time) / 60, 'min')

        # self.lineage = self.get_lineage()

        self.saved_individuals = [individual.json_pickle() for individual in self.population[:10]] + \
            [individual.json_pickle()
             for individual in random.sample(self.population[10:], 10)]

        return self.population[0], generation + 1

    def get_offspring(self):
        """
        Generate offspring from the current population through crossover and mutation.

        This method selects pairs of parents from the population, performs crossover to create new individuals, 
        and applies mutation to introduce variability. The resulting offspring are returned for inclusion in the 
        next generation.

        :return: A list of mutated offspring generated from the current population.
        """
        start_time = time.time()
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

        population = [self.mutate(individual) for individual in offspring]
        # print('Evolution time:', time.time() - start_time)
        return population

    def select_best(self, population: list):
        """
        Select the best individuals from the given population based on fitness scores.

        This method evaluates the fitness of each individual in the population using parallel processing, 
        sorts them by their fitness scores, and selects the top individuals to form the new population. 
        It also updates the fitness history with the best score from the current selection.

        :param population: The population of individuals to evaluate.

        :return: A tuple containing two lists: the best individuals and their corresponding fitness scores.
        """
        start_time = time.time()
        ns = [self.inputs for _ in range(len(population))]
        # ns = [1 for _ in range(len(population))]
        with ProcessPoolExecutor(cpus) as executor:
            fitness_list = list(executor.map(
                self.fitness_function, population, ns))
        # print('Selection time:', time.time() - start_time)
        individuals_and_fitness = sorted(
            zip(population, fitness_list), key=lambda x: x[1], reverse=True)
        best_individuals = [individual for individual,
                            _ in individuals_and_fitness[:self.population_size]]
        best_fitness_scores = [
            fitness for _, fitness in individuals_and_fitness[:self.population_size]]

        self.fitness_history.append(best_fitness_scores[0])
        return best_individuals, best_fitness_scores

    def crossover(self, parent1: Genome, parent2: Genome, parents_indexes: list):
        """
        Perform crossover between two parent individuals to create two children.

        This method generates two offspring by combining genetic information from the provided parent individuals. 
        It utilizes the indices of the parents to facilitate the crossover process.

        :param parent1: The first parent individual.
        :param parent2: The second parent individual.
        :param parents_indexes: The indices of the parents in the population.

        :return: A tuple containing the two newly created child individuals.
        """
        child1, child2 = self.get_children(parent1, parent2, parents_indexes)

        return child1, child2

    def get_children(self, parent1: Genome, parent2: Genome, parents_indexes: list):
        """
        Create two children from two parent individuals through genetic crossover.

        This method generates two new genomes by combining genetic material from the provided parent individuals. 
        It selects random cut points in the trees of the parents and performs crossover to create offspring that 
        inherit characteristics from both parents.

        :param parent1: The first parent individual.
        :param parent2: The second parent individual.
        :param parents_indexes: The indices of the parents in the population.

        :return: A tuple containing the two newly created child individuals.
        """
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
                tree2.get_node(root).parent = parent.identifier
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
                    root).parent = parent2_node.identifier
                tree2.remove_node(cutpoint2)
                tree2.paste(parent2_node.identifier, subtree1)
                new_tree2 = Tree(tree=tree2, deep=True)
            else:
                new_tree2 = Tree(tree=subtree1, deep=True)
            trees2.append(new_tree2)

        return Genome(trees, parents_indexes), Genome(trees2, parents_indexes)

    def mutate(self, genome: Genome):
        """
        Apply mutation to an individual genome.

        This method introduces small random changes in the genome by modifying the genetic material of the individual. 
        Mutation helps introduce genetic diversity and allows the evolutionary process to explore a larger solution space.

        :param individual: The individual genome to mutate.

        :return: The mutated individual genome.
        """
        for i, tree in enumerate(genome.get_trees()):
            nodes = list(tree.all_nodes_itr())
            for node in nodes:
                if random.random() < self.mutation_rate:
                    if node.tag in ['e', 'n']:
                        new_symbol = random.choice(genome.SYMBOLS)
                        genome.change_symbol(
                            level=i, node_id=node.identifier, symbol=new_symbol)
                    else:
                        new_symbol = (
                            random.choice(genome.DIVISION_SYMBOLS)
                            if node.tag in genome.DIVISION_SYMBOLS
                            else random.choice(genome.OPERATIONAL_SYMBOLS)
                        )
                        tree.update_node(nid=node.identifier, tag=new_symbol)
        return genome

    def get_lineage(self, gens_to_save: int = 5):
        """
        Generate a lineage of the best individuals over generations.

        This method creates a record of the best individuals from each generation, which is useful for tracking 
        the evolutionary progress and visualizing how the population has changed over time.

        :return: A list of the best individuals from each generation.
        """
        def traverse_generations(data: dict, generation_idx: int, individual_idx: int, genomes: list):

            individual = data[generation_idx]['individuals'][individual_idx]
            genomes.append({'generation': generation_idx,
                            'genome': individual['genome']})

            parents = individual['parents']
            if parents is not None and generation_idx > max(len(data) - gens_to_save, 0):
                traverse_generations(
                    data, generation_idx - 1, parents[0], genomes)
                traverse_generations(
                    data, generation_idx - 1, parents[1], genomes)

        genomes = []
        generations = len(self.logs)
        traverse_generations(self.logs, generations - 1, 0, genomes)

        for generation in self.logs:
            del generation['individuals']
        return genomes
