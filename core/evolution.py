import random
import time
from treelib import Tree
from core.genome import Genome
from concurrent.futures import ProcessPoolExecutor
import warnings
import torch
import os
from tasks.parity import compute_fitness

warnings.filterwarnings("ignore")
cpus = os.cpu_count()
torch.set_num_threads(1)


class Evolution:
    """
    Manage the evolution of a population of genomes over multiple generations.

    This class implements an evolutionary algorithm that evolves a population of genomes through selection, 
    crossover, and mutation. It tracks fitness scores, maintains a history of fitness, and allows for the 
    generation of offspring based on the best individuals in the population.

    Args:
        population_size (int, optional): The size of the population. Defaults to 1000.
        generations (int, optional): The number of generations to evolve. Defaults to 300.
        mutation_rate (float, optional): The rate of mutation for individuals. Defaults to 0.05.
        inputs (int, optional): The number of inputs for the genomes. Defaults to 2.
        population (list, optional): An optional initial population. Defaults to None.

    Attributes:
        fitness_history (list): A history of the best fitness scores over generations.
        innovative_individuals (list): A list of individuals that achieved the best fitness scores.
        fitness_grids (dict): A dictionary to store fitness grids.
        fitness_scores (list): A list of fitness scores for the current population.
        logs (list): A log of the evolution process.
        lineage (list): A record of the lineage of the best individuals.
    """

    def __init__(self, population_size=1000, generations=300, mutation_rate=0.05, inputs=2, population=None):
        """
        Initialize the Evolution class with specified parameters.

        This constructor sets up the parameters for the evolutionary process, including the size of the population, 
        the number of generations to evolve, and the mutation rate. It also initializes various attributes to track 
        the fitness history, innovative individuals, and the current population.

        Args:
            population_size (int, optional): The size of the population. Defaults to 1000.
            generations (int, optional): The number of generations to evolve. Defaults to 300.
            mutation_rate (float, optional): The rate of mutation for individuals. Defaults to 0.05.
            inputs (int, optional): The number of inputs for the genomes. Defaults to 2.
            population (list, optional): An optional initial population. Defaults to None.

        Returns:
            None
        """
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

    def initialize_population(self):
        """
        Create and initialize the population of individuals.

        This method generates a list of individuals for the population by calling the method to create each individual. 
        It ensures that the population is ready for the evolutionary process.

        Args:
            self: The instance of the class.

        Returns:
            list: A list containing the initialized individuals in the population.
        """
        return [self.create_individual() for _ in range(self.population_size)]

    def create_individual(self):
        """
        Generate a new individual with a randomly initialized genome.

        This method creates a new instance of a genome and assigns random symbols to the root nodes of each tree 
        within the genome. It ensures that each individual in the population starts with a unique genetic configuration.

        Args:
            self: The instance of the class.

        Returns:
            Genome: The newly created individual represented as a genome.
        """
        genome = Genome()
        symbols = Genome.SYMBOLS
        for n in range(genome.get_levels()):
            symbol = random.choice(symbols)
            root = genome.get_tree(n).root
            genome.change_symbol(level=n, node_id=root, symbol=symbol)

        return genome

    def evolve(self, info=True, max_time=2100, stop=False):    # 2100
        """
        Evolve the population over a specified number of generations.

        This method manages the evolutionary process by generating offspring, selecting the best individuals, 
        and tracking the best fitness scores over generations. It also logs information about the evolution 
        process and stops if the maximum allowed time is exceeded.

        Args:
            info (bool, optional): Whether to log detailed information about each generation. Defaults to True.
            max_time (int, optional): The maximum time allowed for the evolution process in seconds. Defaults to 2100.

        Returns:
            tuple: A tuple containing the best individual from the final population and the number of generations completed.
        """

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
                    # 'fitness_scores': self.fitness_scores
                })

            print(f'{self.generation_time} s')

            if stop and self.fitness_scores[0] == 1:
                break

            print('Elapsed time:', (time.time() - start_time) / 60, 'min')
            if time.time() - start_time > max_time:
                break

        self.lineage = None if self.inputs == 2 else self.get_lineage()

        return self.population[0], generation + 1

    def get_offspring(self):
        """
        Generate offspring from the current population through crossover and mutation.

        This method selects pairs of parents from the population, performs crossover to create new individuals, 
        and applies mutation to introduce variability. The resulting offspring are returned for inclusion in the 
        next generation.

        Args:
            self: The instance of the class.

        Returns:
            list: A list of mutated offspring generated from the current population.
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

    def select_best(self, population):
        """
        Select the best individuals from the given population based on fitness scores.

        This method evaluates the fitness of each individual in the population using parallel processing, 
        sorts them by their fitness scores, and selects the top individuals to form the new population. 
        It also updates the fitness history with the best score from the current selection.

        Args:
            population (list): The population of individuals to evaluate.

        Returns:
            tuple: A tuple containing two lists: the best individuals and their corresponding fitness scores.
        """
        start_time = time.time()
        ns = [self.inputs for _ in range(len(population))]
        with ProcessPoolExecutor(cpus) as executor:
            # fitness_list = list(executor.map(compute_fitness, population, ns))
            fitness_list = list(executor.map(
                self.random_number, population, ns))
        # print('Selection time:', time.time() - start_time)
        individuals_and_fitness = sorted(
            zip(population, fitness_list), key=lambda x: x[1], reverse=True)
        best_individuals = [individual for individual,
                            _ in individuals_and_fitness[:self.population_size]]
        best_fitness_scores = [
            fitness for _, fitness in individuals_and_fitness[:self.population_size]]

        self.fitness_history.append(best_fitness_scores[0])
        return best_individuals, best_fitness_scores

    def edit_population_size(self, acceptable_time=30):
        """
        Adjust the population size based on the time taken for the last generation.

        This method modifies the population size to ensure efficient evolution based on the time taken 
        for the last generation. If the generation time exceeds the acceptable limit, the population size 
        is reduced; if it is below the limit, the population size is increased, within defined minimum 
        and maximum bounds.

        Args:
            acceptable_time (int, optional): The threshold time for generation in seconds. Defaults to 30.

        Returns:
            None
        """
        if self.generation_time > acceptable_time:
            self.population_size = max(
                self.min_population_size, int(self.population_size * .9))
        elif self.generation_time < acceptable_time:
            self.population_size = min(
                self.max_population_size, int(self.population_size * 1.1))

    def crossover(self, parent1, parent2, parents_indexes):
        """
        Perform crossover between two parent individuals to create two children.

        This method generates two offspring by combining genetic information from the provided parent individuals. 
        It utilizes the indices of the parents to facilitate the crossover process.

        Args:
            parent1: The first parent individual.
            parent2: The second parent individual.
            parents_indexes (list): The indices of the parents in the population.

        Returns:
            tuple: A tuple containing the two newly created child individuals.
        """
        child1, child2 = self.get_children(parent1, parent2, parents_indexes)

        return child1, child2

    def get_children(self, parent1, parent2, parents_indexes):
        """
        Create two children from two parent individuals through genetic crossover.

        This method generates two new genomes by combining genetic material from the provided parent individuals. 
        It selects random cut points in the trees of the parents and performs crossover to create offspring that 
        inherit characteristics from both parents.

        Args:
            parent1: The first parent individual.
            parent2: The second parent individual.
            parents_indexes (list): The indices of the parents in the population.

        Returns:
            tuple: A tuple containing two newly created child genomes.
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

    def mutate(self, genome):
        """
        Apply mutations to the given genome based on a mutation rate.

        This method iterates through the trees in the genome and randomly alters the symbols of nodes 
        according to the specified mutation rate. It introduces genetic diversity by changing certain 
        node tags to new symbols, which can be either operational or division symbols.

        Args:
            genome: The genome to be mutated.

        Returns:
            Genome: The mutated genome after applying the changes.
        """
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

    def get_lineage(self, gens_to_save=5):
        """
        Retrieve the lineage of the best individuals over a specified number of generations.

        This method collects and returns the genomes of the best individuals from the evolution process, 
        tracing back through their parent generations. It allows for the analysis of genetic progression 
        by saving a specified number of generations in the lineage.

        Args:
            gens_to_save (int, optional): The number of generations to include in the lineage. Defaults to 5.

        Returns:
            list: A list of dictionaries containing the generation index and the corresponding genome of each individual.
        """

        def traverse_generations(data, generation_idx, individual_idx, genomes):

            individual = data[generation_idx]['individuals'][individual_idx]
            genomes.append({'generation': generation_idx,
                            'genome': individual['genome']})

            parents = individual['parents']
            if parents is not None and generation_idx > len(data) - gens_to_save:
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

    def random_number(self, genome, n):
        return 0
