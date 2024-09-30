import random
import time
import gym
import torch
import copy
from treelib import Tree
from cellular_encoding.genome import Genome
from cellular_encoding.phenotype import Phenotype
from cellular_encoding.neural_network_from_graph import NNFromGraph
from utils.counter import GlobalCounter
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import utils.convert_action as convert
import warnings

warnings.filterwarnings("ignore")
cpus = 12
torch.set_num_threads(1)


class Evolution:

    # * Initialize the evolution
    def __init__(self, num_islands, island_size, generations, exchange_rate=0.01, mutation_rate=0.005, inputs=2, outputs=1):
        self.num_islands = num_islands
        self.island_size = island_size
        self.generations = generations
        self.exchange_rate = exchange_rate
        self.mutation_rate = mutation_rate
        self.inputs = inputs
        self.outputs = outputs
        self.fitness_history = []
        self.fitness_grids = {}

        self.islands = self.initialize_islands()
        self.env = gym.make("CartPole-v1")  # Chosose task
        # self.env = gym.make("Pendulum-v1")

    # * For each island, create the grid and populate it
    def initialize_islands(self):
        islands = []

        for i in range(self.num_islands):
            island = []
            for j in range(self.island_size):
                row = []
                for k in range(self.island_size):
                    # Create an individual for each cell
                    row.append(self.create_individual())
                island.append(row)
            islands.append(island)

        return islands

    # * Create a new individual
    def create_individual(self):
        genome = Genome()  # Create a new genome
        symbols = copy.deepcopy(Genome.SYMBOLS)
        symbols.remove('n2')
        symbols.remove('n1')

        for n in range(genome.get_levels()):
            # Choose a random symbol for the first node

            symbol = random.choice(symbols)
            root = genome.get_tree(n).root
            genome.change_symbol(level=n, node_id=root, symbol=symbol)

        phenotype = Phenotype(genome)  # Create the phenotype
        nn = NNFromGraph(phenotype, inputs=self.inputs, outputs=self.outputs)  # Create the neural network

        return nn

    # * Evolve the population
    def evolve(self):

        for generation in range(self.generations):
            start_time = time.time()  # Start the simulation timer
            print(f"Generation {generation + 1}/{self.generations}")
            if (
                time.time() - start_time > 36_000
            ):  # If more than 10 hours have passed, end the evolution
                break
            for i in range(self.num_islands):
                print(f"Evolving island {i + 1}")
                self.evolve_island(i)  # Evolve each island

            print(f'{time.time() - start_time}s')
            best_individual, best_fitness = self.select_best_among_all_islands()
            self.fitness_history.append(best_fitness)

        # Return the best individual in the evolution
        return self.select_best_among_all_islands()

    # * Evolve a single island
    def evolve_island(self, island_index):
        best_individual, _, self.fitness_grids[island_index] = self.select(self.islands[island_index])
        island = self.islands[island_index]

        for _ in range(self.island_size * self.island_size):
            if (random.random() < self.exchange_rate):  # Migrate an individual with a certain probability
                self.exchange_individual(island_index)
            else:  # Select a random site on the island
                s = (
                    random.randint(0, self.island_size - 1),
                    random.randint(0, self.island_size - 1),
                )
                # Protect the best individual on the island
                if island[s[0]][s[1]] != best_individual:
                    self.evolve_individual(island, island_index, s)

    # * Evolve an individual
    def evolve_individual(self, island, island_index, s):
        parent1 = island[s[0]][s[1]]  # Find best individual in a random walk
        parent2 = self.random_walk(island_index, s)  # Same for the second parent
        tries = 0

        while parent1 == parent2 and tries < 5:  # Do it again if the two individuals are the same
            parent2 = self.random_walk(island_index, s)
            tries += 1
        if parent1 == parent2:
            offspring_phenotype = parent1.phenotype
        else:
            offspring_phenotype = self.crossover(parent1, parent2)  # Perform crossover
            # Mutate the offspring
            offspring_phenotype = self.mutate(offspring_phenotype)
        # Place offspring in the tile
        offspring = NNFromGraph(offspring_phenotype, inputs=self.inputs, outputs=self.outputs)

        island[s[0]][s[1]] = offspring

    # * Perform a random walk and return the best individual
    def random_walk(self, island_index, start, steps=10):
        x, y = start
        best_individual = None
        best_fitness = None
        moves = []

        island = self.islands[island_index]

        for _ in range(steps):  # Perform 10 steps of random walk
            direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])  # Choose a random direction

            moves.append(direction)
            # Move in the x direction
            x = (x + direction[0]) % self.island_size
            # Move in the y direction
            y = (y + direction[1]) % self.island_size
            fitness = self.fitness_grids[island_index][x][y]
            if best_individual is None or fitness > best_fitness:
                # If there is something on the tile and it is better than the current best, update it
                best_individual = island[x][y]
                best_fitness = fitness

        return best_individual  # Return the best individual

    # * Let an individual migrate to a neighbor island
    def exchange_individual(self, island_index):
        island = self.islands[island_index]  # Get the current island
        border_sites = (
            [(i, 0) for i in range(self.island_size)]
            + [(i, self.island_size - 1) for i in range(self.island_size)]
            + [(0, i) for i in range(self.island_size)]
            + [(self.island_size - 1, i) for i in range(self.island_size)]
        )  # Get the border sites of the island

        s = random.choice(border_sites)  # Choose a random site on the border
        best_individual = self.random_walk(island_index, s)  # Perform a random walk from the site
        neighbor_index = self.get_random_neighbor_index(island_index)  # Choose a random neighbor island
        self.receive_individual(neighbor_index, best_individual, s)  # Get the individual from neighboring island

    # * Get a random neighbor island index
    def get_random_neighbor_index(self, island_index):
        grid_size = int(self.num_islands**0.5)  # Get the length of the grid
        row = island_index // grid_size
        col = island_index % grid_size

        neighbors = [
            ((row - 1) % grid_size) * grid_size + col,  # Up
            ((row + 1) % grid_size) * grid_size + col,  # Down
            row * grid_size + (col - 1) % grid_size,  # Left
            row * grid_size + (col + 1) % grid_size,  # Right
        ]

        return random.choice(neighbors)

    # * Receive the best individual from another island
    def receive_individual(self, island_index, individual, s):
        island = self.islands[island_index]

        # Get the opposite site
        opposite_s = (self.island_size - 1 - s[0], self.island_size - 1 - s[1])
        # Place the individual in the opposite site
        island[opposite_s[0]][opposite_s[1]] = individual

    # * Select the fittest individual from the island
    def select(self, island):
        best_individual = None
        best_fitness = float("-inf")

        with ProcessPoolExecutor(cpus) as executor:
            fitness_list = executor.map(self.compute_fitness, [individual for row in island for individual in row])
        fitness_list = list(fitness_list)

        # fitness_list = [self.compute_fitness(individual) for row in island for individual in row]

        fitness_grid = [[0 for _ in range(len(island[0]))]
                        for _ in range(len(island))]
        best_individual, best_fitness = self.update_fitness_grid(island, best_fitness, fitness_list, fitness_grid)

        return best_individual, best_fitness, fitness_grid

    def update_fitness_grid(self, island, best_fitness, fitness_list, fitness_grid):
        index = 0

        for i in range(len(island)):
            for j in range(len(island[i])):
                fitness_grid[i][j] = fitness_list[index]
                if fitness_list[index] > best_fitness:
                    best_fitness = fitness_list[index]
                    best_individual = island[i][j]
                index += 1

        return best_individual, best_fitness

    # * Compute the fitness of an individual
    def compute_fitness(self, individual):
        obs = self.env.reset()  # Reset the environment
        obs = obs[0]
        done = False
        total_reward = 0
        max_steps = 200
        trials = 5

        for _ in range(trials):
            for _ in range(max_steps):  # Simulate the environment
                if done:
                    break
                # Get the action from the individual
                obs = torch.tensor(obs.tolist()).float()
                action = individual.forward(obs)
                action = convert.cartpole(action)
                obs, reward, done, truncated, info = self.env.step(action)  # Perform the action
                total_reward += reward

        return total_reward / trials  # Mean of the reward on the trials

    # * Perform crossover between two parents
    def crossover(self, parent1, parent2):
        trees = []
        g = Genome()

        for level in range(g.get_levels()):
            tree1 = Tree(tree=parent1.phenotype.genome.get_tree(level), deep=True)
            tree2 = Tree(tree=parent2.phenotype.genome.get_tree(level), deep=True)
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
            tree = self.update_ids(tree)  # Update ids for avoiding duplicates
            trees.append(tree)  # Save the tree in the genome

        g = Genome(trees)
        p = Phenotype(genome=g)

        return p

    # * Update the identifiers of the nodes in the tree to be unique
    def update_ids(self, tree):

        tree2 = Tree(tree=tree, deep=True)  # Copy the tree

        for node in tree.all_nodes_itr():
            # Update the identifier
            tree2.update_node(node.identifier, identifier=GlobalCounter.next())

        return tree2

    # * Mutate a symbol of the individual
    def mutate(self, phenotype):
        genome = phenotype.genome
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

        p = Phenotype(new_genome)

        return p

    # * Select the best individual from all islands
    def select_best_among_all_islands(self):
        best_individual = None
        best_fitness = float("-inf")

        for island_index in range(self.num_islands):
            fitness_grid = self.fitness_grids[island_index]
            for i in range(self.island_size):
                for j in range(self.island_size):
                    if fitness_grid[i][j] > best_fitness:
                        best_fitness = fitness_grid[i][j]
                        best_individual = self.islands[island_index][i][j]

        return best_individual, best_fitness

    # * Display the genotype of each individual
    def display_individuals(self):
        for island_index, island in enumerate(self.islands):
            print(f"Island {island_index}:")
            for row_index, row in enumerate(island):
                for col_index, individual in enumerate(row):
                    if individual is not None:
                        print(f"Individual at ({row_index}, {col_index}):")
                        individual.phenotype.genome.print()
                    else:
                        print(f"Empty slot at ({row_index}, {col_index})")

    # * Plot the best fitness in every generation
    def plot_fitness_history(self):
        generations = list(range(1, len(self.fitness_history) + 1))
        plt.plot(generations, self.fitness_history, marker='o')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness History Over Generations')
        plt.grid(True)
        plt.show()
