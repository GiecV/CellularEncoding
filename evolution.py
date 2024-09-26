import random
import time
import gym
from gym.wrappers import record_video
import torch
import copy
from treelib import Tree
from genome import Genome
from phenotype import Phenotype
from neural_network_from_graph import NNFromGraph
from counter import GlobalCounter
import concurrent.futures


class Evolution:

    # * Initialize the evolution
    def __init__(self, num_islands, island_size, generations, exchange_rate=0.01, inputs=1, outputs=4):
        self.num_islands = num_islands
        self.island_size = island_size
        self.generations = generations
        self.exchange_rate = exchange_rate
        self.inputs = inputs
        self.outputs = outputs

        self.islands = self.initialize_islands()
        self.env = gym.make("CartPole-v1")  # Chosen task is cart-pole

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

        for n in range(genome.LEVELS):
            symbol = random.choice(
                Genome.JUMPING_SYMBOLS + Genome.DIVISION_SYMBOLS + Genome.OTHER_SYMBOLS
            )  # Choose a random symbol for the first node
            root = genome._trees[n].root
            genome.change_symbol(level=n, node_id=root, symbol=symbol)

        phenotype = Phenotype(genome)  # Create the phenotype
        # Expand the inputs and outputs of the phenotype
        phenotype.expand_some_inputs_and_outputs(
            # The outputs to the NN are the inputs to the env and vice versa
            inputs=self.outputs, outputs=self.inputs)

        # Generate the corresponding neural network
        nn = NNFromGraph(phenotype)

        return nn

    # * Evolve the population
    def evolve(self):
        start_time = time.time()  # Start the simulation timer
        for generation in range(self.generations):
            print(f"Generation {generation+1}/{self.generations}")
            if (
                time.time() - start_time > 36_000
            ):  # If more than 2 hours have passed, end the evolution
                break

            for i in range(self.num_islands):
                print(f"Evolving island {i+1}")
                self.evolve_island(i)  # Evolve each island

            print(f'{time.time() - start_time}s')

        # Return the best individual in the evolution

        return self.select_best_individual()

    # * Evolve a single island
    def evolve_island(self, island_index):

        island = self.islands[island_index]
        for _ in range(self.island_size * self.island_size):
            if (
                random.random() < self.exchange_rate
            ):  # Migrate an individual with a certain probability
                # print('Migrating')
                self.exchange_individual(island_index)
            else:  # Select a random site on the island
                s = (
                    random.randint(0, self.island_size - 1),
                    random.randint(0, self.island_size - 1),
                )
                # Check if the tile is not empty
                if island[s[0]][s[1]] is not None:
                    # Find best individual in a random walk
                    parent1 = self.random_walk(island, s)
                    # Same for the second parent
                    parent2 = self.random_walk(island, s)
                    tries = 0
                    while parent1 == parent2 and tries < 5:  # Do it again if the two individuals are the same
                        parent2 = self.random_walk(island, s)
                        tries += 1
                    if parent1 == parent2:
                        offspring = parent1
                    else:
                        # print(f'Crossing over {(s[0], s[1])}')
                        offspring = self.crossover(
                            parent1, parent2)  # Perform crossover
                        # Mutate the offspring
                        # print(f'Mutating {(s[0], s[1])}')
                        offspring = self.mutate(offspring)
                    # Place offspring in the tile
                    island[s[0]][s[1]] = offspring

    # * Perform a random walk and return the best individual
    def random_walk(self, island, start):

        x, y = start
        best_individual = island[x][y]

        for _ in range(10):  # Perform 10 steps of random walk
            # Choose a random direction
            direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
            # Move in the x direction
            x = (x + direction[0]) % self.island_size
            # Move in the y direction
            y = (y + direction[1]) % self.island_size
            if best_individual is None or self.compute_fitness(island[x][y]) > self.compute_fitness(best_individual):
                # If there is something on the tile and it is better than the current best, update it
                best_individual = island[x][y]

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
        best_individual = self.random_walk(
            island, s
        )  # Perform a random walk from the site
        neighbor_index = self.get_random_neighbor_index(
            island_index
        )  # Choose a random neighbor island
        self.receive_individual(
            neighbor_index, best_individual, s
        )  # Get the individual from neighboring island

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

        for row in island:
            for individual in row:
                if individual is not None:
                    # Compute the fitness of each individual
                    fitness = self.compute_fitness(individual)
                    if fitness > best_fitness:  # If it improves the best, update it
                        best_fitness = fitness
                        best_individual = individual

        return best_individual

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
                action = individual.forward(torch.tensor(obs).float())
                if action > 0:
                    action = 1
                else:
                    action = 0
                obs, reward, done, truncated, info = self.env.step(
                    action)  # Perform the action
                # print(f'Action {action} for individual {
                #       individual.phenotype.genome}')
                total_reward += reward

        return total_reward/trials  # Mean of the reward on the trials

    # * Perform crossover between two parents
    def crossover(self, parent1, parent2):

        trees = []
        g = Genome()

        for level in range(Genome.LEVELS):

            tree1 = Tree(
                tree=parent1.phenotype.genome._trees[level], deep=True)
            tree2 = Tree(
                tree=parent2.phenotype.genome._trees[level], deep=True)

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
        p.expand_some_inputs_and_outputs(
            # The outputs to the NN are the inputs to the env and vice versa
            inputs=self.outputs, outputs=self.inputs)
        nn = NNFromGraph(p)

        return nn

    # * Update the identifiers of the nodes in the tree to be unique
    def update_ids(self, tree):

        tree2 = Tree(tree=tree, deep=True)  # Copy the tree

        for node in tree.all_nodes_itr():
            # Update the identifier
            tree2.update_node(node.identifier, identifier=GlobalCounter.next())

        return tree2

    # * Mutate a symbol of the individual
    def mutate(self, individual):

        genome = individual.phenotype.genome
        new_genome = copy.deepcopy(genome)

        i = 0
        for tree in genome._trees:
            for node in tree.all_nodes_itr():
                if random.random() < 0.005:  # There is a small chance to mutate a symbol
                    if node.tag in Genome.DIVISION_SYMBOLS:  # Randomly choose a division symbol
                        new_symbol = random.choice(
                            Genome.DIVISION_SYMBOLS)
                    else:  # Randomly choose an unary symbol
                        new_symbol = random.choice(
                            Genome.TERMINAL_SYMBOLS
                            + Genome.JUMPING_SYMBOLS
                            + Genome.OTHER_SYMBOLS
                            # + Genome.OPERATIONAL_SYMBOLS
                        )
                        # if new_symbol in Genome.OPERATIONAL_SYMBOLS:
                        #         new_symbol += str(random.uniform(0, 1))

                    new_genome.change_symbol(  # Change the symbol
                        level=i, node_id=node.identifier, symbol=new_symbol
                    )

            i += 1

        p = Phenotype(new_genome)
        p.expand_some_inputs_and_outputs(
            # The outputs to the NN are the inputs to the env and vice versa
            inputs=self.outputs, outputs=self.inputs)
        nn = NNFromGraph(p)
        return nn

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

    # * Select the best individual from all islands
    def select_best_individual(self):
        best_individual = None
        best_fitness = float("-inf")

        for island in self.islands:
            for row in island:
                for individual in row:
                    if individual is not None:
                        fitness = self.compute_fitness(individual)
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_individual = individual

        # self.save_video(best_individual)

        return best_individual, best_fitness

    def save_video(self, individual):

        wrapped_env = record_video(self.env, path='./video.mp4')
        obs = wrapped_env.reset()  # Reset the environment
        obs = obs[0]
        done = False
        total_reward = 0
        max_steps = 200

        for _ in range(max_steps):  # Simulate the environment
            if done:
                break
            # Get the action from the individual
            action = individual.forward(torch.tensor(obs).float())
            if action > 0:
                action = 1
            else:
                action = 0
            obs, reward, done, truncated, info = wrapped_env.step(
                action)  # Perform the action
            total_reward += reward

        return total_reward  # Mean of the reward on the trials
