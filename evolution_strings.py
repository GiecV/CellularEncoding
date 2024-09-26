import random
import time
import string


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
        self.display_individuals()

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

        # Define the characters to choose from
        characters = string.ascii_letters + string.digits
        # Generate a random string
        random_string = ''.join(random.choices(characters, k=10))

        return random_string

    # * Evolve the population

    def evolve(self):
        start_time = time.time()  # Start the simulation timer
        for generation in range(self.generations):
            print(f"Generation {generation+1}/{self.generations}")
            if (
                time.time() - start_time > 7200
            ):  # If more than 2 hours have passed, end the evolution
                break
            for i in range(self.num_islands):
                print(f"Evolving island {i}")
                self.evolve_island(i)  # Evolve each island

        # Return the best individual in the evolution
        return self.select_best_individual()

    # * Evolve a single island
    def evolve_island(self, island_index):

        island = self.islands[island_index]
        for _ in range(self.island_size * self.island_size):
            if (
                random.random() < self.exchange_rate
            ):  # Migrate an individual with a certain probability
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
                        offspring = self.crossover(
                            parent1, parent2)  # Perform crossover
                        # Mutate the offspring
                        offspring = self.mutate(offspring)
                    # Place offspring in the tile
                    island[s[0]][s[1]] = offspring

    # * Perform a random walk and return the best individual
    def random_walk(self, island, start):

        x, y = start
        best_individual = island[x][y]

        steps = []
        for _ in range(3):  # Perform 10 steps of random walk
            # Choose a random direction
            direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
            steps.append(direction)
            # Move in the x direction
            x = (x + direction[0]) % self.island_size
            # Move in the y direction
            y = (y + direction[1]) % self.island_size
            if best_individual is None or self.compute_fitness(island[x][y]) > self.compute_fitness(best_individual):
                # If there is something on the tile and it is better than the current best, update it
                best_individual = island[x][y]
        # print(steps)

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
    def compute_fitness(self, s1):

        s2 = 'abcde12345'

        correct_chars = 0

        for i in range(len(s1)):
            if s1[i] == s2[i]:
                correct_chars += 1

        return correct_chars

    # * Perform crossover between two parents
    def crossover(self, parent1, parent2):

        cutpoint = random.randint(0, len(parent1))
        offspring = parent1[:cutpoint] + parent2[cutpoint:]
        return offspring

    # * Mutate a symbol of the individual
    def mutate(self, individual):

        for i in range(len(individual)):
            if random.random() < 0.05:
                new_char = random.choice(
                    string.ascii_letters + string.digits)
                individual = individual[:i] + new_char + individual[i+1:]

        return individual

    # * Display the genotype of each individual
    def display_individuals(self):
        for island_index, island in enumerate(self.islands):
            print(f"Island {island_index}:")
            for row_index, row in enumerate(island):
                for col_index, individual in enumerate(row):
                    if individual is not None:
                        print(f"Individual at ({row_index}, {col_index}):")
                        print(individual)
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

        return best_individual, best_fitness
