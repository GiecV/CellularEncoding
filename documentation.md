# Documentation

### Cellular Encoding
#### Genome
###### Constants
- `LEVELS`: amount of trees that the genome can hold
- `STARTING_SYMBOL`: symbol kept by newborn cells
- `SYMBOLS = TERMINAL + JUMPING + DIVISION + OPERATIONAL`
###### Methods
- `__init__(self, trees=None)`: create the trees or use the given parameter
- `change_symbol(self, level, node, symbol)`: update a node (within a level) with a different symbol. Then, create children if needed.
- `get_symbol(self, level, node_id)`: return the symbol of a node (within a level).
- `print(self, level)`: print one or all levels (if level is None).
- `get_genome_from_starting_point(self, node_id)`: Get the subtrees from a starting point
- `get_left_child_genome(self)`: Get left subtree
- `get_right_child_genome(self)`: Get right subtree
- `get_root_symbol(self)`: get the next symbol to be parsed
- `jump_to_other_level(self, n)`: cut subtrees and go directly to another one
#### Phenotype
###### Methods
- `__init__(self, genome)`: create the initial structure for the neural network (input, hidden neuron, output) and connect them with edges of weight 1.
- `add_cell(self)`: add a new cell to the structure, its id is a progressive number starting from 0. Return the corresponding id.
- `develop(self)`: for each node in the structural nodes of the network (i.e. no input or output nodes), parse the root symbol and perform the operation. Supported operations:
	- e, w, n, p, s, r, c, d, i, +, -
- `print(self)`: show graphically the structure
- `print_no_position(self)`: show the graph without caring about the position of the nodes (useful for not connected components)
- `development_finished(self)`: return True if every cell finished developing, otherwise False
- `expand_inputs_and_outputs(self, inputs, outputs, has_bias)`: expand the single input and output to match the number of neurons in the first layer
- `create_bias(self)`: add a bias neuron to the structure with tag IB, connected to all hidden cells with a weight 1

#### NNFromGraph
###### Methods
- `__init__(self, phenotype, depth=4, inputs=2, outputs=1, has_bias=True)`: create the neural network from a graph and find input and output nodes
- `forward(self, obs)`: forward pass means propagating information from the input nodes to the output nodes by doing a weighted sum of the input

#### Evolution
###### Methods
- `__init__(self, num_islands, island_size, generations, exchange_rate=0.01, mutation_rate=0.005, inputs=2, outputs=1)`: initialize the islands and select the task
- `initialize_islands(self)`: create the grid for each island and populate it
- `create_individual(self)`: create a new individual
- `evolve(self)`:  evolve the population controlling generations and islands
- `evolve_island(self, island_index)`: decide if an individual has to migrate or to breed.
- `evolve_individual(self, island, island_index, s)`: perform random walk, crossover and mutation
- `random_walk(island, start)`: perform a random walk and return the best individual
- `exchange_individual(self, island_index)`: migrate individual to a neighbor island
- `get_random_neighbor_index(self, island_index)`: get index of a random neighbor island
- `receive_individual(self, island_index, individual, s)`: receive individual from another island
- `select(self, island)`: select the fittest individual from the island
- `update_fitness_grid(self, island, best_fitness, fitness_list, fitness_grid)`: save the fitness for each individual in an island
- `compute_fitness(self, individual)`: compute the fitness of an individual
- `crossover(self, parent1, parent2)`: perform crossover between two parents
- `update_ids(self, tree)`: update the identifiers of the nodes in the tree to be unique
- `mutate(self, phenotype)`: mutate a symbol of the individual
- `display_individuals(self)`: display the genotype of each individual
- `select_best_among_all_islands(self)`: select the best individual from all islands
- `plot_fitness_history(self)`: plot the best fitness in each generation
### Tasks
#### xor_gate.py
- `compute_fitness_information(individual)`: compute the xor for each combination of 2 inputs. Compute the mutual information (using `sklearn.metrics.mutual_info_score(f_nn, f_target)`). Compute `t` (add 0.5 if the input layer is of the correct size, add 0.5 if the output layer is of the correct size). Compute and return the fitness with: $$fitness = 0.85 \frac{\tau(f_{nn}, f_{target})}{\tau(f_{target}, f_{target})}+0.15*t$$
- `compute_each_input_combination(individual)`: compute the output of the neural network for each combination of inputs and print the result
### Utils
#### convert_action.py
- `pendulum(action)`: returns $4 * (action - 0.5)$
- `cartpole(action)`: returns 1 if `action` > 0 else 0
#### Counter
###### Constants
- `_counter`: the last id that has been returned
###### Methods
- `next(cls)`: increment the counter and return the value
- `next_str(cls)`: increment the counter and return the value as a string
