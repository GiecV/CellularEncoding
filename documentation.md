### Cellular Encoding
#### Genome
###### Constants
- `LEVELS`: amount of trees that the genome can hold
- `STARTING_SYMBOL`: symbol kept by newborn cells
- `SYMBOLS = TERMINAL_SYMBOLS + DIVISION_SYMBOLS + OPERATIONAL_SYMBOLS`
###### Methods
- `__init__(self, trees:list = None)`: create the trees or use the given parameter
- `change_symbol(self, level: int, node_id: str, symbol: str)`: update node with id `node_id` in level `level` with a symbol `symbol`. Create children nodes if needed.
- `get_symbol(self, level: str, node_id: str)`: return the tag of node `node_id` in level `level`.
- `print(self, level: int = None)`: print one or all levels (if level is None).
- `get_genome_from_starting_point(self, node_id: str)`: Get the subtrees from a starting point
- `get_left_child_genome(self)`: Get left subtree
- `get_right_child_genome(self)`: Get right subtree
- `get_root_symbol(self)`: get the next symbol to be parsed
- `jump(self)`: return `trees[1:] + [tree]`, where `tree` is a starting tree
- `get_trees(self)`: return the list of trees
- `get_levels(self)`: return `self.LEVELS`
- `get_tree(self, level: int)`: return a specific tree
- `update_ids(self)`: update the id of each node in the trees using unique numbers
#### Phenotype
###### Methods
- `__init__(self, genome)`: create the initial structure for the neural network:
- 1 input neuron `I`
- 1 hidden neuron with variable `id`
- 1 output neuron `O`
`I` is connected to `id` with weight +1
`id` is connected to `O` with weight -1
- `add_cell(self)`: add a new hidden neuron to the structure
- `develop(self)`: for each node in the structural nodes of the network (i.e. no input or output nodes), parse the root symbol and perform the operation. 
- `perform_operation(self, structural_node, symbol, genome)`: perform the proper operation according to the symbol
- `edit_threshold(self, structural_node)`: edit threshold of a cell
-`jump(self, structural_node, genome)`: jump to the next level
-`split_parallel(self, structural_node`: perform a parallel split
-`split_sequential(self, structural_node)`: perform a sequential split
-`add_recurrent_edge(self, structural_node)`: add a recurrent edge
-`edit_register(self, value)`: set `self.internal_register` to `value`
-`read_genome(self, structural_node, new_node, symbol, genome)`: move the pointer of the cell to the next symbol
-`split(self, structural_node, new_node, genome)`: choose the next symbols for the new cells
-`continue_reading(self, structural_node, genome)`: move to the next symbol
-`change_weight(self, structural_node, weight)`: set `weight` as the weight of the input edge pointed by `self.internal_register`
- `development_finished(self)`: return `True` if every cell cannot develop further, otherwise `False`
- `expand_inputs_and_outputs(self, inputs, outputs)`: expand the single input and output to match the number of neurons in the first layer
#### NNFromGraph
###### Methods
- `__init__(self, phenotype, depth=4, inputs=2, outputs=1)`: create the neural network with the correc number of inputs and outputs starting from a phenotype.
- `forward(self, obs)`: propagate information from the input nodes to the output nodes by doing a weighted sum of the input.

#### Evolution
###### Methods
- `__init__(self, population_size=400, generations=200, exchange_rate=0.01, mutation_rate=0.005, inputs=2)`: set class variables.
- `initialize_population(self)`: create the initial population.
- `create_individual(self)`: create an individual with random initial tag.
- `evolve(self)`:  evolve the population controlling generations. Return the best individual.
- `get_offspring(self)`: make each individual reproduce with a random partner. There is a small chance for the offspring to mutate. Return the entire offspring.
- `select_best(self, population)`: select the best `self.population_size` individuals, discard the others.
- `crossover(self, parent1, parent2)`: perform crossover between two parents.
- `get_children(self, parent1, parent2)`: create two children starting from two parents
- `mutate(self, genome)`: mutate symbols of the individual
### Tasks
#### XOR
- `compute_fitness(individual)`: return the fitness score, which is the number of correct guesses out of all possible combinations of inputs, divided by the total. 
- `compute_fitness_information_formula(individual)`: compute the xor for each combination of 2 inputs and return the fitness score: $$fitness = 0.85 \frac{\tau(f_{nn}, f_{target})}{\tau(f_{target}, f_{target})}+0.15*t$$
	- $\tau$ is the mutual information `sklearn.metrics.mutual_info_score(f_nn, f_target)`
	- `t` is 0 + 0.5 if the input layer has the correct size + 0.5 if the output layer has the correct size
- `compute_fitness_target(individual, print_info=False)`: return the similarity score between `individual` and a network that computes XOR. The score is computed using the following formula: $$fitness = \frac{s_{depth} + s_{nodes} + s_{tags} + s_{successors}}{4}$$
	- $s_{depth} = 1 - \frac{|depth_{individual} - depth_{target}|}{\max(depth_{individual}, depth_{target})}$ 
	- $s_{nodes} = 1 - \frac{|nodes_{individual} - nodes_{target}|}{\max(nodes_{individual}, nodes_{target})}$ 
	- $s_{tags}= \frac{tags_{common} - \Delta_{tags}}{tags_{target}}$
	- $s_{successors}= \frac{successors_{common} - \Delta_{successors}}{successors_{target}}$
#### Cartpole
- `compute_fitness(individual)`: return the average score of the individual over 5 trials with 200 maximum steps.
#### Parity
- `compute_fitness(individual, n=2)`: computes `sklearn.normalized_mutual_info_score` between the output of the NN and the label. The label is 1 if there is an odd number of inputs that are 1, 0 otherwise.
### Utils
#### Counter
###### Constants
- `_counter`: the last id that has been returned.
###### Methods
- `next(cls)`: increment the counter and return the value.
- `next_str(cls)`: increment the counter and return the value as a string.
#### Visualizer
- `__init__(self, inputs=2, outputs=1)`: set class variables.
- `_calculate_node_positions(self, structure)`: find node positions for printing the NN.
- `print_innovative_networks(self, innovative_individuals, save=False)`: print the NNs that caused a jump in fitness and their genome.
- `treelib_to_nx(self, tree, node_labels)`: convert a `tree` to a `nx.DiGraph`.
- `print_population(self, population)`: print the genome of each individual in the population.
- `plot_fitness_history(self, history, save=False)`: plot the best fitness for each generation.
- `print_phenotype(self, phenotype, save=False)`: print `phenotype`.
- `save_file_with_name(self, name)`: save file specifying a name.
- `print_no_position(self, phenotype)`: print without fixed positions. Useful for not connected components.