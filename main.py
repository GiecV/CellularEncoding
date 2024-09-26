from genome import Genome
from phenotype import Phenotype
from neural_network_from_graph import NNFromGraph
import torch

# Create a new genome
g = Genome()

# Hardcoded mutations
g.change_symbol(level=0, node_id='0' + str(0), symbol="r")
g.change_symbol(level=0, node_id='0' + str(1), symbol="t")
g.change_symbol(level=0, node_id='0' + str(2), symbol="n1")
g.change_symbol(level=0, node_id='0' + str(3), symbol="n1")
g.change_symbol(level=1, node_id='0' + str(0), symbol="a")
g.change_symbol(level=1, node_id='0' + str(1), symbol="w")
g.change_symbol(level=1, node_id='0' + str(2), symbol="a")
g.change_symbol(level=1, node_id='0' + str(3), symbol="n1")
g.change_symbol(level=1, node_id='0' + str(4), symbol="n1")
g.change_symbol(level=1, node_id='0' + str(5), symbol="n1")
g.change_symbol(level=2, node_id='0' + str(0), symbol="k2")
g.change_symbol(level=2, node_id='0' + str(1), symbol="g")
g.change_symbol(level=2, node_id='0' + str(2), symbol="p")
g.change_symbol(level=2, node_id='0' + str(3), symbol="h")
g.change_symbol(level=2, node_id='0' + str(4), symbol="l")
g.change_symbol(level=2, node_id='0' + str(5), symbol="d0")
g.change_symbol(level=2, node_id='0' + str(6), symbol="r")
g.change_symbol(level=2, node_id='0' + str(7), symbol="p")
g.change_symbol(level=2, node_id='0' + str(8), symbol="l")
g.change_symbol(level=2, node_id='0' + str(9), symbol="u")
g.change_symbol(level=2, node_id='0' + str(10), symbol="l")
g.change_symbol(level=2, node_id='0' + str(11), symbol="p")
g.change_symbol(level=2, node_id='0' + str(12), symbol="d0")
g.change_symbol(level=2, node_id='0' + str(13), symbol="d0")
g.change_symbol(level=2, node_id='0' + str(14), symbol="l")
g.change_symbol(level=2, node_id='0' + str(15), symbol="l")

# Print the resulting trees
g.print()

# Create a new phenotype with the mutated genotype
p = Phenotype(genome=g)

i = 0
while not p.development_finished():
    print(f"Iteration {i}")
    p.develop()
    i += 1

# Expand the inputs and outputs of the network
# p.expand_inputs_and_outputs()
p.expand_some_inputs_and_outputs(2, 2)

# Show the resulting network
try:
    p.print()
except:
    print("Graph is not connected")
    p.print_no_position()  # Show the network without caring about highlighting layers, this is easier in the case of not connected components

# Create a neural network from the graph
model = NNFromGraph(p.structure)

# Print the output of the neural network
print(model.forward(torch.tensor([1, 0], dtype=torch.float)))
