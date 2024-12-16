from tasks.stepping_gates import compute_fitness

from core.genome import Genome
from core.phenotype import Phenotype
from core.neural_network_from_graph import NNFromGraph

g = Genome()

fitness = compute_fitness(individual=g, max_gate=1)
print(fitness)
