from tasks.maze_task import compute_fitness
from core.genome import Genome

individual = Genome()
fitness = compute_fitness(individual)
print(fitness)