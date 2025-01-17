import json
import numpy as np
import matplotlib.pyplot as plt

paths = ['logs/lunar_lander_new.json']

# Load the data from the JSON file
with open(paths[0], 'r') as f:
    data = json.load(f)

# Initialize an array to store the average best fitness for each iteration
average_best_fitness = np.zeros(200)

# Iterate over each run
for run in data:
    # Iterate over each iteration and accumulate the best fitness
    for i, iteration in enumerate(run):
        average_best_fitness[i] += iteration['best_fitness']

# Compute the average best fitness by dividing by the number of runs
average_best_fitness /= len(data)

# Plot the results
plt.plot(average_best_fitness)
plt.xlabel('Iteration')
plt.ylabel('Average Best Fitness')
plt.title('Average Best Fitness over Iterations')
plt.show()
