from utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import os
import json
import numpy as np

json_file_paths = ['logs/real_acrobot.json']

v = Visualizer()

if not isinstance(json_file_paths, list):
    raise ValueError("json_file_paths should be a list of file paths.")

for json_file_path in json_file_paths:
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(
            f"The file {json_file_path} does not exist.")

with open(json_file_path, 'r') as file:
    data = json.load(file)

scores = {}

for run in data:
    inputs = run['inputs']
    iteration = run['iteration']

    if iteration not in scores:
        scores[iteration] = []

    for generation in run['log']:
        fitness = generation['best_score']
        scores[iteration].append(fitness)

max_length = max(len(fitness_values)
                    for fitness_values in scores.values())
for iteration in scores:
    last_value = scores[iteration][-1]
    scores[iteration].extend(
        [last_value] * (max_length - len(scores[iteration])))

avg_fitness = [sum(fitness_values[i] for fitness_values in scores.values()) / len(scores)
                for i in range(max_length)]
std_dev = [np.std([values[i] for values in scores.values()])
            for i in range(max_length)]

plt.plot(avg_fitness, label='Policy Network')
plt.fill_between(range(max_length),
                    [avg_fitness[i] - std_dev[i]
                        for i in range(max_length)],
                    [avg_fitness[i] + std_dev[i]
                        for i in range(max_length)],
                    alpha=0.2)

plt.axhline(y=-60, color='red', linestyle='--', label='Optimum Value')

plt.grid(True)
#plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=))
#plt.ylim(-95, -55)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Average Fitness per Generation')
plt.legend()
plt.show()