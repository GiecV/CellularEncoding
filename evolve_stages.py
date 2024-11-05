from core.evolution import Evolution
import os
from datetime import datetime
import json


def run():
    clear_console()
    inputs = [3, 5]
    iterations = 10
    generations = [80, 200]
    log = []
    populations = []

    for input, generation in zip(inputs, generations):
        log, populations = evolve_stage(
            ins=input, iterations=iterations, gen=generation, log=log, pops=populations)

    save(log)


def evolve_stage(ins, iterations, gen, log, pops=None):
    if pops == []:
        pops = [None] * iterations

    for i in range(iterations):
        print(f'Individual {i + 1} with {ins} inputs:')
        evolution = Evolution(inputs=ins, population=pops[i], generations=gen)
        best_individual = evolution.evolve(max_time=3600)

        log.append({
            'iteration': i,
            'inputs': ins,
            'log': evolution.logs,
            'lineage': evolution.lineage
        })

        pops[i] = evolution.population
        clear_console()

    return log, pops


def save(item):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join('logs', f'log_{timestamp}.json')
    os.makedirs('logs', exist_ok=True)
    with open(log_filename, 'w') as log_file:
        json.dump(item, log_file, indent=4)


def clear_console():
    if os.name == 'posix':
        os.system('clear')
    elif os.name == 'nt':
        os.system('cls')


if __name__ == "__main__":
    run()