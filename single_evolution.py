from core.evolution import Evolution
from utils.visualizer import Visualizer
import os
from datetime import datetime
import json


def run():
    clear_console()
    inputs = 6
    iterations = 10
    log = []
    populations = []

    log, populations = evolve_stage(
        ins=inputs, iterations=iterations, gen=250, log=log)

    save(log)


def evolve_stage(ins, iterations, gen, log, pops=None):
    if pops is None:
        pops = [None] * iterations

    for i in range(iterations):
        print(f'Individual {i + 1} with {ins} inputs:')
        evolution = Evolution(inputs=ins, population=pops[i], generations=gen)
        best_individual = evolution.evolve()

        log.append({
            'iteration': i,
            'inputs': ins,
            'log': evolution.logs
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
