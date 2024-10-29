from core.evolution import Evolution
from utils.visualizer import Visualizer
import os
from datetime import datetime
import json


def run():
    clear_console()
    inputs1 = 3
    inputs2 = 5
    iterations = 10
    log = []
    populations = []

    log, populations = evolve_stage(inputs=inputs1, iterations=iterations, log=log)
    log, populations = evolve_stage(inputs=inputs2, iterations=iterations, log=log, populations=populations)

    save(log)


def evolve_stage(inputs, iterations, log, populations=None):
    if populations is None:
        populations = [None] * iterations

    for i in range(iterations):
        print(f'Individual {i + 1} with {inputs} inputs:')
        evolution = Evolution(inputs=inputs, population=populations[i])
        best_individual, generations = evolution.evolve()

        log.append({
            'iteration': i,
            'inputs': inputs,
            'generations': generations,
            'log': evolution.logs
        })

        populations[i] = evolution.population

        clear_console()

    return log, populations


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
