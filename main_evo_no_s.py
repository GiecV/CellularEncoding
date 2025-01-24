from core.evo_no_s import Evolution
import os
from datetime import datetime
import json
import torch

torch.set_num_threads(1)

inputs = [2]  # [3,6]
iterations = 5
gen_budget = [100] * iterations


def run():
    """
    Execute the evolution stages for specified input and generation configurations.

    This function clears the console, sets up parameters for multiple evolution stages, and iteratively
    calls the `evolve_stage` function for different input and generation combinations. It logs the results
    of each evolution stage and saves the final log.

    Args:
        None

    Returns:
        None
    """
    clear_console()
    log = []
    populations = []

    for input in inputs:
        log, populations = evolve_stage(
            ins=input, iterations=iterations, log=log, pops=populations)
    save(log)


def evolve_stage(ins, iterations, log, pops=None):
    """
    Conduct a series of evolution stages for a specified number of iterations.

    This function initializes the evolution process for a given number of individuals and generations,
    creating an instance of the Evolution class for each individual. It logs the results of each iteration
    and updates the population for future iterations.

    Args:
        ins (int): The number of inputs for the evolution process.
        iterations (int): The number of iterations to perform.
        gen (int): The number of generations for each evolution instance.
        log (list): A list to store logs of the evolution process.
        pops (list, optional): An optional list of populations for the iterations. Defaults to None.

    Returns:
        tuple: A tuple containing the updated log and population lists.
    """
    if pops == []:
        pops = [None] * iterations

    for i in range(iterations):
        print(f'Individual {i + 1} with {ins} inputs:')
        evolution = Evolution(
            inputs=ins, population=pops[i], generations=gen_budget[i])
        best_individual = evolution.evolve(index=i)

        log.append({
            'iteration': i,
            'inputs': ins,
            'log': evolution.logs,
            # 'lineage': evolution.lineage,
            'individuals': evolution.saved_individuals
        })

        gen_budget[i] -= len(evolution.logs)
        print(len(evolution.logs))
        pops[i] = evolution.population
        clear_console()

    return log, pops


def save(item):
    """
    Save the provided item to a JSON file with a timestamped filename.

    This function creates a directory for logs if it does not already exist and writes the given item
    to a JSON file, naming the file with the current timestamp. This allows for organized storage of
    logs over time.

    Args:
        item: The data to be saved in JSON format.

    Returns:
        None
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join('logs', f'log_{timestamp}.json')
    os.makedirs('logs', exist_ok=True)
    with open(log_filename, 'w') as log_file:
        json.dump(item, log_file, indent=4)


def clear_console():
    """
    Clear the console screen based on the operating system.

    This function detects the operating system type and executes the appropriate command to clear
    the console screen. It helps maintain a clean output during the execution of programs.

    Args:
        None

    Returns:
        None
    """
    if os.name == 'posix':
        os.system('clear')
    elif os.name == 'nt':
        os.system('cls')


if __name__ == "__main__":
    run()
