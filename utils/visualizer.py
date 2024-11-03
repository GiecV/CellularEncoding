from core.phenotype import Phenotype
from core.neural_network_from_graph import NNFromGraph
from core.genome import Genome
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime
import json
import numpy as np


class Visualizer:

    def __init__(self, inputs=2, outputs=1):
        """
        Initialize the Visualizer with the number of inputs and outputs.

        Args:
            inputs (int, optional): The number of input nodes. Default is 2.
            outputs (int, optional): The number of output nodes. Default is 1.
        """
        self.inputs = inputs
        self.outputs = outputs

    def _calculate_node_positions(self, structure):
        """
        Calculate the positions of nodes in the neural network graph.

        Args:
            structure (networkx.DiGraph): The graph structure of the neural network.

        Returns:
            dict: A dictionary mapping nodes to their positions.
        """
        start_nodes = [
            node for node in structure.nodes if node.startswith("I")]
        pos = {node: (i / (len(start_nodes) + 1), 1.0)
               for i, node in enumerate(start_nodes)}

        levels = defaultdict(list)
        queue = deque((node, 0) for node in start_nodes)
        visited = set()

        while queue:
            node, level = queue.popleft()
            if node not in visited:
                visited.add(node)
                levels[level].append(node)
                queue.extend((successor, level + 1)
                             for successor in structure.successors(node))

        pos |= {
            node: (
                i / (len(nodes) + 1),
                0.0 if node.startswith("O") else 1 -
                (level / (len(levels) + 1)),
            )
            for level, nodes in levels.items()
            for i, node in enumerate(nodes)
        }

        return pos

    def print_innovative_networks(self, innovative_individuals, save=False):
        """
        Print and optionally save the innovative neural networks.

        Args:
            innovative_individuals (list): A list of innovative individuals.
            save (bool, optional): Whether to save the plots. Default is False.
        """
        num_individuals = len(innovative_individuals)
        rows = num_individuals
        cols = 4  # 1 for NN + 3 for trees

        fig = plt.figure(figsize=(13, rows * 6))
        gs = GridSpec(rows, cols, figure=fig)

        for idx, (individual, fitness_score) in enumerate(innovative_individuals):
            phenotype = Phenotype(individual)
            nn = NNFromGraph(phenotype, inputs=self.inputs,
                             outputs=self.outputs)

            # Neural network plot (col 0)
            pos = self._calculate_node_positions(nn.phenotype.structure)
            node_labels = {node: f"{node}[{nn.phenotype.structure.nodes[node]['threshold']}]"
                           for node in nn.phenotype.structure.nodes}

            ax_nn = fig.add_subplot(gs[idx, 0])
            nx.draw(nn.phenotype.structure, pos, labels=node_labels, with_labels=True, node_size=500,
                    node_color="skyblue", font_size=10, font_weight="bold", arrows=True, ax=ax_nn)
            labels = nx.get_edge_attributes(nn.phenotype.structure, 'weight')
            nx.draw_networkx_edge_labels(
                nn.phenotype.structure, pos, edge_labels=labels, ax=ax_nn)
            ax_nn.set_title(f'Neural Network\nFitness Score: {
                            "{:.2f}".format(fitness_score)}', fontsize=14)

            # Genome: visualizing 3 trees (cols 1, 2, 3)
            genome = individual.get_trees()  # Assuming the genome is a list of 3 trees
            for tree_idx, tree in enumerate(genome):
                # tree_idx + 1 to skip the NN column
                ax_tree = fig.add_subplot(gs[idx, tree_idx + 1])

                node_labels = {}

                # Convert treelib tree to NetworkX graph
                tree_graph = self.treelib_to_nx(tree, node_labels)

                # Layout for tree graph
                tree_pos = nx.bfs_layout(
                    tree_graph, tree.root, align='horizontal')  # type: ignore
                max_y = max(y for x, y in tree_pos.values())
                tree_pos = {node: (x, max_y - y)
                            for node, (x, y) in tree_pos.items()}

                # Plot the tree as a NetworkX graph
                nx.draw(tree_graph, pos=tree_pos, labels=node_labels, with_labels=True, node_size=500, node_color="lightgreen",
                        font_size=10, font_weight="bold", arrows=False, ax=ax_tree)
                ax_tree.set_title(f'Tree {tree_idx + 1}', fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.8)
        if save:
            self.save_file_with_name('innovative_networks_')
        plt.show()

    def treelib_to_nx(self, tree, node_labels):
        """
        Convert a tree structure from treelib to a NetworkX directed graph.

        Args:
            tree (treelib.Tree): The tree structure to convert.
            node_labels (dict): A dictionary to store node labels.

        Returns:
            networkx.DiGraph: The converted directed graph.
        """
        graph = nx.DiGraph()  # Create a directed graph

        def add_edges(node):
            # Use children() to get the child nodes
            for child in tree.children(node.identifier):
                # Add edge from parent to child
                graph.add_edge(node.identifier, child.identifier)
                node_labels[child.identifier] = child.tag
                add_edges(child)  # Recurse on the child node

        # Start the recursive process from the root
        node_labels[tree.root] = tree.get_node(tree.root).tag
        add_edges(tree[tree.root])
        if len(tree.nodes) == 1:
            root_node = tree[tree.root]
            node_labels[root_node.identifier] = root_node.tag
            graph.add_node(root_node.identifier)
        return graph

    def print_population(self, population):
        """
        Print the details of each individual in the population.

        Args:
            population (list): A list of individuals in the population.
        """
        for individual in population:
            individual.print()

    def plot_fitness_history(self, history, save=False):
        """
        Plot the fitness history over generations.

        Args:
            history (list): A list of fitness values over generations.
            save (bool, optional): Whether to save the plot. Default is False.
        """
        generations = list(range(1, len(history) + 1))
        plt.plot(generations, history, marker='o')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness History Over Generations')
        plt.grid(True)
        if save:
            self.save_file_with_name('fitness_history_')
        plt.show()

    def print_phenotype(self, phenotype, save=False):
        """
        Print and optionally save the phenotype structure.

        Args:
            phenotype (Phenotype): The phenotype to print.
            save (bool, optional): Whether to save the plot. Default is False.
        """
        try:
            pos = self._calculate_node_positions(phenotype.structure)

            node_labels = {node: f"{node}[{phenotype.structure.nodes[node]['threshold']}]"
                           for node in phenotype.structure.nodes}

            nx.draw(
                phenotype.structure,
                pos,
                labels=node_labels,
                with_labels=True,
                node_size=500,
                node_color="skyblue",
                font_size=10,
                font_weight="bold",
                arrows=True,
            )
            labels = nx.get_edge_attributes(phenotype.structure, 'weight')
            nx.draw_networkx_edge_labels(
                phenotype.structure, pos, edge_labels=labels)
            if save:
                self.save_file_with_name('phenotype_')
            plt.show()
        except Exception:
            print("Cannot set positions")
            self.print_no_position(phenotype)

    def save_file_with_name(self, name):
        """
        Save the current plot with a given name.

        Args:
            name (str): The base name for the file to save.
        """
        os.makedirs('img', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join('img', f'{name}{timestamp}.png'))

    def print_no_position(self, phenotype):
        nx.draw(phenotype.structure, with_labels=True)
        plt.show()

    def plot_average_fitness(self, json_file_path, save=False):
        """
        Plot the average fitness over generations with standard deviation.

        Args:
            scores (dict): A dictionary containing fitness scores for each run.
            max_length (int): The maximum length of the generations.
            save (bool, optional): Whether to save the plot. Default is False.
        """
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
            # scores[iteration].extend(
            #     [1] * (max_length - len(scores[iteration])))
            last_value = scores[iteration][-1]
            scores[iteration].extend(
                [last_value] * (max_length - len(scores[iteration])))

        avg_fitness = [sum(fitness_values[i] for fitness_values in scores.values()) / len(scores)
                       for i in range(max_length)]
        std_dev = [np.std([values[i] for values in scores.values()])
                   for i in range(max_length)]

        plt.plot(avg_fitness, label='3i5i')
        plt.fill_between(range(max_length),
                         [avg_fitness[i] - std_dev[i]
                             for i in range(max_length)],
                         [avg_fitness[i] + std_dev[i]
                             for i in range(max_length)],
                         color='b', alpha=0.2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Average Fitness, 10 Runs')
        plt.legend()

        if save:
            self.save_file_with_name('fitness_')
        plt.show()

    def plot_all_runs(self, json_file_path, save=False):
        """
        Plot the fitness of all runs from a JSON file.

        Args:
            json_file_path (str): The path to the JSON file containing the run data.
            save (bool, optional): Whether to save the plot. Default is False.

        Raises:
            FileNotFoundError: If the JSON file does not exist.
        """
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

        for iteration, fitness_values in scores.items():
            plt.plot(fitness_values, label=f'Run {iteration}')

        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Average Fitness, 10 Runs')
        plt.legend()

        if save:
            self.save_file_with_name('runs_')
        plt.show()

    def plot_time(self, json_file_path, save=False):
        """
        Plot the time taken for each generation.

        Args:
            time_data (list): A list of time values for each generation.
            save (bool, optional): Whether to save the plot. Default is False.
        """
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
                time = generation['generation_time']
                scores[iteration].append(time)

        max_length = max(len(fitness_values)
                         for fitness_values in scores.values())
        for iteration in scores:
            last_value = scores[iteration][-1]
            scores[iteration].extend(
                [last_value] * (max_length - len(scores[iteration])))

        avg_time = [sum(values[i] for values in scores.values()) / len(scores)
                    for i in range(max_length)]
        std_dev = [np.std([values[i] for values in scores.values()])
                   for i in range(max_length)]

        plt.plot(avg_time, label='3i5i')
        plt.fill_between(range(max_length),
                         [avg_time[i] - std_dev[i] for i in range(max_length)],
                         [avg_time[i] + std_dev[i] for i in range(max_length)],
                         color='b', alpha=0.2)
        plt.xlabel('Generation')
        plt.ylabel('Time (s)')
        plt.title('Average Time, 10 Runs')
        plt.legend()

        if save:
            self.save_file_with_name('time_')
        plt.show()

    def print_lineage(self, json_individuals, save=False):
        """
        Print and optionally save the innovative neural networks.

        Args:
            innovative_individuals (list): A list of innovative individuals.
            save (bool, optional): Whether to save the plots. Default is False.
        """
        sorted_individuals = sorted(
            json_individuals, key=lambda x: x['generation'], reverse=True)
        for i, individual in enumerate(sorted_individuals):
            genome = Genome()
            genome.from_json_pickle(individual)
            sorted_individuals[i] = (genome, individual['generation'])

        num_individuals = len(json_individuals)
        rows = num_individuals
        cols = 4  # 1 for NN + 3 for trees

        fig = plt.figure(figsize=(13, rows * 6))
        gs = GridSpec(rows, cols, figure=fig)

        for idx, (individual, generation) in enumerate(sorted_individuals):
            phenotype = Phenotype(individual)
            nn = NNFromGraph(phenotype, inputs=self.inputs,
                             outputs=self.outputs)

            # Neural network plot (col 0)
            pos = self._calculate_node_positions(nn.phenotype.structure)
            node_labels = {node: f"{node}[{nn.phenotype.structure.nodes[node]['threshold']}]"
                           for node in nn.phenotype.structure.nodes}

            ax_nn = fig.add_subplot(gs[idx, 0])
            nx.draw(nn.phenotype.structure, pos, labels=node_labels, with_labels=True, node_size=500,
                    node_color="skyblue", font_size=10, font_weight="bold", arrows=True, ax=ax_nn)
            labels = nx.get_edge_attributes(nn.phenotype.structure, 'weight')
            nx.draw_networkx_edge_labels(
                nn.phenotype.structure, pos, edge_labels=labels, ax=ax_nn)
            ax_nn.set_title(f'Neural Network\nFitness Score: {
                            "{:.2f}".format(generation)}', fontsize=14)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.8)
        if save:
            self.save_file_with_name('innovative_networks_')
        plt.show()
