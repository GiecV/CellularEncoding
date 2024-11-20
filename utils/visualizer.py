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
    """
    A class for visualizing neural networks, phenotypes, and evolutionary runs.

    This class provides methods to visualize neural network structures, the
    phenotype of individuals, and data from evolutionary runs. It supports
    various types of plots including network graphs, tree visualizations,
    and fitness over time.

    :param inputs: The number of input nodes in the neural network. Default is 2.
    :type inputs: int, optional
    :param outputs: The number of output nodes in the neural network. Default is 1.
    :type outputs: int, optional
    """

    def __init__(self, inputs=2, outputs=1):
        """
        Initialize the Visualizer with the number of inputs and outputs.

        :param inputs: The number of input nodes. Default is 2.
        :type inputs: int, optional
        :param outputs: The number of output nodes. Default is 1.
        :type outputs: int, optional
        """
        self.inputs = inputs
        self.outputs = outputs

    def _calculate_node_positions(self, structure):
        """
        Calculate the positions of nodes in the neural network graph.

        :param structure: The graph structure of the neural network.
        :type structure: networkx.DiGraph
        :return: A dictionary mapping nodes to their positions.
        :rtype: dict
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

        :param innovative_individuals: A list of innovative individuals.
        :type innovative_individuals: list
        :param save: Whether to save the plots. Default is False.
        :type save: bool, optional
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
            ax_nn.set_title(f'Fitness Score: {"{:.2f}".format(fitness_score)}')

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

        :param tree: The tree structure to convert.
        :type tree: treelib.Tree
        :param node_labels: A dictionary to store node labels.
        :type node_labels: dict
        :return: The converted directed graph.
        :rtype: networkx.DiGraph
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

        :param population: A list of individuals in the population.
        :type population: list
        """
        for individual in population:
            individual.print()

    def print_phenotype(self, phenotype, save=False):
        """
        Print and optionally save the phenotype structure.

        :param phenotype: The phenotype to print.
        :type phenotype: Phenotype
        :param save: Whether to save the plot. Default is False.
        :type save: bool, optional
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

        :param name: The base name for the file to save.
        :type name: str
        """
        os.makedirs('img', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join('img', f'{name}{timestamp}.png'))

    def print_no_position(self, phenotype):
        """
        Print the phenotype without positioning information.

        :param phenotype: The phenotype to print.
        :type phenotype: Phenotype
        """
        nx.draw(phenotype.structure, with_labels=True)
        plt.show()

    def plot_all_runs(self, json_file_path, save=False):
        """
        Plot the fitness of all runs from a JSON file.

        :param json_file_path: The path to the JSON file containing the run data.
        :type json_file_path: str
        :param save: Whether to save the plot. Default is False.
        :type save: bool, optional
        :raises FileNotFoundError: If the JSON file does not exist.
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
        plt.title('Fitness, 10 Runs')
        plt.legend()

        if save:
            self.save_file_with_name('runs_')
        plt.show()

    def plot_all_times(self, json_file_path, save=False):
        """
        Plot the fitness of all runs from a JSON file.

        :param json_file_path: The path to the JSON file containing the run data.
        :type json_file_path: str
        :param save: Whether to save the plot. Default is False.
        :type save: bool, optional
        :raises FileNotFoundError: If the JSON file does not exist.
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

                    fitness = generation['generation_time']
                    scores[iteration].append(fitness)

        for iteration, fitness_values in scores.items():
            plt.plot(fitness_values, label=f'Run {iteration}')

        plt.xlabel('Generation')
        plt.ylabel('Time (s)')
        plt.title('Time, 10 Runs')
        plt.legend()

        if save:
            self.save_file_with_name('runs_')
        plt.show()

    def save_lineage(self, json_path, show=False):
        """
        Print and optionally save the innovative neural networks.

        :param json_path: Path to the JSON file containing lineage data.
        :type json_path: str
        :param show: Whether to display the plots. Defaults to False.
        :type show: bool, optional
        """

        with open(json_path, 'r') as file:
            json_file = json.load(file)

        cols = 4  # 1 for NN + 3 for trees

        for run in json_file:
            json_individuals = run['lineage']

            for i, individual in enumerate(json_individuals):
                genome = Genome()
                genome.from_json_pickle(individual)
                json_individuals[i] = (genome, individual['generation'])

            num_individuals = len(json_individuals)
            rows = num_individuals
            fig = plt.figure(figsize=(13, rows * 6))
            gs = GridSpec(rows, cols, figure=fig)

            for idx, (individual, generation) in enumerate(json_individuals):
                phenotype = Phenotype(individual)
                nn = NNFromGraph(phenotype, inputs=run['inputs'],
                                 outputs=1)

                # Neural network plot (col 0)
                pos = self._calculate_node_positions(nn.phenotype.structure)
                node_labels = {node: f"{node}[{nn.phenotype.structure.nodes[node]['threshold']}]"
                               for node in nn.phenotype.structure.nodes}

                ax_nn = fig.add_subplot(gs[idx, 0])
                nx.draw(nn.phenotype.structure, pos, labels=node_labels, with_labels=True, node_size=500,
                        node_color="skyblue", font_size=10, font_weight="bold", arrows=True, ax=ax_nn)
                labels = nx.get_edge_attributes(
                    nn.phenotype.structure, 'weight')
                nx.draw_networkx_edge_labels(
                    nn.phenotype.structure, pos, edge_labels=labels, ax=ax_nn)
                ax_nn.set_title(f'Generation: {generation}', fontsize=14)

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
            self.save_file_with_name('innovative_networks_')
            if show:
                plt.show()

    def plot_times(self, json_file_paths, save=False):
        """
        Plot the time taken for each generation for multiple JSON files.

        :param json_file_paths: A list of paths to JSON files.
        :type json_file_paths: list of str
        :param save: Whether to save the plot. Defaults to False.
        :type save: bool, optional
        :raises ValueError: If `json_file_paths` is not a list of file paths.
        :raises FileNotFoundError: If any of the specified JSON files do not exist.
        """
        if not isinstance(json_file_paths, list):
            raise ValueError("json_file_paths should be a list of file paths.")

        for json_file_path in json_file_paths:
            if not os.path.exists(json_file_path):
                raise FileNotFoundError(
                    f"The file {json_file_path} does not exist.")

        for json_file_path in json_file_paths:
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
                last_value = sum(scores[iteration][-5:]) / 5
                scores[iteration].extend(
                    [last_value] * (max_length - len(scores[iteration])))

            avg_time = [sum(values[i] for values in scores.values()) / len(scores)
                        for i in range(max_length)]
            std_dev = [np.std([values[i] for values in scores.values()])
                       for i in range(max_length)]

            plt.plot(avg_time, label=os.path.basename(json_file_path))
            plt.fill_between(range(max_length),
                             [avg_time[i] - std_dev[i]
                                 for i in range(max_length)],
                             [avg_time[i] + std_dev[i]
                                 for i in range(max_length)],
                             alpha=0.2)

        plt.xlim(0, max_length)
        plt.ylim(0, 80)
        plt.xlabel('Generation')
        plt.ylabel('Average Time (s)')
        plt.title('Average Time per Generation')
        plt.legend()

        if save:
            self.save_file_with_name('runs_')
        plt.show()

    def plot_avg_fitness(self, json_file_paths, save=False):
        """
        Plot the average fitness over generations with standard deviation for multiple JSON files.

        :param json_file_paths: A list of paths to JSON files.
        :type json_file_paths: list of str
        :param save: Whether to save the plot. Defaults to False.
        :type save: bool, optional
        :raises ValueError: If `json_file_paths` is not a list of file paths.
        :raises FileNotFoundError: If any of the specified JSON files do not exist.
        """
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

            plt.plot(avg_fitness, label=os.path.basename(json_file_path))
            plt.fill_between(range(max_length),
                             [avg_fitness[i] - std_dev[i]
                                 for i in range(max_length)],
                             [avg_fitness[i] + std_dev[i]
                                 for i in range(max_length)],
                             alpha=0.2)

        plt.grid(True)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Average Fitness per Generation')
        plt.legend()
        if save:
            self.save_file_with_name('fitness_multiple_files_')
        plt.show()

    def create_boxplots(self, json_file_paths, save=False):
        """
        Creates boxplots to visualize the average number of generations required to achieve optimal results across different input configurations from multiple JSON files. This method aggregates data from the specified JSON files and generates a bar plot with error bars representing the standard deviation.

        :param json_file_paths: A list of paths to the JSON files containing the data.
        :type json_file_paths: list of str
        :param save: If True, saves the generated plot to a file. Defaults to False.
        :type save: bool
        :raises FileNotFoundError: If any of the specified JSON files do not exist.
        :examples:
            >>> Visualizer.create_boxplots(['logs/6i.json', 'logs/36i.json'], save=True)
        """
        all_data = {}

        for json_file_path in json_file_paths:
            if not os.path.exists(json_file_path):
                raise FileNotFoundError(
                    f"The file {json_file_path} does not exist.")

            with open(json_file_path, 'r') as file:
                all_data[json_file_path] = {}
                data = json.load(file)
                for run in data:
                    inputs = run['inputs']
                    generations = len(run['log'])
                    if inputs not in all_data[json_file_path]:
                        all_data[json_file_path][inputs] = []
                    all_data[json_file_path][inputs].append(generations)

        color_map = {'logs/6i.json': '#d9e7dc', 'logs/36i.json': '#eb3e56',
                     'logs/23456i.json': '#8296b5'}
        names = ['1 stage', '2 stages', '5 stages']

        # Prepare figure and axes
        plt.figure(figsize=(10, 6))

        sorted_keys = sorted(
            {key for sub_data in all_data.values() for key in sub_data}
        )

        # Set a position counter for each bar
        current_pos = 1  # Starting position for bar plots
        x_positions = []
        x_labels = []
        offset = 0.35

        for key in sorted_keys:
            bar_count = sum(key in sub_data for sub_data in all_data.values())
            pos_offset = -(bar_count - 1) * offset / 2
            for file, sub_data in all_data.items():
                if key in sub_data:
                    values = sub_data[key]

                    mean_val = np.mean(values)
                    std_dev = np.std(values)

                    plt.bar(current_pos + pos_offset, mean_val, yerr=std_dev, color=color_map[file],
                            width=0.3, capsize=5, edgecolor='black', zorder=3)

                    # Annotate each bar with mean ± std deviation
                    plt.text(current_pos + pos_offset, mean_val + std_dev + 2, f"{mean_val:.1f}±{std_dev:.1f}",
                             ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', zorder=4)

                    # Append x-axis position and corresponding key label
            # Only label the first bar in each group
                    if pos_offset == -(bar_count - 1) * offset / 2:
                        x_positions.append(current_pos)
                        x_labels.append(key)

                    pos_offset += offset

                    # Move to the next x position
            current_pos += 1

        plt.xticks(x_positions, x_labels)

        handles = [plt.Line2D([0], [0], color=color_map[file], lw=6)
                   for file in color_map]
        plt.legend(handles, names, title="Curriculums", loc="upper left")

        plt.xticks(x_positions, x_labels)
        # Custom legend and labels
        plt.xlabel("Number of Inputs")
        plt.ylabel("Avg Generations ± Std Dev")
        plt.title("Average Generations to Achieve the Optimum at Each Stage")

        plt.grid(axis='y', linestyle='-', linewidth=0.5, zorder=1)

        if save:
            self.save_file_with_name('boxplot_')

        # Show plot
        plt.show()

    def compute_averages(self, all_data):
        averages = {}
        std_devs = {}

        for file, data in all_data.items():
            # Find the maximum length of the lists in the file
            max_length = max(len(values) for values in data.values())

            # Initialize a list to store the summed values
            summed_values = [0] * max_length

            # Sum the values with different keys but the same index
            for values in data.values():
                for i, value in enumerate(values):
                    summed_values[i] += value

            # Compute the average for each index
            averages[file] = sum(summed_values) / len(summed_values)
            std_devs[file] = np.std(summed_values)

        return averages, std_devs

    def create_sum_boxplots(self, json_file_paths, save=False):
        """
        Generates boxplots to visualize the average number of generations required to achieve optimal results across different input configurations from multiple JSON files. This method aggregates data from the specified JSON files, computes averages and standard deviations, and creates a bar plot with error bars.

        :param json_file_paths: A list of paths to the JSON files containing the data.
        :type json_file_paths: list of str
        :param save: If True, saves the generated plot to a file. Defaults to False.
        :type save: bool
        :raises FileNotFoundError: If any of the specified JSON files do not exist.
        :examples:
            >>> Visualizer.create_sum_boxplots(['logs/6i.json', 'logs/36i.json'], save=True)
        """
        all_data = {}

        for json_file_path in json_file_paths:
            if not os.path.exists(json_file_path):
                raise FileNotFoundError(
                    f"The file {json_file_path} does not exist.")

            with open(json_file_path, 'r') as file:
                all_data[json_file_path] = {}
                data = json.load(file)
                for run in data:
                    inputs = run['inputs']
                    generations = len(run['log'])
                    if inputs not in all_data[json_file_path]:
                        all_data[json_file_path][inputs] = []
                    all_data[json_file_path][inputs].append(generations)

        print(all_data)
        averages, std_devs = self.compute_averages(all_data)

        color_map = {'logs/6i.json': '#d9e7dc', 'logs/36i.json': '#eb3e56',
                     'logs/23456i.json': '#8296b5'}
        names = ['1 stage', '2 stages', '5 stages']

        # Prepare figure and axes
        plt.figure(figsize=(6, 6))

        current_pos = 0
        for file in averages:
            current_pos += 0.35
            plt.bar(current_pos, averages[file], yerr=std_devs[file], color=color_map[file],
                    width=0.1, capsize=5, edgecolor='black', zorder=3)

            # Annotate each bar with mean ± std deviation
            plt.text(current_pos, averages[file] + std_devs[file] + 2, f"{averages[file]:.1f}±{std_devs[file]:.1f}",
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', zorder=4)

        handles = [plt.Line2D([0], [0], color=color_map[file], lw=6)
                   for file in color_map]
        plt.legend(handles, names, title="Curriculums", loc="upper right")

        plt.xticks([])

        # Custom legend and labels
        plt.xlabel("Number of Stages")
        plt.ylabel("Avg Generations ± Std Dev")
        plt.title("Average Generations to Achieve the Optimum at Each Stage")

        plt.grid(axis='y', linestyle='-', linewidth=0.5, zorder=1)

        if save:
            self.save_file_with_name('boxplot_')

        # Show plot
        plt.show()

        averages, std_devs = self.compute_averages(all_data)

    def save_best_networks(self, json_path, show=False):
        """
        Saves visual representations of the best neural networks and their associated trees from a JSON file. This method reads the JSON data, processes the neural networks and trees, and generates plots for each iteration.

        :param json_path: The path to the JSON file containing network data.
        :type json_path: str
        :param show: If True, displays the generated plots. Defaults to False.
        :type show: bool
        :returns: None
        :examples:
            >>> Visualizer.save_best_networks('path/to/json_file.json', show=True)
        """
        with open(json_path, 'r') as file:
            json_file = json.load(file)

            cols = 4  # 1 for NN + 3 for trees
            individuals = {}

        for run in json_file:
            json_individual = run['lineage'][0]

            genome = Genome()
            genome.from_json_pickle(json_individual)
            json_individual = genome

            if run['iteration'] not in individuals:
                individuals[run['iteration']] = []
            individuals[run['iteration']].append(
                (run['inputs'], json_individual))

        num_individuals = len(individuals[0])
        rows = num_individuals

        for iteration, individuals_list in individuals.items():
            fig = plt.figure(figsize=(13, rows * 6))
            gs = GridSpec(rows, cols, figure=fig)
            for idx, (ins, individual) in enumerate(individuals_list):
                phenotype = Phenotype(individual)
                nn = NNFromGraph(phenotype, inputs=ins,
                                 outputs=1)

                # Neural network plot (col 0)
                pos = self._calculate_node_positions(nn.phenotype.structure)
                node_labels = {node: f"{node}[{nn.phenotype.structure.nodes[node]['threshold']}]"
                               for node in nn.phenotype.structure.nodes}

                ax_nn = fig.add_subplot(gs[idx, 0])
                nx.draw(nn.phenotype.structure, pos, labels=node_labels, with_labels=True, node_size=500,
                        node_color="skyblue", font_size=10, font_weight="bold", arrows=True, ax=ax_nn)
                labels = nx.get_edge_attributes(
                    nn.phenotype.structure, 'weight')
                nx.draw_networkx_edge_labels(
                    nn.phenotype.structure, pos, edge_labels=labels, ax=ax_nn)
                ax_nn.set_title(f'Iteration: {iteration}', fontsize=14)

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
            self.save_file_with_name(f'champions_{iteration}')
            if show:
                plt.show()
