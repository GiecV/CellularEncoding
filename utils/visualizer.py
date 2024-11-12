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

    @classmethod
    def __init__(cls, inputs=2, outputs=1):
        """
        Initialize the Visualizer with the number of inputs and outputs.

        Args:
            inputs (int, optional): The number of input nodes. Default is 2.
            outputs (int, optional): The number of output nodes. Default is 1.
        """
        cls.inputs = inputs
        cls.outputs = outputs

    @classmethod
    def _calculate_node_positions(cls, structure):
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

    @classmethod
    def print_innovative_networks(cls, innovative_individuals, save=False):
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
            nn = NNFromGraph(phenotype, inputs=cls.inputs, outputs=cls.outputs)

            # Neural network plot (col 0)
            pos = cls._calculate_node_positions(nn.phenotype.structure)
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
                tree_graph = cls.treelib_to_nx(tree, node_labels)

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
            cls.save_file_with_name('innovative_networks_')
        plt.show()

    @classmethod
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

    @classmethod
    def print_population(cls, population):
        """
        Print the details of each individual in the population.

        Args:
            population (list): A list of individuals in the population.
        """
        for individual in population:
            individual.print()

    @classmethod
    def plot_fitness_history(cls, history, save=False):
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
            cls.save_file_with_name('fitness_history_')
        plt.show()

    @classmethod
    def print_phenotype(cls, phenotype, save=False):
        """
        Print and optionally save the phenotype structure.

        Args:
            phenotype (Phenotype): The phenotype to print.
            save (bool, optional): Whether to save the plot. Default is False.
        """
        try:
            pos = cls._calculate_node_positions(phenotype.structure)

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
                cls.save_file_with_name('phenotype_')
            plt.show()
        except Exception:
            print("Cannot set positions")
            cls.print_no_position(phenotype)

    @classmethod
    def save_file_with_name(cls, name):
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

    @classmethod
    def plot_all_runs(cls, json_file_path, save=False):
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
        plt.title('Fitness, 10 Runs')
        plt.legend()

        if save:
            cls.save_file_with_name('runs_')
        plt.show()

    @classmethod
    def plot_all_times(cls, json_file_path, save=False):
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

                    fitness = generation['generation_time']
                    scores[iteration].append(fitness)

        for iteration, fitness_values in scores.items():
            plt.plot(fitness_values, label=f'Run {iteration}')

        plt.xlabel('Generation')
        plt.ylabel('Time (s)')
        plt.title('Time, 10 Runs')
        plt.legend()

        if save:
            cls.save_file_with_name('runs_')
        plt.show()

    @classmethod
    def save_lineage(cls, json_path, show=False):
        """
        Print and optionally save the innovative neural networks.

        Args:
            innovative_individuals (list): A list of innovative individuals.
            save (bool, optional): Whether to save the plots. Default is False.
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
                pos = cls._calculate_node_positions(nn.phenotype.structure)
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
                    tree_graph = cls.treelib_to_nx(tree, node_labels)

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
            cls.save_file_with_name('innovative_networks_')
            if show:
                plt.show()

    @classmethod
    def plot_times(cls, json_file_paths, save=False):
        """
        Plot the time taken for each generation for multiple JSON files.

        Args:
            json_file_paths (list): A list of paths to JSON files.
            save (bool, optional): Whether to save the plot. Default is False.
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
            cls.save_file_with_name('runs_')
        plt.show()

    @classmethod
    def plot_avg_fitness(cls, json_file_paths, save=False):
        """
        Plot the average fitness over generations with standard deviation for multiple JSON files.

        Args:
            json_file_paths (list): A list of paths to JSON files.
            save (bool, optional): Whether to save the plot. Default is False.
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
            cls.save_file_with_name('fitness_multiple_files_')
        plt.show()

    @classmethod
    def create_boxplots(cls, json_file_paths, save=False):
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

        color_map = {'logs/6i.json': '#f2a964', 'logs/36i.json': '#709ec7',
                     'logs/23456i.json': '#7fbb74'}
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
                            width=0.35, capsize=5, edgecolor='black')

                    # Annotate each bar with mean ± std deviation
                    plt.text(current_pos + pos_offset, mean_val + std_dev + 2, f"{mean_val:.1f}±{std_dev:.1f}",
                             ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

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
        plt.legend(handles, names, title="Files", loc="upper left")

        plt.xticks(x_positions, x_labels)
        # Custom legend and labels
        plt.xlabel("Number of Inputs")
        plt.ylabel("Avg Generations ± Std Dev")
        plt.title("Average Generations to Achieve the Optimum at Each Stage")

        # Show plot
        plt.show()
