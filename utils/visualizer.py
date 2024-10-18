from core.phenotype import Phenotype
from core.neural_network_from_graph import NNFromGraph
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime


class Visualizer:

    def __init__(self, inputs=2, outputs=1):
        self.inputs = inputs
        self.outputs = outputs

    def _calculate_node_positions(self, structure):
        start_nodes = [node for node in structure.nodes if node.startswith("I")]
        pos = {node: (i / (len(start_nodes) + 1), 1.0) for i, node in enumerate(start_nodes)}

        levels = defaultdict(list)
        queue = deque((node, 0) for node in start_nodes)
        visited = set()

        while queue:
            node, level = queue.popleft()
            if node not in visited:
                visited.add(node)
                levels[level].append(node)
                queue.extend((successor, level + 1) for successor in structure.successors(node))

        pos |= {
            node: (
                i / (len(nodes) + 1),
                0.0 if node.startswith("O") else 1 - (level / (len(levels) + 1)),
            )
            for level, nodes in levels.items()
            for i, node in enumerate(nodes)
        }

        return pos

    def print_innovative_networks(self, innovative_individuals, save=False):
        num_individuals = len(innovative_individuals)
        rows = num_individuals
        cols = 4  # 1 for NN + 3 for trees

        fig = plt.figure(figsize=(13, rows * 6))
        gs = GridSpec(rows, cols, figure=fig)

        for idx, (individual, fitness_score) in enumerate(innovative_individuals):
            phenotype = Phenotype(individual)
            nn = NNFromGraph(phenotype, inputs=self.inputs, outputs=self.outputs)

            # Neural network plot (col 0)
            pos = self._calculate_node_positions(nn.phenotype.structure)
            node_labels = {node: f"{node}[{nn.phenotype.structure.nodes[node]['threshold']}]"
                           for node in nn.phenotype.structure.nodes}

            ax_nn = fig.add_subplot(gs[idx, 0])
            nx.draw(nn.phenotype.structure, pos, labels=node_labels, with_labels=True, node_size=500,
                    node_color="skyblue", font_size=10, font_weight="bold", arrows=True, ax=ax_nn)
            labels = nx.get_edge_attributes(nn.phenotype.structure, 'weight')
            nx.draw_networkx_edge_labels(nn.phenotype.structure, pos, edge_labels=labels, ax=ax_nn)
            ax_nn.set_title(f'Neural Network\nFitness Score: {
                            "{:.2f}".format(fitness_score)}', fontsize=14)

            # Genome: visualizing 3 trees (cols 1, 2, 3)
            genome = individual.get_trees()  # Assuming the genome is a list of 3 trees
            for tree_idx, tree in enumerate(genome):
                ax_tree = fig.add_subplot(gs[idx, tree_idx + 1])  # tree_idx + 1 to skip the NN column

                node_labels = {}

                # Convert treelib tree to NetworkX graph
                tree_graph = self.treelib_to_nx(tree, node_labels)

                # Layout for tree graph
                tree_pos = nx.bfs_layout(tree_graph, tree.root, align='horizontal')
                max_y = max(y for x, y in tree_pos.values())
                tree_pos = {node: (x, max_y - y) for node, (x, y) in tree_pos.items()}

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
        graph = nx.DiGraph()  # Create a directed graph

        def add_edges(node):
            for child in tree.children(node.identifier):  # Use children() to get the child nodes
                graph.add_edge(node.identifier, child.identifier)  # Add edge from parent to child
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
        for individual in population:
            individual.print()

    def plot_fitness_history(self, history, save=False):
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
            nx.draw_networkx_edge_labels(phenotype.structure, pos, edge_labels=labels)
            if save:
                self.save_file_with_name('phenotype_')
            plt.show()
        except Exception:
            print("Cannot set positions")
            self.print_no_position(phenotype)

    def save_file_with_name(self, name):
        os.makedirs('img', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join('img', f'{name}{timestamp}.png'))

    def print_no_position(self, phenotype):
        nx.draw(phenotype.structure, with_labels=True)
        plt.show()
