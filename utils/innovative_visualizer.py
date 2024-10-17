from core.phenotype import Phenotype
from core.neural_network_from_graph import NNFromGraph
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime


class Visualizer:

    @staticmethod
    def _calculate_node_positions(structure):
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

    @staticmethod
    def print_innovative_networks(innovative_individuals, save=False):
        num_individuals = len(innovative_individuals)
        rows = num_individuals
        cols = 2

        fig = plt.figure(figsize=(12, rows * 5))
        gs = GridSpec(rows, cols, figure=fig)

        for idx, (individual, fitness_score) in enumerate(innovative_individuals):
            phenotype = Phenotype(individual)
            nn = NNFromGraph(phenotype)

            pos = Visualizer._calculate_node_positions(nn.phenotype.structure)

            node_labels = {node: f"{node}[{nn.phenotype.structure.nodes[node]['threshold']}]"
                           for node in nn.phenotype.structure.nodes}

            # Plot the neural network structure (phenotype)
            ax_nn = fig.add_subplot(gs[idx, 0])
            nx.draw(nn.phenotype.structure, pos, labels=node_labels, with_labels=True, node_size=500,
                    node_color="skyblue", font_size=10, font_weight="bold", arrows=True, ax=ax_nn)
            labels = nx.get_edge_attributes(nn.phenotype.structure, 'weight')
            nx.draw_networkx_edge_labels(nn.phenotype.structure, pos, edge_labels=labels, ax=ax_nn)
            ax_nn.set_title(f'Neural Network Structure\nFitness Score: {fitness_score}', fontsize=14, fontweight='bold')

            # Print the genotype
            ax_genotype = fig.add_subplot(gs[idx, 1])
            ax_genotype.axis('off')
            ax_genotype.text(0.5, 1, str(individual), horizontalalignment='left',
                             verticalalignment='top', wrap=True, fontsize=12)
            ax_genotype.set_title('Genotype', fontsize=14, fontweight='bold')

        plt.tight_layout()
        if save:
            os.makedirs('img', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join('img', f'innovative_networks_{timestamp}.png'))
        plt.show()

    @staticmethod
    def print_population(population):
        for individual in population:
            individual.print()

    @staticmethod
    def plot_fitness_history(history, save=False):
        generations = list(range(1, len(history) + 1))
        plt.plot(generations, history, marker='o')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness History Over Generations')
        plt.grid(True)
        if save:
            os.makedirs('img', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join('img', f'fitness_history_{timestamp}.png'))
        plt.show()

    @staticmethod
    def print_phenotype(phenotype, save=False):
        try:
            pos = Visualizer._calculate_node_positions(phenotype.structure)

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
                os.makedirs('img', exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plt.savefig(os.path.join('img', f'phenotype_{timestamp}.png'))
            plt.show()
        except Exception:
            print("Cannot set positions")
            Visualizer.print_no_position(phenotype)

    @staticmethod
    def print_no_position(phenotype):
        nx.draw(phenotype.structure, with_labels=True)
        plt.show()
