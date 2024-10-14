import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from core.genome import Genome
import copy


class Phenotype:

    def __init__(self, genome) -> None:
        self.structure = nx.DiGraph()
        self.genome = genome
        self.cell_count = 0
        self.internal_register = 0

        # Define initial structure: Input, Initial Cell, Output
        self.structure.add_node("I", attr=genome, type="input", threshold=0)
        self.structure.add_node("O", attr=genome, type="output", threshold=0)
        identificator = self.add_cell()

        # Connect the initial structure
        self.structure.add_edge("I", identificator, weight=1)
        self.structure.add_edge(identificator, "O", weight=-1)

    # * Add a new cell to the structure
    def add_cell(self):
        genome = self.genome
        self.structure.add_node(str(self.cell_count), attr=genome, type="hidden", threshold=0)
        self.cell_count += 1

        return str(self.cell_count - 1)

    # * Divide cells or do operations
    def develop(self):

        old_structure = copy.deepcopy(self.structure)

        for structural_node in old_structure.nodes:

            if structural_node[0] not in ["I", "O"]:

                genome = self.structure.nodes[structural_node]["attr"]
                symbol = genome.get_root_symbol()

                # Threshold symbol, set the threshold to 0.5
                if symbol == "t":
                    self.structure.nodes[structural_node]["threshold"] = 1
                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.get_left_child_genome()
                    )

                # Wait symbol, do nothing
                if symbol == "w":
                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.get_left_child_genome()
                    )

                elif symbol[0] == "n":
                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.jump_to_other_level(symbol[1])
                    )

                elif symbol == "p":
                    new_node = self.add_cell()

                    predecessors = list(
                        self.structure.predecessors(structural_node))
                    successors = list(
                        self.structure.successors(structural_node))

                    # Change pre-existing recurrent links to old->new link
                    if self.structure.has_edge(structural_node, structural_node):
                        self.structure.remove_edge(
                            structural_node, structural_node)
                        if not self.structure.has_edge(structural_node, new_node):
                            self.structure.add_edge(
                                structural_node,
                                new_node,
                                weight=1,
                            )

                    # Copy predecessor links
                    for predecessor in predecessors:
                        self.structure.add_edge(
                            predecessor,
                            new_node,
                            weight=1,
                        )

                    # Copy successor links
                    for successor in successors:
                        self.structure.add_edge(
                            new_node,
                            successor,
                            weight=1,
                        )

                    # Update the genome of the two cells
                    self.structure.nodes[new_node]["attr"] = Genome(
                        genome.get_right_child_genome()
                    )
                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.get_left_child_genome()
                    )

                elif symbol == "s":
                    new_node = self.add_cell()

                    successors = list(
                        self.structure.successors(structural_node))

                    # Change pre-existing recurrent links to old->new link
                    if self.structure.has_edge(structural_node, structural_node):
                        self.structure.remove_edge(
                            structural_node, structural_node)
                        if not self.structure.has_edge(structural_node, new_node):
                            self.structure.add_edge(
                                structural_node,
                                new_node,
                                weight=1,
                            )

                    # Successor links have to be removed from the parent and added to the new cell
                    for successor in successors:
                        if self.structure.has_edge(structural_node, successor):
                            self.structure.remove_edge(
                                structural_node, successor)
                            self.structure.add_edge(
                                new_node,
                                successor,
                                weight=1,
                            )

                    # Connect the two cells
                    self.structure.add_edge(
                        structural_node, new_node, weight=1)

                    # Update the genome of the two cells
                    self.structure.nodes[new_node]["attr"] = Genome(
                        genome.get_right_child_genome()
                    )
                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.get_left_child_genome()
                    )

                elif symbol == "r":

                    # Add recurrent link
                    self.structure.add_edge(
                        structural_node,
                        structural_node,
                        weight=1,
                    )

                    # Update the genome
                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.get_left_child_genome()
                    )

                elif symbol == "i":
                    self.internal_register += 1

                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.get_left_child_genome()
                    )

                elif symbol == "d":
                    self.internal_register -= 1

                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.get_left_child_genome()
                    )

                elif symbol == "+":

                    link_to_edit = self.internal_register

                    predecessors = list(
                        self.structure.predecessors(structural_node))

                    if 0 <= link_to_edit < len(predecessors):
                        predecessor = predecessors[link_to_edit]
                    elif link_to_edit >= len(predecessors):
                        predecessor = predecessors[-1]
                    else:
                        predecessor = predecessors[0]

                    w = self.structure.get_edge_data(
                        predecessor, structural_node)["weight"]
                    self.structure.remove_edge(predecessor, structural_node)
                    self.structure.add_edge(
                        predecessor,
                        structural_node,
                        weight=+1,
                    )

                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.get_left_child_genome()
                    )

                elif symbol == "-":
                    link_to_edit = self.internal_register

                    predecessors = list(
                        self.structure.predecessors(structural_node))

                    if 0 <= link_to_edit < len(predecessors):
                        predecessor = predecessors[link_to_edit]
                    elif link_to_edit >= len(predecessors):
                        predecessor = predecessors[-1]
                    else:
                        predecessor = predecessors[0]

                    w = self.structure.get_edge_data(
                        predecessor, structural_node)["weight"]
                    self.change_weight(predecessor, structural_node, -1, genome)
                elif symbol == "c":
                    link_to_edit = self.internal_register

                    predecessors = list(
                        self.structure.predecessors(structural_node))

                    if 0 <= link_to_edit < len(predecessors):
                        predecessor = predecessors[link_to_edit]
                    elif link_to_edit >= len(predecessors):
                        predecessor = predecessors[-1]
                    else:
                        predecessor = predecessors[0]

                    self.change_weight(predecessor, structural_node, 0, genome)

    # * Change weight of desired edge
    def change_weight(self, predecessor, structural_node, weight, genome):
        self.structure.remove_edge(predecessor, structural_node)
        self.structure.add_edge(predecessor, structural_node, weight=weight)

        self.structure.nodes[structural_node]["attr"] = Genome(
            genome.get_left_child_genome()
        )

    # * Show graphically the structure
    def print(self):
        try:
            pos = {}

            # Check if 'I' is present
            if "I" in self.structure.nodes:
                pos["I"] = (0.5, 1.0)  # Top center
                start_nodes = ["I"]
            else:
                start_nodes = [
                    node for node in self.structure.nodes if node.startswith("I")
                ]
                for i, node in enumerate(start_nodes):
                    pos[node] = (i / (len(start_nodes) + 1), 1.0)

            # Perform BFS to determine levels of each node
            levels = defaultdict(list)
            queue = [(node, 0) for node in start_nodes]
            visited = set()

            while queue:
                node, level = queue.pop(0)
                if node not in visited:
                    visited.add(node)
                    levels[level].append(node)
                    queue.extend(
                        (successor, level + 1)
                        for successor in self.structure.successors(node)
                    )
            # Assign positions to nodes based on levels
            for level, nodes in levels.items():
                for i, node in enumerate(nodes):
                    pos[node] = (i / (len(nodes) + 1), 1.0 - (level + 1) * 0.1)

            # Assign fixed position to not visited nodes
            fixed_pos = (0.0, 0.0)
            for node in self.structure.nodes:
                if node not in pos:
                    pos[node] = fixed_pos

            node_labels = {node: f"{node}[{self.structure.nodes[node]['threshold']}]" for node in self.structure.nodes}
            # Draw the graph
            nx.draw(
                self.structure,
                pos,
                labels=node_labels,
                with_labels=True,
                node_size=500,
                node_color="skyblue",
                font_size=10,
                font_weight="bold",
                arrows=True,
            )
            labels = nx.get_edge_attributes(self.structure, 'weight')
            nx.draw_networkx_edge_labels(self.structure, pos, edge_labels=labels)
            plt.show()
        except Exception:
            print("Cannot set positions")
            self.print_no_position()

    # * Show the graph without caring about the position of the nodes (useful for not connected components)
    def print_no_position(self):
        nx.draw(self.structure, with_labels=True)
        plt.show()

    # * Return True if every cell finished developing, otherwise False
    def development_finished(self):

        for node in self.structure.nodes:
            if node[0] not in ["I", "O"]:
                genome = self.structure.nodes[node]["attr"]
                symbol = genome.get_root_symbol()

                # If the node that has to be processed is not terminal, then continue
                if symbol not in genome.TERMINAL_SYMBOLS:
                    return False

        return True

    # * Expand the single input and output to match the number of neurons in the first layer
    def expand_inputs_and_outputs(self, inputs, outputs):

        if "O" not in self.structure.nodes:
            print("Structure already expanded")
            raise ValueError("Structure already expanded")

        else:
            predecessors = list(self.structure.predecessors("O"))
            successors = list(self.structure.successors("I"))

            for i in range(inputs):
                node_name = f"I{i}"
                self.structure.add_node(node_name, attr=self.genome, type="input", threshold=0)
                for successor in successors:
                    w = self.structure.get_edge_data("I", successor)["weight"]
                    self.structure.add_edge(node_name, successor, weight=w)

            for i in range(outputs):
                node_name = f"O{i}"
                self.structure.add_node(node_name, attr=self.genome, type="output", threshold=0)
                for predecessor in predecessors:
                    w = self.structure.get_edge_data(predecessor, "O")["weight"]
                    self.structure.add_edge(predecessor, node_name, weight=w)

            self.structure.remove_node("O")
            self.structure.remove_node("I")

            while self.development_finished() == False:
                self.develop()

            predecessors = list(self.structure.predecessors("O0"))
            successors = list(self.structure.successors("I0"))

            t = 0

            if len(self.structure.nodes) != inputs + outputs + 1:
                if len(predecessors) == outputs:
                    t += 0.5
                if len(successors) == inputs:
                    t += 0.5

            return t
