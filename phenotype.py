import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from genome import Genome
import copy
import random


class Phenotype:

    def __init__(self, genome) -> None:
        self.structure = nx.DiGraph()
        self.genome = genome
        self.cell_count = 0
        self.internal_register = 0

        # Define initial structure: Input, Initial Cell, Output
        self.structure.add_node("I", attr=genome, type="input")
        self.structure.add_node("O", attr=genome, type="output")
        id = self.add_cell()

        # Connect the initial structure
        self.structure.add_edge("I", id, weight=1)
        self.structure.add_edge(id, "O", weight=1)

    # * Add a new cell to the structure, its id is a progressive number starting from 0
    def add_cell(self):
        genome = self.genome
        self.structure.add_node(str(self.cell_count),
                                attr=genome, type="hidden")
        self.cell_count += 1

        return str(self.cell_count - 1)

    # * Divide cells or do operations
    def develop(self):

        old_structure = copy.deepcopy(self.structure)

        for structural_node in old_structure.nodes:

            if structural_node[0] != "I" and structural_node[0] != "O":

                genome = self.structure.nodes[structural_node]["attr"]
                symbol = genome.get_root_symbol()

                # Terminal symbol, do nothing
                if symbol == "e":
                    pass

                # Wait symbol, do nothing
                if symbol == "w":
                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.get_left_child_genome()
                    )

                # Jump to another level
                elif symbol[0] == "n":
                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.jump_to_other_level(symbol[1])
                    )

                # Parallel division: the two cells have the same edges
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

                # Sequential division: one of the two cells inherits the input, the other the output
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

                # T division: sequential division but the first predecessor is connected with the new cell and the first successor is connected with the parent
                elif symbol == "t":
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

                    self.structure.add_edge(
                        predecessors[0],
                        new_node,
                        weight=1,
                    )
                    self.structure.add_edge(
                        structural_node,
                        successors[0],
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

                # A division: new cell connected to the first two predecessors, the first two successors and in both directions with old cell
                elif symbol == "a":
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

                    # Connect at most the first two predecessors with the new cell
                    possible_links = min(len(predecessors), 2)
                    for predecessor in predecessors[:possible_links]:
                        self.structure.add_edge(
                            predecessor,
                            new_node,
                            weight=1,
                        )

                    # Add an edge from the new cell to the old cell and vice versa
                    self.structure.add_edge(
                        new_node,
                        structural_node,
                        weight=1,
                    )
                    self.structure.add_edge(
                        structural_node, new_node, weight=1)

                    # Split successors into two halves
                    mid = len(successors) // 2
                    second_half = successors[mid:]

                    # Remove edges from the old cell to the second half of the successors and add them to the new cell
                    for successor in second_half:
                        if self.structure.has_edge(structural_node, successor):
                            self.structure.remove_edge(
                                structural_node, successor)
                            self.structure.add_edge(
                                new_node,
                                successor,
                                weight=1,
                            )

                    # Add edges from the new cell to the first two successors
                    possible_links = min(len(successors), 2)
                    for successor in successor[:possible_links]:
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

                # B division: just like A but with one link from new cell to the successors (in the paper it is A division, maybe a typo)
                elif symbol == "b":
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

                    # Connect at most the first two predecessors with the new cell
                    possible_links = min(len(predecessors), 2)
                    for predecessor in predecessors[:possible_links]:
                        self.structure.add_edge(
                            predecessor,
                            new_node,
                            weight=1,
                        )

                    # Add an edge from the new cell to the old cell and vice versa
                    self.structure.add_edge(
                        new_node,
                        structural_node,
                        weight=1,
                    )
                    # self.structure.add_edge(structural_node, new_node, weight=1)

                    # Split successors into two halves
                    mid = len(successors) // 2
                    second_half = successors[mid:]

                    # Remove edges from the old cell to the second half of the successors and add them to the new cell
                    for successor in second_half:
                        if self.structure.has_edge(structural_node, successor):
                            self.structure.remove_edge(
                                structural_node, successor)
                            self.structure.add_edge(
                                new_node,
                                successor,
                                weight=1,
                            )

                    # Add edges from the new cell to the first two successors
                    possible_links = min(len(successors), 1)
                    for successor in successor[:possible_links]:
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

                # H division: link to first successor removed. Link from new cell to first successor
                elif symbol == "h":
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

                    # Connect the new cell to the old one
                    self.structure.add_edge(
                        structural_node, new_node, weight=1)

                    # Remove edge to first successor
                    if self.structure.has_edge(structural_node, successors[0]):
                        self.structure.remove_edge(
                            structural_node, successors[0])

                    # Add edge to first successor
                    self.structure.add_edge(
                        new_node,
                        successors[0],
                        weight=1,
                    )

                    # Update the genome of the two cells
                    self.structure.nodes[new_node]["attr"] = Genome(
                        genome.get_right_child_genome()
                    )
                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.get_left_child_genome()
                    )

                # G division: opposite of H division
                elif symbol == "g":
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

                    # Connect the new cell to the old one
                    self.structure.add_edge(
                        structural_node, new_node, weight=1)

                    # Connect all predecessors but the first to the new cell
                    for predecessor in predecessors[1:]:
                        if self.structure.has_edge(predecessor, structural_node):
                            self.structure.remove_edge(
                                predecessor, structural_node)
                            self.structure.add_edge(
                                predecessor, new_node, weight=1)

                    # Remove edges from old cell to successors and add them to the new cell
                    for successor in successors:
                        if self.structure.has_edge(
                            structural_node, successor
                        ):  # In the case of recurrent links, this link is removed at the previous step
                            self.structure.remove_edge(
                                structural_node, successor)
                        self.structure.add_edge(new_node, successor, weight=1)

                    # Update the genome of the two cells
                    self.structure.nodes[new_node]["attr"] = Genome(
                        genome.get_right_child_genome()
                    )
                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.get_left_child_genome()
                    )

                # R operation: create a new recurrent link
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

                # I operation: increment link_register
                elif symbol == "i":
                    self.internal_register += 1

                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.get_left_child_genome()
                    )

                # D operation: decrement link_register
                elif symbol == "d":
                    self.internal_register -= 1

                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.get_left_child_genome()
                    )

                # + operation: sets the input link with id internal_register to 1
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

                # - operation: sets the input link with id internal_register to -1
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
                    self.structure.remove_edge(predecessor, structural_node)
                    self.structure.add_edge(
                        predecessor,
                        structural_node,
                        weight=-1,
                    )

                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.get_left_child_genome()
                    )

                # C operation: sets the input link with id internal_register to 0
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

                    self.structure.remove_edge(predecessor, structural_node)
                    self.structure.add_edge(
                        predecessor,
                        structural_node,
                        weight=0
                    )

                    self.structure.nodes[structural_node]["attr"] = Genome(
                        genome.get_left_child_genome()
                    )

                # # C operation: cut an input link
                # elif symbol[0] == "c":

                #     self.depth += 1

                #     # Get the link to cut
                #     link_to_cut = int(symbol[1])

                #     predecessors = list(
                #         self.structure.predecessors(structural_node))

                #     # Cut the link if possible
                #     if len(predecessors) > link_to_cut:
                #         self.structure.remove_edge(
                #             predecessors[link_to_cut], structural_node
                #         )

                #     # Otherwise rise an error
                #     else:
                #         raise ValueError("Link to cut not present")

                #     # Update the genome
                #     self.structure.nodes[structural_node]["attr"] = Genome(
                #         genome.get_left_child_genome()
                #     )

                # # D operation: edit an input link to -1
                # elif symbol[0] == "d":

                #     self.depth += 1

                #     # Get the link to edit
                #     link_to_edit = int(symbol[1])

                #     predecessors = list(
                #         self.structure.predecessors(structural_node))

                #     # Edit the link if possible
                #     if len(predecessors) > link_to_edit:

                #         level = self.structure.get_edge_data(
                #             predecessors[link_to_edit], structural_node
                #         )["level"]
                #         depth = self.structure.get_edge_data(
                #             predecessors[link_to_edit], structural_node
                #         )["depth"]

                #         self.structure.remove_edge(
                #             predecessors[link_to_edit], structural_node
                #         )
                #         self.structure.add_edge(
                #             predecessors[link_to_edit],
                #             structural_node,
                #             weight=-1,
                #             level=level,
                #             depth=depth,
                #         )

                #     # Otherwise rise an error
                #     else:
                #         raise ValueError("Link to edit not present")

                #     # Update the genome
                #     self.structure.nodes[structural_node]["attr"] = Genome(
                #         genome.get_left_child_genome()
                #     )

                # # K operation: edit all output links with a specific weight to -1
                # elif symbol[0] == "k":

                #     self.depth += 1

                #     value = int(symbol[1])

                #     successors = list(
                #         self.structure.successors(structural_node))

                #     # Change weight to -1 if the of the link is the one specified in the parameter
                #     for successor in successors:
                #         if (
                #             self.structure.get_edge_data(structural_node, successor)[
                #                 "weight"
                #             ]
                #             == value
                #         ):

                #             level = self.structure.get_edge_data(
                #                 structural_node, successor
                #             )["level"]
                #             depth = self.structure.get_edge_data(
                #                 structural_node, successor
                #             )["depth"]

                #             self.structure.remove_edge(
                #                 structural_node, successor)
                #             self.structure.add_edge(
                #                 successor,
                #                 structural_node,
                #                 weight=-1,
                #                 level=level,
                #                 depth=depth,
                #             )

                #     # Update the genome
                #     self.structure.nodes[structural_node]["attr"] = Genome(
                #         genome.get_left_child_genome()
                #     )

                # # I operation: edit an input link to 1
                # elif symbol[0] == "i":

                #     self.depth += 1

                #     # Get the link to edit
                #     link_to_edit = int(symbol[1])

                #     predecessors = list(
                #         self.structure.predecessors(structural_node))

                #     # Edit the link if possible
                #     if len(predecessors) > link_to_edit:

                #         level = self.structure.get_edge_data(
                #             predecessors[link_to_edit], structural_node
                #         )["level"]
                #         depth = self.structure.get_edge_data(
                #             predecessors[link_to_edit], structural_node
                #         )["depth"]

                #         self.structure.remove_edge(
                #             predecessors[link_to_edit], structural_node
                #         )
                #         self.structure.add_edge(
                #             predecessors[link_to_edit],
                #             structural_node,
                #             weight=1,
                #             level=level,
                #             depth=depth,
                #         )

                    # # Otherwise rise an error
                    # else:
                    #     raise ValueError("Link to edit not present")

                    # # Update the genome
                    # self.structure.nodes[structural_node]["attr"] = Genome(
                    #     genome.get_left_child_genome()
                    # )

                # 3 more symbols but they are picked to have a possible hand-coded solution

    # * Show graphically the structure
    def print(self):
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
                for successor in self.structure.successors(node):
                    queue.append((successor, level + 1))

        # Assign positions to nodes based on levels
        for level, nodes in levels.items():
            for i, node in enumerate(nodes):
                pos[node] = (i / (len(nodes) + 1), 1.0 - (level + 1) * 0.1)

        # Assign fixed position to not visited nodes
        fixed_pos = (0.0, 0.0)
        for node in self.structure.nodes:
            if node not in pos:
                pos[node] = fixed_pos

        # Draw the graph
        nx.draw(
            self.structure,
            pos,
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

    # * Show the graph without caring about the position of the nodes (useful for not connected components)
    def print_no_position(self):
        nx.draw(self.structure, with_labels=True)
        plt.show()

    # Return True if every cell finished developing, otherwise False
    def development_finished(self):

        for node in self.structure.nodes:
            if node[0] != "I" and node[0] != "O":
                genome = self.structure.nodes[node]["attr"]
                symbol = genome.get_root_symbol()

                # If the node that has to be processed is not terminal, then continue
                if symbol not in genome.TERMINAL_SYMBOLS:
                    return False

        return True

    # * Expand the single input and output to match the number of neurons in the first layer
    def expand_inputs_and_outputs(self, inputs, outputs):

        t = 0

        predecessors = list(self.structure.predecessors("O"))
        successors = list(self.structure.successors("I"))

        if len(predecessors) == outputs:
            t += 0.5
        if len(successors) == inputs:
            t += 0.5

        for i in range(inputs):
            node_name = f"I{i}"
            self.structure.add_node(node_name, attr=self.genome, type="input")
            for successor in successors:
                w = self.structure.get_edge_data("I", successor)["weight"]
                self.structure.add_edge(
                    node_name, successor, weight=w
                )

        for i in range(outputs):
            node_name = f"O{i}"
            self.structure.add_node(node_name, attr=self.genome, type="output")
            for predecessor in predecessors:
                w = self.structure.get_edge_data(predecessor, "O")["weight"]
                self.structure.add_edge(
                    predecessor, node_name, weight=w
                )

        self.structure.remove_node("O")
        self.structure.remove_node("I")

        return t

    def expand_some_inputs_and_outputs(self, inputs, outputs):

        t = 0

        predecessors = list(self.structure.predecessors("O"))
        successors = list(self.structure.successors("I"))

        input_neurons = len(successors)
        output_neurons = len(predecessors)

        if input_neurons < inputs:
            remaining_inputs = inputs - input_neurons
            for i in range(input_neurons):
                node_name = f"I{i}"
                self.structure.add_node(
                    node_name, attr=self.genome, type="input")
                self.structure.add_edge(
                    node_name, successors[i], weight=1
                )
            for i in range(remaining_inputs):
                node_name = f"I{input_neurons+i}"
                self.structure.add_node(
                    node_name, attr=self.genome, type="input")

                random_successor = random.choice(successors)
                self.structure.add_edge(
                    node_name, random_successor, weight=1
                )
                index = successors.index(random_successor)
        else:
            t += 0.5
            for i in range(inputs):
                node_name = f"I{i}"
                self.structure.add_node(
                    node_name, attr=self.genome, type="input")
                self.structure.add_edge(
                    node_name, successors[i], weight=1
                )

        if output_neurons < outputs:
            remaining_outputs = outputs - output_neurons
            for i in range(output_neurons):
                node_name = f"O{i}"
                self.structure.add_node(
                    node_name, attr=self.genome, type="output")
                self.structure.add_edge(
                    predecessors[i], node_name, weight=1)
            for i in range(remaining_outputs):
                node_name = f"O{output_neurons+i}"
                self.structure.add_node(
                    node_name, attr=self.genome, type="output")
                random_predecessor = random.choice(predecessors)
                self.structure.add_edge(
                    random_predecessor, node_name, weight=1
                )
                index = predecessors.index(random_predecessor)
        else:
            t += 0.5
            for i in range(outputs):
                node_name = f"O{i}"
                self.structure.add_node(
                    node_name, attr=self.genome, type="output")
                self.structure.add_edge(
                    predecessors[i], node_name, weight=1
                )

        for predecessor in predecessors:
            self.structure.remove_edge(predecessor, "O")
        for successor in successors:
            self.structure.remove_edge("I", successor)

        self.structure.remove_node("I")
        self.structure.remove_node("O")

        return t

# TODO Edit the genome to create the tuned edge
# ? Maybe should be in another class, keep in mind for the future
# def back_code(self, parameter, first_node, second_node):

#     if self.structure.has_edge(first_node, second_node):
#         level = self.structure.get_edge_data(first_node, second_node)["level"]
#         depth = self.structure.get_edge_data(first_node, second_node)["depth"]

#         self.genome.back_code(symbol, parameter, level, depth)
#     else:
#         raise ValueError("Link to edit not present")
