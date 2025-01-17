import networkx as nx
from core.genome import Genome
import copy


class Phenotype:
    """
    Represents the phenotype of a genome in a directed graph structure.

    This class manages the creation and manipulation of nodes representing cells, providing methods for 
    developing the phenotype based on its genome, adding cells, and modifying connections.

    :ivar structure: Directed graph representing the phenotype's structure.
    :ivar genome: The genome associated with this phenotype.
    :ivar cell_count: Count of cells in the phenotype.
    :ivar internal_register: Register used for internal operations within the phenotype.
    """

    def __init__(self, genome: Genome) -> None:
        """
        Initializes a Phenotype instance with a given genome.

        Sets up the initial structure of the phenotype, including input and output nodes, 
        and establishes connections between them.

        :param genome: The genome associated with this phenotype.
        """
        self.structure = nx.DiGraph()
        self.genome = genome
        self.cell_count = 0
        self.internal_register = 0
        self.level_limit = 20

        # Define initial structure: Input, Initial Cell, Output
        self.structure.add_node("I", attr=genome, type="input", threshold=0)
        self.structure.add_node("O", attr=genome, type="output", threshold=0)
        identificator = self.add_cell()

        # Connect the initial structure
        self.structure.add_edge("I", identificator, weight=1)
        # ! Set to -1 to go back to usual setting
        self.structure.add_edge(identificator, "O", weight=1)

    # * Add a new cell to the structure
    def add_cell(self):
        """
        Adds a new cell to the phenotype's structure.

        Creates a new hidden cell node and increments the cell count.

        :return: Identifier of the newly added cell.
        :rtype: str
        """
        genome = self.genome
        self.structure.add_node(str(self.cell_count),
                                attr=genome, type="hidden", threshold=0)
        self.cell_count += 1

        return str(self.cell_count - 1)

    # * Divide cells or do operations
    def develop(self):
        """
        Develops the phenotype's structure based on its genome.

        Modifies the phenotype's structure by applying operations based on the associated genome, 
        focusing on hidden nodes to evolve the structure.
        """

        old_structure = copy.deepcopy(self.structure)

        for structural_node in old_structure.nodes:
            hidden_units = sum(
                self.structure.nodes[node]["type"] == "hidden" for node in self.structure.nodes)
            inputs = sum(
                self.structure.nodes[node]["type"] == "input" for node in self.structure.nodes)
            outputs = sum(
                self.structure.nodes[node]["type"] == "output" for node in self.structure.nodes)

            if hidden_units / (inputs + outputs) > 4:
                r = 0
                t = 0
                break
            if structural_node[0] in ["I", "O"]:
                continue
            genome = self.structure.nodes[structural_node]["attr"]
            symbol = genome.get_root_symbol()

            new_node = self.perform_operation(structural_node, symbol, genome)
            self.read_genome(structural_node, new_node, symbol, genome)

    # * Perform the proper operation according to the symbol
    def perform_operation(self, structural_node, symbol, genome):
        """
        Performs an operation on a structural node based on the given symbol.

        Interprets the symbol to determine the appropriate operation for the specified node, 
        modifying the node's properties or structure as needed.

        :param structural_node: Node on which the operation is performed.
        :param symbol: Character representing the operation.
        :param genome: Genome associated with the structural node.
        :return: Newly created node if applicable; otherwise, None.
        """
        new_node = None

        if symbol == "t":
            self.edit_threshold(structural_node, delta=1)
        if symbol == "u":
            self.edit_threshold(structural_node, delta=-1)
        elif symbol == "w":
            pass
        elif symbol == "n":
            self.jump(structural_node, genome)
        elif symbol == "p":
            new_node = self.split_parallel(structural_node)
        elif symbol == "s":
            new_node = self.split_sequential(structural_node)
        elif symbol == "r":
            self.add_recurrent_edge(structural_node)
        elif symbol == "i":
            self.edit_register(+1)
        elif symbol == "d":
            self.edit_register(-1)
        elif symbol == "+":
            self.change_weight(structural_node, 1)
        elif symbol == "-":
            self.change_weight(structural_node, -1)
        elif symbol == "c":
            self.change_weight(structural_node, 0)

        return new_node

    # * Edit threshold of a cell
    def edit_threshold(self, structural_node, delta):
        """
        Sets the threshold of a specified node to 1.

        Updates the threshold attribute of the node to modify its behavior.

        :param structural_node: Node whose threshold is edited.
        """
        self.structure.nodes[structural_node]["threshold"] += delta

    # * Jump to the next level
    def jump(self, structural_node, genome):
        """
        Performs a jump operation on the specified node.

        :param structural_node: Node on which the jump operation is performed.
        :param genome: Genome associated with the structural node.
        """
        pass

    def split_parallel(self, structural_node):
        """
        Creates a new node that splits connections in parallel.

        Adds a new cell, connecting it to the predecessors and successors of the given node.

        :param structural_node: Node to be split in parallel.
        :return: Identifier of the newly created node.
        :rtype: str
        """
        new_node = self.add_cell()

        predecessors = list(self.structure.predecessors(structural_node))
        successors = list(self.structure.successors(structural_node))

        # Change pre-existing recurrent links to old->new link
        if self.structure.has_edge(structural_node, structural_node):
            self.structure.remove_edge(structural_node, structural_node)
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

        return new_node

    # * Perform a sequential split
    def split_sequential(self, structural_node):
        """
        Creates a new node that splits connections sequentially.

        Adds a new cell and modifies the connections of the original node, transferring its 
        successors to the new node.

        :param structural_node: Node to be split sequentially.
        :return: Identifier of the newly created node.
        :rtype: str
        """
        new_node = self.add_cell()

        successors = list(self.structure.successors(structural_node))

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
        self.structure.add_edge(structural_node, new_node, weight=1)

        return new_node

    # * Add a recurrent edge
    def add_recurrent_edge(self, structural_node):
        """
        Adds a recurrent edge to the specified node by creating a self-loop.

        :param structural_node: Node to which the recurrent edge is added.
        """
        self.structure.add_edge(
            structural_node,
            structural_node,
            weight=1,
        )

    # * Set self.internal_register to value
    def edit_register(self, value):
        """
        Modifies the internal register by adding a specified value.

        :param value: Amount to add to the internal register.
        """
        self.internal_register += value

    def read_genome(self, structural_node, new_node, symbol, genome):
        """
        Processes the genome based on the specified symbol and updates the node accordingly.

        :param structural_node: Node to be processed.
        :param new_node: New node that may be created or modified.
        :param symbol: Character representing the action.
        :param genome: Genome associated with the structural node.
        """
        if symbol in ["p", "s"]:
            self.split(structural_node, new_node, genome)
        elif symbol in ["t", "u", "w", "r", "i", "d", "+", "-", "c"]:
            self.continue_reading(structural_node, genome)
        elif symbol == "n":
            self.structure.nodes[structural_node]["attr"] = Genome(
                genome.jump()
            )

    # * Choose next symbols for the new cells
    def split(self, structural_node, new_node, genome):
        """
        Splits genome attributes between specified nodes.

        Assigns the right and left child genomes to the new and original nodes.

        :param structural_node: Original node whose genome will be updated.
        :param new_node: New node receiving the right child genome.
        :param genome: Genome from which child genomes are derived.
        """
        self.structure.nodes[new_node]["attr"] = Genome(
            genome.get_right_child_genome()
        )
        self.structure.nodes[structural_node]["attr"] = Genome(
            genome.get_left_child_genome()
        )

    # * Move to the next symbol
    def continue_reading(self, structural_node, genome):
        """
        Updates the genome attribute of the specified node, continuing the genome reading process.

        :param structural_node: Node whose genome attribute is updated.
        :param genome: Genome from which the left child genome is derived.
        """
        self.structure.nodes[structural_node]["attr"] = Genome(
            genome.get_left_child_genome()
        )

    # * Set weight as the weight of the input edge pointed by the internal register
    def change_weight(self, structural_node, weight):
        """
        Updates the weight of the edge connecting a predecessor to the specified node.

        Modifies the edge weight from a selected predecessor to the node based on the internal register.

        :param structural_node: Node whose incoming edge weight is changed.
        :param weight: New weight to assign to the edge.
        """
        link_to_edit = self.internal_register

        predecessors = list(self.structure.predecessors(structural_node))

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
            weight=weight,
        )

    # * Return True if every cell finished developing, otherwise False
    def development_finished(self):
        """
        Checks if development of all cells is complete.

        :return: True if all cells have finished developing; otherwise, False.
        :rtype: bool
        """

        hidden_units = sum(
            self.structure.nodes[node]["type"] == "hidden" for node in self.structure.nodes)
        inputs = sum(
            self.structure.nodes[node]["type"] == "input" for node in self.structure.nodes)
        outputs = sum(
            self.structure.nodes[node]["type"] == "output" for node in self.structure.nodes)

        # print(f"Hidden units: {hidden_units} vs. Inputs: {
        #       inputs} vs. Outputs: {outputs}")

        if hidden_units / (inputs + outputs) > 4:
            r = 0
            t = 0
            return True

        for node in self.structure.nodes:
            if node[0] not in ["I", "O"]:
                genome = self.structure.nodes[node]["attr"]
                symbol = genome.get_root_symbol()

                # If the node that has to be processed is not terminal, then continue
                # if symbol not in genome.TERMINAL_SYMBOLS:
                if symbol != "e":
                    return False

        return True

    # * Expand the single input and output to match the number of neurons in the first layer
    def expand_inputs_and_outputs(self, inputs, outputs):
        """
        Expands the input and output nodes of the structure, ensuring validity of connections.

        :param inputs: Number of input nodes to add.
        :param outputs: Number of output nodes to add.
        :raises ValueError: If the structure has already been expanded.
        :return: Tuple containing two metrics related to the structure configuration.
        :rtype: tuple
        """
        if "O" not in self.structure.nodes:
            print("Structure already expanded")
            raise ValueError("Structure already expanded")

        else:
            predecessors = list(self.structure.predecessors("O"))
            successors = list(self.structure.successors("I"))

            for i in range(inputs):
                node_name = f"I{i}"
                self.structure.add_node(
                    node_name, attr=self.genome, type="input", threshold=0)
                for successor in successors:
                    w = self.structure.get_edge_data("I", successor)["weight"]
                    self.structure.add_edge(node_name, successor, weight=w)

            for i in range(outputs):
                node_name = f"O{i}"
                self.structure.add_node(
                    node_name, attr=self.genome, type="output", threshold=0)
                for predecessor in predecessors:
                    w = self.structure.get_edge_data(
                        predecessor, "O")["weight"]
                    self.structure.add_edge(predecessor, node_name, weight=w)

            self.structure.remove_node("O")
            self.structure.remove_node("I")

            i = 0
            while self.development_finished() == False and i < self.level_limit:
                i += 1
                self.develop()

            predecessors = list(self.structure.predecessors("O0"))
            successors = list(self.structure.successors("I0"))

            t = 0
            r = 1

            if len(self.structure.nodes) != inputs + outputs + 1:
                if len(predecessors) == outputs:
                    t += 0.5
                if len(successors) == inputs:
                    t += 0.5

            hidden_units = sum(
                self.structure.nodes[node]["type"] == "hidden" for node in self.structure.nodes)

            print(f'Levels: {i}')

            if hidden_units / (inputs + outputs) > 4 or i>= self.level_limit:
                r = 0
                t = 0

            return t, r
