import networkx as nx
from core.genome import Genome
import copy


class Phenotype:
    """
    Represent the phenotype of a genome in a directed graph structure.

    This class manages the structure and behavior of a phenotype, including the creation and manipulation of nodes 
    that represent cells. It provides methods for developing the phenotype based on its genome, adding cells, 
    and modifying connections between nodes.

    Attributes:
        structure (nx.DiGraph): The directed graph representing the phenotype's structure.
        genome: The genome associated with this phenotype.
        cell_count (int): The count of cells in the phenotype.
        internal_register (int): A register used for internal operations within the phenotype.
    """

    def __init__(self, genome: Genome) -> None:
        """
        Initialize a Phenotype instance with a given genome.

        This constructor sets up the initial structure of the phenotype, including input and output nodes, 
        and establishes connections between them. It also initializes various attributes related to the phenotype's structure.

        Args:
            genome: The genome associated with this phenotype.

        Returns:
            None
        """
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
        """
        Add a new cell to the phenotype's structure.

        This method creates a new hidden cell node in the phenotype's structure and increments the cell count. 
        It returns the identifier of the newly added cell.

        Args:
            self: The instance of the class.

        Returns:
            str: The identifier of the newly added cell.
        """
        genome = self.genome
        self.structure.add_node(str(self.cell_count),
                                attr=genome, type="hidden", threshold=0)
        self.cell_count += 1

        return str(self.cell_count - 1)

    # * Divide cells or do operations
    def develop(self):
        """
        Develop the phenotype's structure based on its genome.

        This method modifies the phenotype's structure by iterating through its nodes and applying operations
        based on the associated genome. It skips the input and output nodes, focusing on the hidden nodes to
        evolve the structure further.

        Args:
            self: The instance of the class.

        Returns:
            None
        """

        old_structure = copy.deepcopy(self.structure)

        for structural_node in old_structure.nodes:
            if structural_node[0] in ["I", "O"]:
                continue
            genome = self.structure.nodes[structural_node]["attr"]
            symbol = genome.get_root_symbol()

            new_node = self.perform_operation(structural_node, symbol, genome)
            self.read_genome(structural_node, new_node, symbol, genome)

    # * Perform the proper operation according to the symbol
    def perform_operation(self, structural_node, symbol, genome):
        """
        Perform an operation on a structural node based on the given symbol.

        This method interprets the provided symbol to determine the appropriate operation to execute on the 
        specified structural node. It may modify the node's properties or structure and returns a new node if applicable.

        Args:
            structural_node: The node on which the operation is to be performed.
            symbol: A character representing the operation to be executed.
            genome: The genome associated with the structural node.

        Returns:
            The newly created node if applicable; otherwise, None.
        """
        new_node = None

        if symbol == "t":
            self.edit_threshold(structural_node)
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
    def edit_threshold(self, structural_node):
        """
        Set the threshold of a specified structural node to 1.

        This method updates the threshold attribute of the given structural node in the phenotype's structure.
        It is typically used to modify the behavior of the node during the development process.

        Args:
            structural_node: The node whose threshold is to be edited.

        Returns:
            None
        """
        self.structure.nodes[structural_node]["threshold"] = 1

    # * Jump to the next level
    def jump(self, structural_node, genome):
        """
        Perform a jump operation on the specified structural node.

        This method is intended to modify the behavior or state of the given structural node based on the associated genome. 
        The specific implementation details of the jump operation are not defined in this method.

        Args:
            structural_node: The node on which the jump operation is to be performed.
            genome: The genome associated with the structural node.

        Returns:
            None
        """
        pass

    def split_parallel(self, structural_node):
        """
        Create a new node that splits the connections of the specified structural node in parallel.

        This method adds a new cell and establishes connections from the new node to the predecessors and successors 
        of the given structural node. It also handles any recurrent links associated with the original node.

        Args:
            structural_node: The node to be split in parallel.

        Returns:
            The newly created node that represents the split.
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
        Create a new node that splits the connections of the specified structural node sequentially.

        This method adds a new cell and modifies the connections of the original structural node by transferring 
        its successors to the new node. It also handles any recurrent links associated with the original node.

        Args:
            structural_node: The node to be split sequentially.

        Returns:
            The newly created node that represents the split.
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
        Add a recurrent edge to the specified structural node.

        This method creates a self-loop on the given structural node by adding an edge that connects the node to itself.
        This is typically used to represent recurrent connections in the phenotype's structure.

        Args:
            structural_node: The node to which the recurrent edge will be added.

        Returns:
            None
        """
        self.structure.add_edge(
            structural_node,
            structural_node,
            weight=1,
        )

    # * Set self.internal_register to value
    def edit_register(self, value):
        """
        Modify the internal register by adding a specified value.

        This method updates the internal register by incrementing it with the provided value. 
        It is typically used to adjust the state of the register during the phenotype's operations.

        Args:
            value: The amount to add to the internal register.

        Returns:
            None
        """
        self.internal_register += value

    def read_genome(self, structural_node, new_node, symbol, genome):
        """
        Process the genome based on the specified symbol and update the structural node accordingly.

        This method interprets the provided symbol to determine the appropriate action to take on the structural node, 
        which may involve splitting the node, continuing the reading process, or updating the node's attributes. 
        It facilitates the interaction between the genome and the phenotype's structure.

        Args:
            structural_node: The node to be processed based on the genome.
            new_node: The new node that may be created or modified during the process.
            symbol: A character representing the action to be taken.
            genome: The genome associated with the structural node.

        Returns:
            None
        """
        if symbol in ["p", "s"]:
            self.split(structural_node, new_node, genome)
        elif symbol in ["t", "w", "r", "i", "d", "+", "-", "c"]:
            self.continue_reading(structural_node, genome)
        elif symbol == "n":
            self.structure.nodes[structural_node]["attr"] = Genome(
                genome.jump()
            )

    # * Choose next symbols for the new cells
    def split(self, structural_node, new_node, genome):
        """
        Split the genome attributes between the specified structural nodes.

        This method assigns the right and left child genomes to the new and original structural nodes, respectively. 
        It facilitates the division of genetic information during the development of the phenotype.

        Args:
            structural_node: The original node whose genome will be updated.
            new_node: The new node that will receive the right child genome.
            genome: The genome from which the child genomes are derived.

        Returns:
            None
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
        Update the genome attribute of the specified structural node.

        This method assigns the left child genome to the given structural node, effectively continuing the reading 
        process of the genome. It is used to progress through the genetic information associated with the phenotype.

        Args:
            structural_node: The node whose genome attribute will be updated.
            genome: The genome from which the left child genome is derived.

        Returns:
            None
        """
        self.structure.nodes[structural_node]["attr"] = Genome(
            genome.get_left_child_genome()
        )

    # * Set weight as the weight of the input edge pointed by the internal register
    def change_weight(self, structural_node, weight):
        """
        Update the weight of the edge connecting a predecessor to the specified structural node.

        This method modifies the weight of the edge from a selected predecessor node to the given structural node 
        based on the current internal register. It ensures that the connection reflects the desired weight for the 
        phenotype's structure.

        Args:
            structural_node: The node whose incoming edge weight will be changed.
            weight: The new weight to assign to the edge.

        Returns:
            None
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
        Check if the development of all cells in the phenotype is complete.

        This method iterates through the nodes in the phenotype's structure to determine if all non-terminal nodes 
        have reached a terminal state. It returns True if all cells have finished developing, and False otherwise.

        Args:
            self: The instance of the class.

        Returns:
            bool: True if all cells have finished developing, otherwise False.
        """

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
        Expand the input and output nodes of the phenotype's structure.

        This method adds the specified number of input and output nodes to the phenotype's structure, 
        connecting them to the existing nodes while ensuring the structure remains valid. It also 
        removes the original input and output nodes and checks the development status of the phenotype.

        Args:
            inputs: The number of input nodes to add.
            outputs: The number of output nodes to add.

        Raises:
            ValueError: If the structure has already been expanded.

        Returns:
            tuple: A tuple containing two values, t and r, which represent specific metrics 
            related to the structure's configuration.
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

            while self.development_finished() == False:
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

            if hidden_units / inputs + outputs > 4:
                r = 0
                t = 0

            return t, r
