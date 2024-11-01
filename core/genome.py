from treelib import Tree
from utils.counter import GlobalCounter
import base64
import pickle
"""
genome.py

This module provides functionalities for genome manipulation.
"""


class Genome:
    """
    Represents a genome consisting of multiple trees, each containing genes and their relationships.

    The Genome class manages a collection of trees, allowing for operations such as symbol changes, subtree retrieval, and printing of tree structures. It provides methods to manipulate and access the genetic information represented in the trees.
    """

    LEVELS = 3  # How many trees to consider
    STARTING_SYMBOL = "e"  # Each newborn gene will start with the end symbol
    TERMINAL_SYMBOLS = ["e", "n"]
    DIVISION_SYMBOLS = ["p", "s"]
    OPERATIONAL_SYMBOLS = ["w", "i", "d", "+", "-", "c", "r", "t"]

    SYMBOLS = TERMINAL_SYMBOLS + DIVISION_SYMBOLS + OPERATIONAL_SYMBOLS

    # * Create the list of trees if none is provided, otherwise use the provided trees
    def __init__(self, trees: list = None, parents=None) -> None:
        """
        Initializes a genome with a specified number of trees or uses provided trees.

        This constructor sets up the genome's tree structure. If no trees are provided, it creates a default set of trees, each initialized with a single gene node.

        Args:
            trees (list, optional): A list of trees to initialize the genome with. Defaults to None.
            parents (optional): A parameter to specify parent relationships. Defaults to None.
        """
        if trees is None:
            self._trees = []
            for level in range(self.LEVELS):  # Create the trees with a single gene
                self._trees.append(Tree())

                self._trees[level].create_node(
                    tag=self.STARTING_SYMBOL,
                    identifier=GlobalCounter.next(),
                    parent=None,
                )

        else:
            self.LEVELS = len(trees)
            self._trees = trees

        self.parents = parents

    # * Edit the symbol of a gene and create the proper amount of children
    def change_symbol(self, level: int, node_id: str, symbol: str):
        """
        Changes the symbol of a specified gene and creates child nodes as necessary.

        This method updates the symbol of a gene at a given level and node ID. If the new symbol is not terminal, it creates additional child nodes based on the symbol type, ensuring the tree structure remains valid.

        Args:
            level (int): The level of the tree where the gene is located.
            node_id (str): The identifier of the node whose symbol is to be changed.
            symbol (str): The new symbol to assign to the gene.

        Raises:
            ValueError: If the provided symbol is not valid.
        """
        if symbol not in self.SYMBOLS and symbol not in self.OPERATIONAL_SYMBOLS:
            raise ValueError(f"Invalid symbol: {symbol}")

        self._trees[level].update_node(nid=node_id, tag=symbol)

        if symbol not in self.TERMINAL_SYMBOLS:
            end_symbol = 'e' if level == 2 else 'n'

            self._trees[level].create_node(
                tag=end_symbol,
                identifier=GlobalCounter.next(),
                parent=node_id,
            )

            if symbol in self.DIVISION_SYMBOLS:
                self._trees[level].create_node(
                    tag=end_symbol,
                    identifier=GlobalCounter.next(),
                    parent=node_id,
                )

    # * Get the symbol of a gene
    def get_symbol(self, level: int, node_id: str):
        """
        Retrieves the symbol of a specified gene from a given tree level.

        This method accesses the tree structure to obtain the symbol associated with a gene identified by its node ID at the specified level. It provides a straightforward way to query the genetic information stored in the genome.

        Args:
            level (int): The level of the tree from which to retrieve the symbol.
            node_id (str): The identifier of the node whose symbol is to be retrieved.

        Returns:
            str: The symbol of the specified gene.

        """
        return self._trees[level].get_node(node_id).tag

    # * Print a specific level or all of them
    def print(self, level: int = None):
        """
        Prints the structure of the genome's trees at a specified level or all levels.

        This method displays the tree structures contained within the genome. If a specific level is provided, it prints only that level; otherwise, it prints all levels of trees.

        Args:
            level (int, optional): The level of the tree to print. If None, all levels are printed. Defaults to None.

        """
        if level is None:
            for tree in self._trees:
                tree.show(idhidden=False)

        elif level >= 0 and level < self.LEVELS:
            self._trees[level].show(idhidden=False)

    # * Get the subtree from a starting point
    def get_genome_from_starting_point(self, node_id: str):
        """
        Retrieves the subtree of the genome starting from a specified node.

        This method collects and returns the subtree from the first tree at the given node ID while including the remaining trees unchanged. It allows for focused exploration of the genome structure from a specific starting point.

        Args:
            node_id (str): The identifier of the node from which to start the subtree retrieval.

        Returns:
            list: A list containing the subtree from the starting node and the other trees.

        """
        trees = []

        for i in range(len(self._trees)):
            if i == 0:
                trees.append(self._trees[i].subtree(nid=node_id))
            else:
                trees.append(self._trees[i])

        return trees

    # * Get the left subtree
    def get_left_child_genome(self):
        """
        Retrieves the genome structure starting from the left child of the root node.

        This method identifies the left child of the root node in the first tree and returns the subtree starting from that child. It provides a way to explore the genetic information specifically from the left branch of the genome.

        Returns:
            list: A list containing the subtree from the left child and the other trees.

        """

        root = self._trees[0].root

        left_child = self._trees[0].children(root)[0].identifier
        return self.get_genome_from_starting_point(left_child)

    # * Get the right subtree
    def get_right_child_genome(self):
        """
        Retrieves the genome structure starting from the right child of the root node.

        This method identifies the right child of the root node in the first tree and returns the subtree starting from that child. It allows for focused exploration of the genetic information specifically from the right branch of the genome.

        Returns:
            list: A list containing the subtree from the right child and the other trees.

        """
        root = self._trees[0].root

        right_child = self._trees[0].children(root)[1].identifier
        return self.get_genome_from_starting_point(right_child)

    # * Get the symbol of the root node
    def get_root_symbol(self):
        return self._trees[0].get_node(self._trees[0].root).tag

    # * Cut the first level
    def jump(self):
        """
        Cuts off the first level of the genome and adds a new tree.

        This method removes the first tree from the genome and creates a new tree initialized with a starting symbol. It effectively allows for a reset of the first level while retaining the remaining structure of the genome.

        Returns:
            list: A list containing the remaining trees and the newly created tree.

        """
        tree = Tree()
        tree.create_node(
            tag=self.STARTING_SYMBOL,
            identifier=GlobalCounter.next(),
            parent=None,
        )
        return self._trees[1:] + [tree]

    # * Get the number of trees in the genome
    def get_trees(self):
        return self._trees

    # * Get the number of levels in the genome
    def get_levels(self):
        return self.LEVELS

    # * Get the tree at a specific level
    def get_tree(self, level):
        """
        Retrieves the list of trees in the genome.

        This method provides access to the internal collection of trees that represent the genome structure. It allows users to view or manipulate the trees as needed.

        Returns:
            list: The list of trees in the genome.

        """
        return self._trees[level]

    # * Update the identifiers of the nodes
    def update_ids(self):
        """
        Updates the identifiers of all nodes in the genome's trees.

        This method iterates through each tree in the genome and assigns new unique identifiers to all nodes. It ensures that each node has a fresh identifier, which can be useful for maintaining uniqueness in the genome structure.

        Returns:
            None
        """

        for tree in self._trees:
            node_ids = [node.identifier for node in tree.all_nodes_itr()]
            for node_id in node_ids:
                tree.update_node(node_id, identifier=GlobalCounter.next())

    def json(self):
        """
        Converts the genome structure into a JSON-serializable format.

        This method prepares the genome data for serialization by converting each tree into a JSON representation while excluding detailed node data. It also includes the parent information, making it suitable for storage or transmission in a structured format.

        Returns:
            dict: A dictionary containing the JSON representation of the genome and parent information.

        """

        return {
            "genome": [tree.to_json(with_data=False) for tree in self._trees],
            "parents": self.parents
        }

    def json_pickle(self):
        """
        Serializes the genome data using pickle.

        This method prepares the genome data for serialization by converting each tree into a pickled representation. It also includes the parent information, making it suitable for storage or transmission in a binary format.

        Returns:
            dict: A dictionary containing the pickled representation of the genome and parent information.
        """

        encoded_trees = [pickle.dumps(tree) for tree in self._trees]
        encoded_trees = [base64.b64encode(tree).decode(
            'utf-8') for tree in encoded_trees]

        return {
            "genome": encoded_trees,
            "parents": self.parents
        }

    def from_json_pickle(self, json_individual):
        """
        Deserializes the genome data from a pickled format.

        This method reconstructs the genome data from a pickled representation. It converts each pickled tree back into a tree object and updates the genome's tree list.

        Args:
            json_individual (dict): A dictionary containing the pickled representation of the genome and parent information.

        Returns:
            None
        """
        self._trees = [base64.b64decode(tree)
                       for tree in json_individual['genome']]
        self._trees = [pickle.loads(tree)
                       for tree in self._trees]

    def update_parents(self, a, b):
        """
        Updates the parent relationships in the genome.

        This method sets the parents of the genome to the specified values, allowing for the modification of parent information. It provides a way to manage and update the relationships between different elements in the genome structure.

        Args:
            a: The first parent to set.
            b: The second parent to set.

        Returns:
            None

        """
        self.parents = [a, b]
