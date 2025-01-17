from treelib import Tree
from utils.counter import GlobalCounter
import base64
import pickle
import random


class Genome:
    """
    Represents a genome consisting of multiple trees. The nodes are operations to be performed on a neural network (NN).

    The Genome class manages a collection of trees, allowing for operations such as symbol changes, subtree retrieval,
    and printing of tree structures. It provides methods to manipulate and access the genetic information represented 
    in the trees.
    """

    def __init__(self, trees: list = None, parents: list = None) -> None:
        """
        Initializes a genome with a specified number of trees or uses provided trees.

        This constructor sets up the genome's tree structure. If no trees are provided, it creates a default set of 
        trees, each initialized with a single gene node.

        :param trees: A list of trees to initialize the genome with. Defaults to None.
        :type trees: list, optional
        :param parents: A parameter to specify parent relationships. Defaults to None.
        :type parents: list, optional
        """

        self.LEVELS = 3  # How many trees to consider
        self.STARTING_SYMBOL = "e"  # Each newborn gene will start with the end symbol
        self.TERMINAL_SYMBOLS = ["e", "n"]
        self.DIVISION_SYMBOLS = ["p", "s"]
        self.OPERATIONAL_SYMBOLS = ["w", "i", "d", "r", "t", "u", "z", "+", "-", "c"] #

        self.SYMBOLS = self.TERMINAL_SYMBOLS + \
            self.DIVISION_SYMBOLS + self.OPERATIONAL_SYMBOLS

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

    def change_symbol(self, level: int, node_id: str, symbol: str):
        """
        Changes the symbol of a specified gene and creates child nodes as necessary.

        This method updates the symbol of a gene at a given level and node ID. If the new symbol is not terminal, 
        it creates additional child nodes based on the symbol type, ensuring the tree structure remains valid.

        :param level: The level of the tree where the gene is located.
        :type level: int
        :param node_id: The identifier of the node whose symbol is to be changed.
        :type node_id: str
        :param symbol: The new symbol to assign to the gene.
        :type symbol: str
        :raises ValueError: If the provided symbol is not valid.
        :example: 
            >>> genome.change_symbol(0, "node0", "w")
        """
        if symbol[0] not in self.SYMBOLS and symbol[0] not in self.OPERATIONAL_SYMBOLS:
            raise ValueError(f"Invalid symbol: {symbol}")

        if symbol == 'z':
            self._trees[level].update_node(nid=node_id, tag=symbol+str(random.randint(-255, 255)))
        else:
            self._trees[level].update_node(nid=node_id, tag=symbol)

        if symbol not in self.TERMINAL_SYMBOLS:
            end_symbol = 'e' if level == 2 else 'n'
        # if symbol not in self.TERMINAL_SYMBOLS:
        #     end_symbol = 'e'

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

    def get_symbol(self, level: int, node_id: str):
        """
        Retrieves the symbol of a specified gene from a given tree level.

        :param level: The level of the tree from which to retrieve the symbol.
        :type level: int
        :param node_id: The identifier of the node whose symbol is to be retrieved.
        :type node_id: str
        :return: The symbol of the specified gene.
        :rtype: str
        """
        return self._trees[level].get_node(node_id).tag

    def print(self, level: int = None):
        """
        Prints the structure of the genome's trees at a specified level or all levels.

        :param level: The level of the tree to print. If None, all levels are printed.
        :type level: int, optional
        """
        if level is None:
            for tree in self._trees:
                tree.show(idhidden=False)

        elif level >= 0 and level < self.LEVELS:
            self._trees[level].show(idhidden=False)

    def get_genome_from_starting_point(self, node_id: str):
        """
        Retrieves the subtree of the genome starting from a specified node.

        :param node_id: The identifier of the node from which to start the subtree retrieval.
        :type node_id: str
        :return: A list containing the subtree from the starting node and the other trees.
        :rtype: list
        """
        trees = []

        for i in range(len(self._trees)):
            if i == 0:
                trees.append(self._trees[i].subtree(nid=node_id))
            else:
                trees.append(self._trees[i])

        return trees

    def get_left_child_genome(self):
        """
        Retrieves the genome structure starting from the left child of the root node.

        :return: A list containing the subtree from the left child and the other trees.
        :rtype: list
        """

        root = self._trees[0].root

        left_child = self._trees[0].children(root)[0].identifier
        return self.get_genome_from_starting_point(left_child)

    def get_right_child_genome(self):
        """
        Retrieves the genome structure starting from the right child of the root node.

        :return: A list containing the subtree from the right child and the other trees.
        :rtype: list
        """
        root = self._trees[0].root

        right_child = self._trees[0].children(root)[1].identifier
        return self.get_genome_from_starting_point(right_child)

    def get_root_symbol(self):
        """
        Retrieves the symbol of the root node.

        :return: The symbol of the root node in the first tree.
        :rtype: str
        """
        return self._trees[0].get_node(self._trees[0].root).tag

    def jump(self):
        """
        Cuts off the first level of the genome and adds a new tree.

        :return: A list containing the remaining trees and the newly created tree.
        :rtype: list
        """
        tree = Tree()
        tree.create_node(
            tag=self.STARTING_SYMBOL,
            identifier=GlobalCounter.next(),
            parent=None,
        )
        return self._trees[1:] + [tree]

    def get_trees(self):
        """
        Retrieves the genome's tree structure.

        :return: A list of all trees within the genome.
        :rtype: list
        """
        return self._trees

    def get_levels(self):
        """
        Retrieves the number of levels in the genome.

        :return: The total number of levels in the genome.
        :rtype: int
        """
        return self.LEVELS

    # * Get the tree at a specific level
    def get_tree(self, level: int):
        """
        Retrieves the tree at a specific level.

        :param level: The level of the tree to retrieve.
        :type level: int
        :return: The tree at the specified level.
        :rtype: Tree
        """
        return self._trees[level]

    # * Update the identifiers of the nodes
    def update_ids(self):
        """
        Updates the identifiers of all nodes in the genome's trees.

        This method assigns new unique identifiers to all nodes in each tree of the genome.

        :return: None
        :rtype: None
        """
        for tree in self._trees:
            node_ids = [node.identifier for node in tree.all_nodes_itr()]
            for node_id in node_ids:
                tree.update_node(node_id, identifier=GlobalCounter.next())

    def json_pickle(self):
        """
        Serializes the genome data using pickle.

        :return: A dictionary containing the pickled representation of the genome and parent information.
        :rtype: dict
        :example: 
            >>> genome.json_pickle()
            >>> # {'genome': ['...', '...', '...'], 'parents': None}
        """
        encoded_trees = [pickle.dumps(tree) for tree in self._trees]
        encoded_trees = [base64.b64encode(tree).decode(
            'utf-8') for tree in encoded_trees]

        return {
            "genome": encoded_trees,
            "parents": self.parents
        }

    def from_json_pickle(self, json_individual: dict):
        """
        Deserializes the genome data from a pickled format.

        :param json_individual: A dictionary containing the pickled representation of the genome and parent information.
        :type json_individual: dict
        :return: None
        :rtype: None
        :example: 
            >>> genome.from_json_pickle({'genome': ['...', '...', '...'], 'parents': None})
        """
        self._trees = [base64.b64decode(tree)
                       for tree in json_individual['genome']]
        self._trees = [pickle.loads(tree)
                       for tree in self._trees]
