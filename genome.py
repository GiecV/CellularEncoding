from treelib import Node, Tree
import numpy as np
from counter import GlobalCounter


class Genome:

    LEVELS = 1  # How many trees to consider
    STARTING_SYMBOL = "e"  # Each newborn gene will start with the end symbol

    TERMINAL_SYMBOLS = ["e"]
    JUMPING_SYMBOLS = ["n1", "n2"]
    DIVISION_SYMBOLS = ["p", "s"]
    OPERATIONAL_SYMBOLS = ["w", "i", "d", "+", "-", "c", "r"]  # a,o

    SYMBOLS = (
        TERMINAL_SYMBOLS + JUMPING_SYMBOLS + DIVISION_SYMBOLS
        + OPERATIONAL_SYMBOLS
    )

    # * Create the list of trees if none is provided, otherwise use the provided trees
    def __init__(self, trees=None) -> None:

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

    # * Edit the symbol of a gene and create the proper amount of children
    def change_symbol(self, level: int, node_id: str, symbol: str):

        if symbol not in self.SYMBOLS and symbol[0] not in self.OPERATIONAL_SYMBOLS:
            raise ValueError(f"Invalid symbol: {symbol}")

        self._trees[level].update_node(nid=node_id, tag=symbol)

        if symbol not in self.TERMINAL_SYMBOLS and symbol not in self.JUMPING_SYMBOLS:
            self._trees[level].create_node(
                tag=self.STARTING_SYMBOL,
                identifier=GlobalCounter.next(),
                parent=node_id,
            )

            if symbol in self.DIVISION_SYMBOLS:
                self._trees[level].create_node(
                    tag=self.STARTING_SYMBOL,
                    identifier=GlobalCounter.next(),
                    parent=node_id,
                )

    # * Get the symbol of a gene
    def get_symbol(self, level: int, node_id: str):
        return self._trees[level].get_node(node_id).tag

    # * Print a specific level or all of them
    def print(self, level=None):
        if level is not None:
            if level >= 0 and level < self.LEVELS:
                self._trees[level].show(idhidden=False)
        else:
            for tree in self._trees:
                tree.show(idhidden=False)

    # * Get the subtree from a starting point
    def get_genome_from_starting_point(self, node_id: str):
        trees = []

        for i in range(len(self._trees)):
            if i == 0:
                trees.append(self._trees[i].subtree(nid=node_id))
            else:
                trees.append(self._trees[i])

        return trees

    # * Get the left subtree
    def get_left_child_genome(self):

        root = self._trees[0].root

        left_child = self._trees[0].children(root)[0].identifier
        trees = self.get_genome_from_starting_point(left_child)

        return trees

    # * Get the right subtree
    def get_right_child_genome(self):

        root = self._trees[0].root

        right_child = self._trees[0].children(root)[1].identifier
        trees = self.get_genome_from_starting_point(right_child)

        return trees

    # * Get the symbol of the root node
    def get_root_symbol(self):
        return self._trees[0].get_node(self._trees[0].root).tag

    # * Cut the first n levels
    def jump_to_other_level(self, n: str):
        n = int(n)
        return self._trees[n:]

    # * Get the number of trees in the genome
    def get_trees(self):
        return len(self._trees)
