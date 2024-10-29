from treelib import Tree
from utils.counter import GlobalCounter


class Genome:

    LEVELS = 3  # How many trees to consider
    STARTING_SYMBOL = "e"  # Each newborn gene will start with the end symbol
    TERMINAL_SYMBOLS = ["e", "n"]
    DIVISION_SYMBOLS = ["p", "s"]
    OPERATIONAL_SYMBOLS = ["w", "i", "d", "+", "-", "c", "r", "t"]

    SYMBOLS = TERMINAL_SYMBOLS + DIVISION_SYMBOLS + OPERATIONAL_SYMBOLS

    # * Create the list of trees if none is provided, otherwise use the provided trees
    def __init__(self, trees: list = None, parents=None) -> None:

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
        return self._trees[level].get_node(node_id).tag

    # * Print a specific level or all of them
    def print(self, level: int = None):
        if level is None:
            for tree in self._trees:
                tree.show(idhidden=False)

        elif level >= 0 and level < self.LEVELS:
            self._trees[level].show(idhidden=False)

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
        return self.get_genome_from_starting_point(left_child)

    # * Get the right subtree
    def get_right_child_genome(self):
        root = self._trees[0].root

        right_child = self._trees[0].children(root)[1].identifier
        return self.get_genome_from_starting_point(right_child)

    # * Get the symbol of the root node
    def get_root_symbol(self):
        return self._trees[0].get_node(self._trees[0].root).tag

    # * Cut the first level
    def jump(self):
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
        return self._trees[level]

    # * Update the identifiers of the nodes
    def update_ids(self):
        for tree in self._trees:
            node_ids = [node.identifier for node in tree.all_nodes_itr()]
            for node_id in node_ids:
                tree.update_node(node_id, identifier=GlobalCounter.next())

    def json(self):
        return {
            "genome": [tree.to_json(with_data=False) for tree in self._trees],
            "parents": self.parents
        }

    def update_parents(self, a, b):
        self.parents = [a, b]
