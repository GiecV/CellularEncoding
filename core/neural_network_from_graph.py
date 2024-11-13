import torch
import torch.nn as nn
import networkx as nx

from core.phenotype import Phenotype

torch.set_num_threads(1)


class NNFromGraph(nn.Module):
    """
    A neural network model constructed from a graph representation of a phenotype.

    This class creates a neural network based on a given graph structure, assigning
    weights, thresholds, and other properties to the nodes and edges of the graph.
    The `forward` method performs a custom feedforward pass through the network 
    based on the graph structure.

    :param Phenotype phenotype: The phenotype object containing the graph structure.
    :param int depth: The number of iterations for forward propagation (default is 7).
    :param int inputs: The number of input nodes (default is 2).
    :param int outputs: The number of output nodes (default is 1).

    **Attributes:**

    - **phenotype** (*Phenotype*): The phenotype object containing the graph structure.
    - **input_ids** (*list*): List of input node indices in the graph.
    - **output_ids** (*list*): List of output node indices in the graph.
    - **depth** (*int*): Number of iterations for the forward propagation.
    - **t** (*int*): 
    - **r** (*int*): 
    - **A** (*torch.Tensor*): Adjacency matrix of the graph.
    - **W** (*torch.Tensor*): Trainable weight matrix for the graph.
    - **thresholds** (*torch.Tensor*): Thresholds for each node in the graph.
    """

    def __init__(self, phenotype: Phenotype, depth: int = 7, inputs: int = 2, outputs: int = 1):
        """
        Initializes the NNFromGraph model with a phenotype and graph parameters.

        This method sets up the model's graph structure, adjacency matrix,
        weight matrix, and node thresholds.

        Args:
            phenotype (Phenotype): The phenotype object containing the structure of the graph.
            depth (int, optional): Number of iterations for the forward propagation. Defaults to 7.
            inputs (int, optional): Number of input nodes. Defaults to 2.
            outputs (int, optional): Number of output nodes. Defaults to 1.
        """
        super(NNFromGraph, self).__init__()

        self.phenotype = phenotype
        self.input_ids = []
        self.output_ids = []
        self.depth = depth

        self.t, self.r = self.phenotype.expand_inputs_and_outputs(
            inputs, outputs)
        self.graph = self.phenotype.structure

        adjacency_matrix = nx.adjacency_matrix(
            self.graph, weight='weight').todense()
        self.A = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.W = nn.Parameter(torch.clone(self.A))
        self.thresholds = torch.zeros(
            len(self.graph.nodes), dtype=torch.float32)

        for i, node in enumerate(self.graph.nodes):
            if self.graph.nodes[node]["type"] == "input":
                self.input_ids.append(i)
            if self.graph.nodes[node]["type"] == "output":
                self.output_ids.append(i)
            self.thresholds[i] = self.graph.nodes[node].get("threshold", 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the NNFromGraph model.

        The forward method propagates the input tensor through the graph-based
        network, updating node activations based on thresholds and adjacency
        weights for each depth iteration.

        Args:
            obs (torch.Tensor): Input tensor with observations for each input node.

        Returns:
            torch.Tensor: Output tensor representing the activations of the output nodes.

        Raises:
            ValueError: If the observation tensor has more elements than input nodes.

        Example:
            >>> phenotype = Phenotype()
            >>> model = NNFromGraph(phenotype)
            >>> obs = torch.tensor([1.0, 0.0])
            >>> output = model.forward(obs)
        """
        if len(self.input_ids) < len(obs):
            raise ValueError('The observation is larger than the input')

        W = (torch.abs(self.W) * self.A)
        x = torch.zeros(len(self.graph.nodes))
        zero_tensor = torch.tensor(0.0)
        one_tensor = torch.tensor(1.0)

        for _ in range(self.depth):
            x[self.input_ids] = obs
            x = torch.where((torch.matmul(W.T, x) - self.thresholds)
                            < 1, zero_tensor, one_tensor)

        return x[self.output_ids].int()
