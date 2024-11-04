import torch
import torch.nn as nn
import networkx as nx

from core.phenotype import Phenotype

torch.set_num_threads(1)


class NNFromGraph(nn.Module):
    """
    Neural network implementation based on a phenotype's graph structure.

    This class constructs a neural network using the structure defined in a phenotype, allowing for 
    the propagation of inputs through the network. It initializes the network parameters and manages 
    the forward pass of data through the network layers.

    Args:
        phenotype (Phenotype): The phenotype that defines the structure of the neural network.
        depth (int, optional): The number of propagation layers in the network. Defaults to 7.
        inputs (int, optional): The number of input nodes. Defaults to 2.
        outputs (int, optional): The number of output nodes. Defaults to 1.
    """

    def __init__(self, phenotype: Phenotype, depth=7, inputs=2, outputs=1):
        """
        Initialize the neural network from a phenotype's graph structure.

        This constructor sets up the neural network by extracting the structure from the provided phenotype,
        initializing parameters, and preparing the adjacency matrix for the network. It also identifies input
        and output nodes based on the phenotype's configuration.

        Args:
            phenotype (Phenotype): The phenotype that defines the structure of the neural network.
            depth (int, optional): The number of propagation layers in the network. Defaults to 7.
            inputs (int, optional): The number of input nodes. Defaults to 2.
            outputs (int, optional): The number of output nodes. Defaults to 1.

        Returns:
            None
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

    def forward(self, obs):
        """
        Propagate the input observations through the neural network.

        This method takes the input observations and processes them through the network for a specified number 
        of layers, applying weights and thresholds to determine the output. It raises an error if the input 
        observations exceed the expected size.

        Args:
            obs: A tensor containing the input observations to be processed by the network.

        Raises:
            ValueError: If the observation size is larger than the expected input size.

        Returns:
            Tensor: The output of the network corresponding to the specified output nodes.
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
