import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import random

from cellular_encoding.phenotype import Phenotype


class NNFromGraph(nn.Module):

    # * Create the neural network from a graph and find input and output nodes
    def __init__(self, phenotype: Phenotype, depth=4, inputs=2, outputs=1, has_bias=False):
        super(NNFromGraph, self).__init__()

        self.phenotype = phenotype
        self.input_ids = []
        self.output_ids = []
        self.bias_id = None
        self.depth = depth  # Can be changed for a better approximation
        self.has_bias = has_bias

        self.t = self.phenotype.expand_inputs_and_outputs(inputs, outputs, has_bias)
        self.graph = self.phenotype.structure

        adjacency_matrix = nx.adjacency_matrix(
            self.graph, weight='weight').todense()
        self.A = torch.tensor(adjacency_matrix, dtype=torch.float32)  # Adjacency matrix
        self.W = nn.Parameter(torch.clone(self.A))  # Weights matrix

        # Get the input and output nodes
        for i, node in enumerate(self.graph.nodes):
            if self.graph.nodes[node]["type"] == "input":
                self.input_ids.append(i)
            if self.graph.nodes[node]["type"] == "output":
                self.output_ids.append(i)

    # * Forward pass means propagating information from the input nodes to the output nodes by doing a weighted sum of the input
    def forward(self, obs):

        if self.has_bias:
            obs = torch.cat((obs, torch.tensor([1.0])))  # Add the bias value to the input

        if len(self.input_ids) < len(obs):
            raise ValueError('The observation is larger than the input')

        # We don't want 0 weights to change (no structural modifications), so we mask the weights with the adjacency matrix (n*0=0) by element-wise multiplication
        W = (torch.abs(self.W) * self.A)
        x = torch.zeros(len(self.graph.nodes))  # State of each node

        for i in range(self.depth):  # Propagate information one layer at a time
            # The value of the output is frozen to the value of the observation for avoiding information loss
            x[self.input_ids] = (obs)
            x = torch.tanh(torch.matmul(W.T, x))  # Weighted sum of the input

        return x[self.output_ids]  # Return the output values
