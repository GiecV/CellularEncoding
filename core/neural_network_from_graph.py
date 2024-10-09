import torch
import torch.nn as nn
import networkx as nx

from core.phenotype import Phenotype


class NNFromGraph(nn.Module):

    def __init__(self, phenotype: Phenotype, depth=4, inputs=2, outputs=1):
        super(NNFromGraph, self).__init__()

        self.phenotype = phenotype
        self.input_ids = []
        self.output_ids = []
        self.depth = depth

        self.t = self.phenotype.expand_inputs_and_outputs(inputs, outputs)
        self.graph = self.phenotype.structure

        adjacency_matrix = nx.adjacency_matrix(
            self.graph, weight='weight').todense()
        self.A = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.W = nn.Parameter(torch.clone(self.A))
        self.thresholds = torch.zeros(len(self.graph.nodes), dtype=torch.float32)

        for i, node in enumerate(self.graph.nodes):
            if self.graph.nodes[node]["type"] == "input":
                self.input_ids.append(i)
            if self.graph.nodes[node]["type"] == "output":
                self.output_ids.append(i)
            self.thresholds[i] = self.graph.nodes[node].get("threshold", 0.0)

    def forward(self, obs):

        if len(self.input_ids) < len(obs):
            raise ValueError('The observation is larger than the input')

        W = (torch.abs(self.W) * self.A)
        x = torch.zeros(len(self.graph.nodes))
        zero_tensor = torch.tensor(0.0)
        one_tensor = torch.tensor(1.0)

        for _ in range(self.depth):
            x[self.input_ids] = obs
            x = torch.where((torch.matmul(W.T, x) - self.thresholds) < 1, zero_tensor, one_tensor)

        return x[self.output_ids].int()
