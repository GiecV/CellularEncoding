import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import random

from phenotype import Phenotype


class NNFromGraph(nn.Module):

    # * Create the neural network from a graph and find input and output nodes
    def __init__(self, phenotype: Phenotype, depth=4, t=0, inputs=4, outputs=1):

        super(NNFromGraph, self).__init__()

        self.phenotype = phenotype
        self.input_ids = []
        self.output_ids = []
        self.depth = depth  # Can be changed for a better approximation

        self.t = self.phenotype.expand_inputs_and_outputs(inputs, outputs)
        while not self.phenotype.development_finished():
            self.phenotype.develop()
        self.graph = self.phenotype.structure

        adjacency_matrix = nx.adjacency_matrix(
            self.graph, weight='weight').todense()
        self.A = torch.tensor(adjacency_matrix)  # Adjacency matrix
        self.W = nn.Parameter(torch.clone(self.A.float()))  # Weights matrix

        # Get the input and output nodes
        for i, node in enumerate(self.graph.nodes):
            if self.graph.nodes[node]["type"] == "input":
                self.input_ids.append(i)
            if self.graph.nodes[node]["type"] == "output":
                self.output_ids.append(i)

    # * Forward pass means propagating information from the input nodes to the output nodes by doing a weighted sum of the input
    def forward(self, obs):

        if len(self.input_ids) < len(obs):
            raise ValueError('The observation is larger than the input')

        W = (
            torch.abs(self.W) * self.A
            # We don't want 0 weights to change (no structural modifications), so we mask the weights with the adjacency matrix (n*0=0) by element-wise multiplication
        )
        x = torch.zeros(len(self.graph.nodes))  # State of each node
        for i in range(self.depth):  # Propagate information one layer at a time
            x[self.input_ids] = (
                obs  # The value of the output is frozen to the value of the observation for avoiding information loss
            )
            x = torch.tanh(torch.matmul(W.T, x))  # Weighted sum of the input
            # x = torch.relu(torch.matmul(W.T, x))

        return x[self.output_ids]  # Return the output values

    # * Perturb the weights of the neural network
    def perturb_weights(self):

        possible_values = [-1, 0, 1]
        new_W = self.W.clone()

        # Randomly select a weight to modify
        i = random.randint(0, self.W.size(0) - 1)
        j = random.randint(0, self.W.size(1) - 1)

        # Modify the selected weight
        value = random.choice(possible_values)
        new_W[i, j] = value

        return new_W, i, j, value

    # * Evaluate the neural network with perturbed weights
    def evaluate(self, data, target):

        self.eval()
        with torch.no_grad():
            output = self.forward(data)
            loss = F.mse_loss(output, target)
        return loss.item()

    # * Stochastic hill climbing algorithm
    def stochastic_hill_climbing(self, data, target):

        changed = False

        best_loss = self.evaluate(data, target)  # Evaluate the initial weights
        best_weights = self.W.clone()  # Save the initial weights as best

        new_weights, predecessor, node, value = self.perturb_weights()
        self.W.data = new_weights
        new_loss = self.evaluate(data, target)  # Evaluate the new weights

        if new_loss < best_loss:  # If the new weights are better, save them
            changed = True
            best_loss = new_loss
            best_weights = new_weights.clone()
        else:  # Accept the new weights anyway with a probability of e^(-0.1)
            if random.random() < torch.exp(-0.1):
                changed = True
                best_loss = new_loss
                best_weights = new_weights.clone()

        self.W.data = best_weights  # Return the best weights

        if value == -1:
            symbol = "d"
        elif value == 0:
            symbol = "c"
        else:
            symbol = "i"

        predecessors = self.graph.predecessors(node)
        index_of_edge = predecessors.index(predecessor)

        return changed, index_of_edge, node, symbol
