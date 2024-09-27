# Cellular Encoding Project

This repository contains the work I completed during my internship at the Creative AI Lab, IT University of Copenhagen. The project focuses on **cellular encoding**, a fascinating technique for evolving neural networks, inspired by biological developmental processes.

## Project Overview

The goal of this project is to explore and extend the concept of cellular encoding for evolving modular neural networks. The method allows for the **automatic synthesis of neural network architectures** using a cell rewriting process, much like how cells differentiate and develop in biological organisms.

### How Cellular Encoding Works

In cellular encoding, a **tree of operations** is used to define how the neural network grows and evolves. Each node in the tree represents an operation, such as a neuron splitting into two or modifying its weights. These operations mimic biological cell division, allowing the structure of the neural network to develop iteratively.

1. **Tree Representation:** The cellular encoding process starts with a tree, where each node corresponds to a growth or modification operation on a neuron.
2. **Neuron Splitting/Modification:** Each operation allows for neurons to either split into multiple neurons or update their weights based on predefined rules.
3. **Tree to Graph Conversion:** Once the tree has defined the growth of the network, it is transformed into a graph, where nodes represent neurons and edges represent synapses.
4. **Graph to Neural Network:** Finally, this graph is converted into a functional neural network capable of performing tasks.

At present, this network is being tested on simple tasks, such as balancing a pole in the **CartPole-v1 environment** from OpenAI's Gym.

## Inspiration

This work is heavily inspired by the following foundational papers:

- Gruau, F. (1994). *Automatic definition of modular neural networks*. Adaptive Behavior, 3(2), 151–183.
- Gruau, F. (1992). *Genetic synthesis of Boolean neural networks with a cell rewriting developmental process*. COGANN-92: International Workshop on Combinations of Genetic Algorithms and Neural Networks, pp. 55-74, doi: [10.1109/COGANN.1992.273948](https://doi.org/10.1109/COGANN.1992.273948).

## Technologies

- Neural Networks
- Genetic Algorithms
- Cellular Encoding
- OpenAI Gym (CartPole-v1)

## Acknowledgments

I would like to thank the Creative AI Lab at ITU Copenhagen for their support and guidance throughout this project.

---

Stay tuned for updates and improvements!

Giacomo