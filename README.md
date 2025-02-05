# Cellular Encoding Project

Welcome to the **Cellular Encoding Project**, a deep dive into evolutionary computation and neural architecture synthesis. This repository showcases work exploring **cellular encoding** — a fascinating method that leverages genetic principles to develop neural network architectures automatically.

## 🚀 Project Overview

This project extends the boundaries of **cellular encoding**, allowing us to evolve modular neural networks that can adapt and scale, inspired by biological processes of cell differentiation and development. Through cellular encoding, we’re not just training a fixed neural network; we’re evolving its structure dynamically, mimicking how organisms grow and adapt in nature.

Key objectives of this project include:

- **Automatic Synthesis of Neural Networks**: Generate network architectures that evolve and improve autonomously, adapting to the tasks they need to solve.
- **Modularity and Scalability**: Develop neural networks that can grow more complex as the problem demands, a key advantage in handling complex, variable environments.

### 🔬 How Cellular Encoding Works

Cellular encoding is an elegant process that uses **evolutionary algorithms** to grow neural networks from a simple set of rules. Here's how it works:

1. **Tree Representation**: Cellular encoding begins with a tree of operations. Each node represents a specific transformation, like splitting neurons or modifying weights. This tree structure lets us evolve complex architectures in a controlled, scalable way.
   
2. **Neuron Splitting and Modification**: Through each node operation, neurons split or alter their characteristics, echoing the biological processes of cell division and differentiation.

3. **Genetic Evolution**: Using genetic algorithms, we evolve these trees across generations. The fittest architectures — those that solve tasks effectively — are selected and evolved, creating increasingly optimized neural networks.

This process embodies the principles of natural selection, allowing neural networks to "grow" and adapt iteratively to meet the demands of various tasks.

## 🌱 Inspiration

This project draws inspiration from seminal research on cellular encoding and developmental neural networks. Notably:

- Gruau, F. (1994). *Automatic Definition of Modular Neural Networks*. Adaptive Behavior, 3(2), 151–183.
- Gruau, F. (1992). *Genetic Synthesis of Boolean Neural Networks with a Cell Rewriting Developmental Process*. COGANN-92, 55-74. [Read here](https://doi.org/10.1109/COGANN.1992.273948).

These papers provide the theoretical backbone for cellular encoding, demonstrating its potential in creating scalable, adaptable neural networks.

## 🛠️ Technologies

The Cellular Encoding Project utilizes a variety of technologies to support the efficient evolution and analysis of neural networks:

- **Neural Networks**: The core of the project, leveraging neural networks as the building blocks for evolutionary structures.
  
- **Genetic Algorithms**: Driving the evolutionary process, with selection, crossover, and mutation fostering adaptability and improvement over generations.

- **Parallel Processing**: To make full use of available computational resources, we leverage parallel processing using Python’s `multiprocessing` library, allowing the evolutionary process to scale efficiently on both local and remote systems.

- **Visualization (Matplotlib)**: Visualizations are key for understanding network growth and performance metrics. The project uses `Matplotlib` to provide graphical insights into the evolutionary process and the structure of generated networks.

- **Logging with JSON**: For a transparent record of the evolutionary process, the project logs all key metrics, configurations, and outcomes in structured JSON format, making it easy to analyze or replicate experiments.

- **Documentation with Sphinx**: The documentation for the Cellular Encoding Project is built with `Sphinx`, ensuring that all modules and functionalities are well-documented and accessible on [GitHub Pages](https://giecv.github.io/CellularEncoding/).

- **OpenAI Gym for Reinforcement Learning Tasks**: `Gym` provides a range of environments that can be used to test the effectiveness of evolved neural networks, especially for reinforcement learning tasks. This allows the project to evaluate how well cellular encoding performs on dynamic, real-world scenarios.

## 📄 Documentation

Explore the [documentation site on GitHub Pages](https://giecv.github.io/CellularEncoding/), which provides:

- **Core Modules**: Detailed breakdowns of the main components.
- **Task Modules**: Descriptions of task environments used for testing and validation.
- **Utilities**: Tools developed for the project, including visualizations for better analysis and understanding.

---

Stay tuned for ongoing improvements and new applications of cellular encoding. This project is a step toward making neural networks that don’t just learn — they evolve.
