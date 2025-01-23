import numpy as np
import matplotlib.pyplot as plt

# Define the range of input values
inputs = np.linspace(-5, 5, 500)

# Update the threshold
threshold = 0

# Recompute the outputs
step_output = np.where(inputs - threshold < 1, 0, 1)
tanh_output = np.tanh(inputs - threshold)

# Plot the updated step function output
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(inputs, step_output, label="step", color="blue")
plt.title("Output of a Binary Neuron (Threshold = 0)")
plt.xlabel("Sum of Inputs - Threshold")
plt.ylabel("Output")
plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
plt.grid(True)
plt.legend()

# Plot the updated tanh function output
plt.subplot(1, 2, 2)
plt.plot(inputs, tanh_output, label="tanh", color="green")
plt.title("Output of a Continuous Neuron (Threshold = 0)")
plt.xlabel("Sum of Inputs - Threshold")
plt.ylabel("Output")
plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()