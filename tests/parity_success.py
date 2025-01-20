import matplotlib.pyplot as plt
import numpy as np

# Data for the plot
categories = ['Cellular Encoding', 'PPO', 'NEAT', 'HyperNEAT']
# Example means for "No curriculum"
no_curriculum_means = [1.0, 0.956, 0.894, 0.83]
# Example means for "Curriculum"
curriculum_means = [1.0, .973, 1.0, 0.825]
# Example standard deviations
no_curriculum_std = [0.0, 0.045, 0.056, 0.168]
# Example standard deviations
curriculum_std = [0.0, 0.079, 0.0, 0.275]

# Number of categories
x = np.arange(len(categories))

# Bar width
width = 0.35

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars with error bars
bars1 = ax.bar(
    x - width / 2, no_curriculum_means, width, yerr=no_curriculum_std,
    capsize=5, label='No curriculum', color='#8296b5'
)
bars2 = ax.bar(
    x + width / 2, curriculum_means, width, yerr=curriculum_std,
    capsize=5, label='Curriculum', color='#ed5a6e'
)

# Labels and title
ax.set_xlabel('Technique')
ax.set_ylabel('Success')
ax.set_title('Success to solve 6-parity')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Annotate means ± std deviation
# for bars, means, std in zip([bars1, bars2], [no_curriculum_means, curriculum_means], [no_curriculum_std, curriculum_std]):
#     for bar, mean, deviation in zip(bars, means, std):
#         ax.text(
#             bar.get_x() + bar.get_width() / 2, bar.get_height(),
#             f'{mean:.1f} ± {deviation:.1f}',
#             ha='center', va='bottom', fontsize=10
#         )

# Show the plot
plt.tight_layout()
plt.show()
