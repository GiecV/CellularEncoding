from utils.visualizer import Visualizer
import matplotlib.pyplot as plt

v = Visualizer()

# paths = ['logs/parity_up_to_n.json', 'logs/parity_no_enf.json']

# v.create_boxplots(paths)
# v.create_sum_boxplots(paths)



color_map = {'logs/parity_up_to_n.json': '#d9e7dc',
                'logs/parity_no_enf.json': '#eb3e56'}
names = ['Enforced Jumps', 'No Enforced Jumps']

# Prepare figure and axes
plt.figure(figsize=(10, 6))

enf_mean = [1.3, 490.8, 325.4]
enf_std = [0.6, 663.2, 470.5]
no_enf_mean = [25.5, 1910.5, 2000]
no_enf_std = [17.0, 193.9, 0]

# Set a position counter for each bar
current_pos = 1  # Starting position for bar plots
x_positions = []
x_labels = []
offset = 0.16

plt.bar(1 - offset, enf_mean[0], yerr=enf_std[0], color=color_map['logs/parity_up_to_n.json'],
        width=0.3, capsize=5, edgecolor='black', zorder=3)
plt.text(1 - offset, enf_mean[0] + enf_std[0] + 2, f"{enf_mean[0]:.1f}±{enf_std[0]:.1f}",
                             ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', zorder=4)
plt.bar(1 + offset, no_enf_mean[0], yerr=no_enf_std[0], color=color_map['logs/parity_no_enf.json'],
        width=0.3, capsize=5, edgecolor='black', zorder=3)
plt.text(1 + offset, no_enf_mean[0] + no_enf_std[0] + 2, f"{no_enf_mean[0]:.1f}±{no_enf_std[0]:.1f}",
                             ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', zorder=4)
plt.bar(2 - offset, enf_mean[1], yerr=enf_std[1], color=color_map['logs/parity_up_to_n.json'],
        width=0.3, capsize=5, edgecolor='black', zorder=3)
plt.text(2 - offset, enf_mean[1] + enf_std[1] + 2, f"{enf_mean[1]:.1f}±{enf_std[1]:.1f}",
                             ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', zorder=4)
plt.bar(2 + offset, no_enf_mean[1], yerr=no_enf_std[1], color=color_map['logs/parity_no_enf.json'],
        width=0.3, capsize=5, edgecolor='black', zorder=3)
plt.text(2 + offset, no_enf_mean[1] + no_enf_std[1] + 2, f"{no_enf_mean[1]:.1f}±{no_enf_std[1]:.1f}",
                             ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', zorder=4)
plt.bar(3 - offset, enf_mean[2], yerr=enf_std[2], color=color_map['logs/parity_up_to_n.json'],
        width=0.3, capsize=5, edgecolor='black', zorder=3)
plt.text(3 - offset, enf_mean[2] + enf_std[2] + 2, f"{enf_mean[2]:.1f}±{enf_std[2]:.1f}",
                             ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', zorder=4)
plt.bar(3 + offset, no_enf_mean[2], yerr=no_enf_std[2], color=color_map['logs/parity_no_enf.json'],
        width=0.3, capsize=5, edgecolor='black', zorder=3)
plt.text(3 + offset, no_enf_mean[2] + no_enf_std[2] + 2, f"{no_enf_mean[2]:.1f}±{no_enf_std[2]:.1f}",
                             ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', zorder=4)

x_positions = [1, 2, 3]
x_labels = ['2', '3', '4']
plt.figure(1)
plt.xticks(x_positions, x_labels)

handles = [plt.Line2D([0], [0], color=color_map[file], lw=6)
            for file in color_map]
plt.legend(handles, names, title="Variants", loc="upper left")

# Custom legend and labels
plt.xlabel("Number of Inputs")
plt.ylabel("Avg Generations ± Std Dev")
plt.title("Average Generations to Achieve the Optimum at Each Stage")

plt.grid(axis='y', linestyle='-', linewidth=0.5, zorder=1)

# Show plot
plt.show()

plt.figure(2)

# Data for the second figure
variants = ['Variant 1', 'Variant 2']
success_rates = [70, 0]

# Plotting the bar chart
plt.bar(variants, success_rates, color=['#d9e7dc', '#eb3e56'], zorder=2,
        width=0.3, capsize=5, edgecolor='black')
plt.text(0, success_rates[0] + 2, f"{success_rates[0]}%", ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
plt.text(1, success_rates[1] + 2, f"{success_rates[1]}%", ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')


# Custom legend and labels for the second figure
plt.xlabel("Variants")
plt.ylabel("Success Rate (%)")
plt.title("Success Rates of Variants")

plt.ylim(0, 100)

plt.grid(axis='y', linestyle='-', linewidth=0.5, zorder=1)

# Show second plot
plt.show()