import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

import sys
sys.path.append("..")

from smoothness_metric import get_smoothness_metric

# Load external font
font_path = 'scripts/eval/Times_New_Roman.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'


# Set global font properties for a professional appearance
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.linewidth': 1.2,  # Thicker axes lines
    'grid.alpha': 0.6,      # Slightly less prominent gridlines
})

b = 18  # Number of bins for histogram
emoji_font_path = 'scripts/eval/NotoEmoji-VariableFont_wght.ttf'
emoji_font = fm.FontProperties(fname=emoji_font_path)

x_labels = [
    "\U0001F6AB",
    "\U0001F579\n\U00002B05",
    "\U0001F579\n\U00002196",
    "\U0001F579\n\U00002B06",
    "\U0001F579\n\U00002197",
    "\U0001F579\n\U000027A1",
    "\U0001F579\n\U00002198",
    "\U0001F579\n\U00002B07",
    "\U0001F579\n\U00002199",
    "\U0001F52B\n\U00002B05",
    "\U0001F52B\n\U00002196",
    "\U0001F52B\n\U00002B06",
    "\U0001F52B\n\U00002197",
    "\U0001F52B\n\U000027A1",
    "\U0001F52B\n\U00002198",
    "\U0001F52B\n\U00002B07",
    "\U0001F52B\n\U00002199",
    "\U0001F52B"
]


def plot_histogram_with_line(data, target, max_value, ax, title, label, color):
    bin_edges = np.linspace(-1, 1, b + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot histogram bars with different colors
    for i in range(len(bin_centers)):
        ax.bar(bin_centers[i], data[i], width=2/b, color=color, alpha=0.7)

    # Set axis limits and labels
    ax.set_ylim(0.0, 1.1 * max_value)  # Y-axis range
    ax.set_ylabel('Probability Mass')

    # Set custom x-ticks and labels, with ha='center' to align them properly
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontproperties=emoji_font)  # Set custom labels for the bins and rotate them

    # Add a grid for better readability
    ax.grid(True, linewidth=0.5)

    # add green dot at label
    y = data[target] + 0.02 # adjust constant here to change how much higher green dot is
    ax.scatter(x=bin_centers[target], y=y, color="tab:green", label="Ground truth label")

    ax.legend()

    # Customize legend and title
    ax.set_title(title, fontsize=14)


def plot_combined_graphs(fourier, linear, target, idx, output_fname):

    # Load the data from numpy files
    fourier = np.load(fourier)[idx]
    linear = np.load(linear)[idx]
    target = np.load(target)[idx]

    # Create a 1x2 grid of subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Adjusted figure size

    # Set a suptitle for the figure, use padding to avoid overlap with plots
    fig.suptitle('Decision Transformer Example: Learned Next Action Distributions', fontsize=20, y=0.95)

    # Plot for each dataset (fourier, linear)
    max_value = max(max(fourier), max(linear))
    smoothness_linear = get_smoothness_metric(np.expand_dims(linear,axis=0))["L2"]["mean"]
    plot_histogram_with_line(linear, target, max_value, axes[0], f'Linear head; smoothness = {smoothness_linear:.4f}', label='Linear', color="tab:red")
    smoothness_fourier = get_smoothness_metric(np.expand_dims(fourier,axis=0))["L2"]["mean"]
    plot_histogram_with_line(fourier, target, max_value, axes[1], f'Fourier head; smoothness = {smoothness_fourier:.4f}', label='Fourier', color="tab:blue")

    # Adjust the layout to prevent overlapping and reduce spacing
    plt.tight_layout(pad=1.5)  # Reduce padding between subplots

    # Save the plot as a high-resolution image
    plt.savefig(output_fname, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    # Example usage: Load from numpy files
    multinomial_fourier     = 'output/fourier_8/test-multinomials-seed-123-epoch-0.npy'
    multinomial_linear      = 'output/linear/test-multinomials-seed-123-epoch-3.npy'
    target                  = 'output/linear/test-targets-seed-123-epoch-3.npy'

    # some interesting ones for 14: 51, 75, 109, 110, 133, 158, 176, 184, 197, 199!!, 205, 207, 
    # some interesting ones for 8:  21, 65, 77, 78, 108, 126(*), 195(*)
    for idx in range(100000):
        plot_combined_graphs(multinomial_fourier, multinomial_linear, target, idx, f"scripts/eval/atari_graph_PMFs-{idx}.png")
