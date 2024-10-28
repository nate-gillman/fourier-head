import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# Load external font
font_path = '../imitation-learning/scripts/eval/Times_New_Roman.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
import sys


import sys
for pth in sys.path:
    if pth.endswith("/time-series-forecasting/scripts/eval/graphing"):
        sys.path.append(pth.replace("/time-series-forecasting/scripts/eval/graphing", ""))
        break

from smoothness_metric import get_smoothness_metric


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

b = 4096  # Number of bins for histogram

def plot_histogram_with_line(data, ax, title, id_min, width, ymax, label, color):

    # import pdb; pdb.set_trace()

    data = data[id_min:id_min+width]
    num_bins = data.shape[0]

    left_limit = -1 + (2 * id_min / b)
    right_limit = -1 + (2 * (id_min + width) / b)
    bin_edges = np.linspace(left_limit, right_limit, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot histogram bars with different colors
    for i in range(len(bin_centers)):
        ax.bar(bin_centers[i], data[i], width=2/b, color=color, alpha=0.7)

    ax.set_xlim(left_limit, right_limit)
    # ax.set_xticks(np.arange(left_limit, right_limit, 0.1))

    # Set axis limits and labels
    ax.set_ylim(0.0, ymax)  # Y-axis range
    ax.set_ylabel('Probability Mass')

    # Add a grid for better readability
    ax.grid(True, linewidth=0.5)

    # Customize legend and title
    ax.set_title(title, fontsize=14)


def plot_combined_graphs(
        fourier, linear, output_fname, idx,
        id_min_fourier, width_fourier, ymax_fourier,
        id_min_linear, width_linear, ymax_linear,
    ):
    # Load the data from numpy files

    # Create a 1x2 grid of subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Adjusted figure size

    # Set a suptitle for the figure, use padding to avoid overlap with plots
    fig.suptitle('Chronos Example: Learned Token Distribution', fontsize=20, y=0.95)

    # Plot for each dataset (fourier, linear)
    smoothness_linear = get_smoothness_metric(np.expand_dims(linear,axis=0))["L2"]["mean"]
    
    plot_histogram_with_line(linear, axes[0], f'Linear head; smoothness = {smoothness_linear:.4f}', id_min_fourier, width_fourier, ymax_fourier, label='Linear', color="tab:red")
    
    smoothness_fourier = get_smoothness_metric(np.expand_dims(fourier,axis=0))["L2"]["mean"]
    plot_histogram_with_line(fourier, axes[1], f'Fourier head; smoothness = {smoothness_fourier:.4f}', id_min_linear, width_linear, ymax_linear, label='Fourier', color="tab:blue")

    # Adjust the layout to prevent overlapping and reduce spacing
    plt.tight_layout(pad=1.5)  # Reduce padding between subplots

    # Save the plot as a high-resolution image
    plt.savefig(output_fname, dpi=300, bbox_inches='tight')
    plt.show()


def main():

    # EXAMPLE 1
    # idx = 40
    # id_min_fourier, width_fourier, ymax_fourier = 2120, 200, 0.025
    # id_min_linear, width_linear, ymax_linear = 1175, 200, 0.0125
    # fourier = ['output/09-10-4096-8xA100-1100-1e-6/multinomials/monash_hospital.npy']
    # linear = ['output/07-07-mini-linear-8gpu-fixed-copy/multinomials/monash_hospital.npy']
    # plot_combined_graphs(
    #     fourier, linear, f"scripts/graphing-chronos/chronos-PMFs.png",
    #     idx,
    #     id_min_fourier, width_fourier, ymax_fourier,
    #     id_min_linear, width_linear, ymax_linear,
    # )

    idx = 200
    id_min_fourier, width_fourier, ymax_fourier = 2300, 125, 0.020
    id_min_linear, width_linear, ymax_linear = 1580, 125, 0.012

    multinomial_fourier = np.load("scripts/eval/graphing/multinomial_fourier.npy")
    multinomial_linear = np.load("scripts/eval/graphing/multinomial_linear.npy")
    # fourier = ['output/09-10-4096-8xA100-1100-1e-6/multinomials/monash_tourism_monthly.npy']
    # linear = ['output/07-07-mini-linear-8gpu-fixed-copy/multinomials/monash_tourism_monthly.npy']
    plot_combined_graphs(
        multinomial_fourier,
        multinomial_linear,
        f"scripts/eval/graphing/chronos-PMF.png",
        idx,
        id_min_fourier, width_fourier, ymax_fourier,
        id_min_linear, width_linear, ymax_linear,
    )






if __name__ == "__main__":
    main()
