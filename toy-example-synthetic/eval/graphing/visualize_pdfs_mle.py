import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

import sys
for path in sys.path:
    if path.endswith("/toy-example-synthetic/eval/graphing"):
        sys.path.append(path.replace("/toy-example-synthetic/eval/graphing", "/"))


font_path = 'eval/graphing/Times_New_Roman.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

plt.rcParams.update({
    'font.size': 11,          # Adjust global font size
    'axes.titlesize': 14,     # Adjust title size
    'axes.labelsize': 12,     # Adjust axis label size
    'legend.fontsize': 10,    # Adjust legend font size
    'xtick.labelsize': 10,    # X-axis tick label size
    'ytick.labelsize': 10,    # Y-axis tick label size
    'axes.linewidth': 1.0,    # Adjust axes line width
    'grid.alpha': 0.6,        # Gridline transparency
})

b = 1000
def plot_model_and_true(data, data_true, ax, title, label, label_true, ylim=0.20):
    bin_edges = np.linspace(-1, 1, b + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    if label == 'Fourier-MLE':
        color = 'tab:blue'  
    else:
        color = 'tab:purple'

    ax.plot(bin_centers, data, color=color, linewidth=2, label=f"{label}")
    if label_true:
        ax.plot(bin_centers, data_true, color='tab:green', linewidth=2, label="True PDF")
    else:
        ax.plot(bin_centers, data_true, color='tab:green', linewidth=2)

    ax.set_ylim(0.0, ylim)
    ax.grid(True, linewidth=0.5)

    ax.legend(loc='upper right', fontsize=8)  # Adjust legend font size

    ax.set_title(title, fontsize=12)

def plot_combined_graphs(prefix, fourier, gmm, true, idxs, output_fname):
    fourier = [np.load(prefix + fourier[j])[idxs[j]] for j in range(3)]
    gmm = [np.load(prefix + gmm[j])[idxs[j]] for j in range(3)]
    true = [np.load(prefix + true[j])[idxs[j]] for j in range(3)]
    titles = ['Gaussian Dataset', 'GMM-2 Dataset', 'Beta Dataset']
    y_lim = [4.5, 4.5, 4.5]
    # Create a 3x2 grid of subplots
    fig, axes = plt.subplots(3, 2, figsize=(12, 5)) 

    # Set a suptitle for the figure, use padding to avoid overlap with plots
    fig.suptitle('Toy Example: Learned Conditional Distribution vs True Conditional Distribution', fontsize=18, y=0.95)

    # Plot for each dataset (fourier, linear, true)
    for i in range(len(fourier)):
        flag = i==0
        plot_model_and_true(gmm[i]*500, true[i]*500, axes[i, 0], titles[i], label='GMM-MLE', label_true=flag, ylim= y_lim[i])
        plot_model_and_true(fourier[i]*500, true[i]*500, axes[i, 1], titles[i], label='Fourier-MLE', label_true=flag, ylim=y_lim[i])

    # Add a single shared y-axis label
    fig.text(0.04, 0.5, 'Probability Density', va='center', rotation='vertical', fontsize=18)  # Single y-axis label
    
    # Adjust the layout to prevent overlapping and reduce spacing
    plt.tight_layout(pad=1.0, rect=[0.05, 0, 1, 0.95])  # Adjusted rect to leave space for the y-label
    
    # Save the plot as a high-resolution image suitable for submission
    plt.savefig(prefix + output_fname, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Change output directory
    output_dir = 'output/'

    # Example usage: choose which model to load from for each of the datasets
    # For each dataset, be sure to specify the same seed for fourier, linear, and true
    fourier = ['gaussian/fourier-mle/0.0/8/pmfs_1.npy', 'gmm2/fourier-mle/0.0/8/pmfs_1.npy', 'beta/fourier-mle/0.0/16/pmfs_42.npy']
    gmm = ['gaussian/gmm-mle/0.0/0/pmfs_1.npy', 'gmm2/gmm-mle/0.0/0/pmfs_1.npy', 'beta/gmm-mle/0.0/0/pmfs_42.npy']
    true = ['gaussian/true_mle_1.npy', 'gmm2/true_mle_1.npy','beta/true_mle_42.npy']

    # Specify which pmf to be visualized for each of the datasets (there are a total 1000 test pmfs)
    pmf_ixs = [350, 500, 400]  
    plot_combined_graphs(output_dir, fourier, gmm, true, pmf_ixs, "toy_mle_predicted_vs_true.png")
    print(f"Saved graph to {output_dir}")