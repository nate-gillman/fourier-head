import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

import sys
for path in sys.path:
    if path.endswith("/toy-example-synthetic/eval/graphing"):
        sys.path.append(path.replace("/toy-example-synthetic/eval/graphing", "/"))

from smoothness_metric import get_smoothness_metric


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

b = 50
def plot_histogram_with_line(mse_val, data_true, ax, title, label, label_true, ylim=0.20):
    bin_edges = np.linspace(-1, 1, b + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    if label == 'Fourier':
        color = 'tab:blue'  
    elif label == 'Linear':
        color =  'tab:red'
    else:
        color = 'tab:purple'

    
     # Plot mse_val as an orange point on the x-axis if it's None
    ax.scatter(mse_val[0], mse_val[1], color='tab:orange', s=20, label='Pointwise Regression') 

    if label_true:
        ax.plot(bin_centers, data_true, color='tab:green', linewidth=2, label="True PDF")
    else:
        ax.plot(bin_centers, data_true, color='tab:green', linewidth=2)

    ax.set_ylim(0.0, ylim)
    #ax.set_ylabel('Probability Density')
    ax.grid(True, linewidth=0.5)

    ax.legend(loc='upper right', fontsize=8)  # Adjust legend font size

    ax.set_title(title, fontsize=12)

def plot_combined_graphs(prefix, true, idxs, output_fname):
    true = [[np.load(prefix + true[j])[i] for i in idxs[j]] for j in range(3)]
    
    titles = ['Gaussian Dataset', 'GMM-2 Dataset', 'Beta Dataset']
    y_lim = [0.21, 0.2, 0.26]
    mse = [[-0.6200, -0.3400, -0.1000], [0.10, -0.10, -0.46], [0.02,0.02,0.02]]
    y_pos = [0.185, 0.08, 0.15]
    # Create a 3x2 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(12, 5)) 

    # Set a suptitle for the figure, use padding to avoid overlap with plots
    #fig.suptitle('Toy Example: Learned Conditional Distribution vs True Conditional Distribution', fontsize=18, y=0.95)

    # Plot for each dataset (fourier, linear, true)
    for i in range(len(true)):
        flag = i==0
        plot_histogram_with_line((mse[i][0],y_pos[i]), true[i][0], axes[i, 0], titles[i], label='Linear', label_true=flag, ylim= y_lim[i])
        plot_histogram_with_line((mse[i][1],y_pos[i]), true[i][1], axes[i, 1], titles[i], label='GMM', label_true=flag,  ylim= y_lim[i])
        plot_histogram_with_line((mse[i][2],y_pos[i]), true[i][2], axes[i, 2], titles[i], label='Fourier', label_true=flag, ylim=y_lim[i])

    # Add a single shared y-axis label
    fig.text(0.04, 0.5, 'Probability Mass', va='center', rotation='vertical', fontsize=18)  # Single y-axis label
    
    # Adjust the layout to prevent overlapping and reduce spacing
    plt.tight_layout(pad=1.0, rect=[0.05, 0, 1, 0.95])  # Adjusted rect to leave space for the y-label
    
    # Save the plot as a high-resolution image suitable for submission
    plt.savefig(prefix + output_fname, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Change output directory
    output_dir = 'eval/graphing/saved_pmfs/'

    # Example usage: choose which model to load from for each of the datasets
    # For each dataset, be sure to specify the same seed for fourier, linear, and true
    true = ['gaussian/true_1.npy', 'gmm2/true_1.npy','beta/true_42.npy']

    # Specify which pmf to be visualized for each of the datasets (there are a total 1000 test pmfs)
    pmf_ixs = [[488, 150, 400], [50,100,331], [90,80,180]]
   # 70 50 130
   
    plot_combined_graphs(output_dir, true, pmf_ixs, "toy_predicted_vs_true_mse.png")
    print(f"Saved graph to {output_dir}")
