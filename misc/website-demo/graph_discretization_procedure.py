import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import matplotlib.font_manager as fm
import os
from tqdm import tqdm

# Set up the same matplotlib styling
font_path = '../../toy-example-synthetic/eval/graphing/Times_New_Roman.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 1.0,
    'grid.alpha': 0.6,
})

def get_y_limits(data_dict, key):
    """Calculate the maximum y value needed for all three plots"""
    multinomial_max = max(data_dict[str(key)]["multinomial"])
    learned_pdf_max = max(data_dict[str(key)]["learned_pdf"])
    true_pdf_max = max(point[1] for point in data_dict["pdf"])
    return max(multinomial_max, learned_pdf_max, true_pdf_max) * 1.1  # Add 10% padding

def generate_ticks(max_val, step=0.005):
    """Generate ticks that are multiples of step up to max_val"""
    # Find how many steps fit within max_val
    n_steps = int(max_val / step)
    return np.arange(0, (n_steps + 1) * step, step)

def plot_single_distribution(data_dict, key, plot_type, y_limit):
    """Create a single plot for learned PDF, true PDF, or multinomial"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if plot_type == "learned_pdf":
        # Plot learned PDF in blue
        x_pdf = torch.linspace(-1, 1, 1000)
        title = f"Learned density, {key} frequencies"
        ylabel = 'Probability Density'
        ax.plot(x_pdf, 
               data_dict[str(key)]["learned_pdf"], 
               color='tab:blue', 
               alpha=0.7,
               linewidth=2, 
               label=title)
        
    elif plot_type == "true_pdf":
        # Plot true PDF in green using the provided x,y coordinates
        x_values = [point[0] for point in data_dict["pdf"]]
        y_values = [point[1] for point in data_dict["pdf"]]
        title = "Ground truth density"
        ylabel = 'Probability Density'
        ax.plot(x_values, 
               y_values, 
               color='tab:green', 
               linewidth=2, 
               label=title)
        
    else:  # plot_type == "multinomial"
        # Plot multinomial histogram
        bin_edges = np.linspace(-1, 1, 128 + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        title = f"Learned and discretized density, {key} frequencies"
        ylabel = 'Probability Mass'
        ax.bar(bin_centers, 
              data_dict[str(key)]["multinomial"], 
              width=np.diff(bin_edges)[0], 
              color='tab:blue', 
              alpha=0.7,
              label=title)

    # Set custom tick locations
    x_ticks = np.linspace(-1, 1, 5)  # 5 ticks from -1 to 1
    y_ticks = generate_ticks(y_limit, 0.005)
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    # Format tick labels to avoid too many decimal places
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, y_limit)
    ax.grid(True, linewidth=0.5)
    ax.legend(loc='upper right', fontsize=20)
    # ax.set_title(f'{title}', fontsize=12)
    
    # Add axis labels
    ax.set_xlabel('Latent value', fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_dir = "output/discretization_graphs"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'distribution_{key}_{plot_type}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_separate_distribution_plots(data_file):
    # Load the JSON data
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # For each frequency, create all three types of plots
    for key in tqdm(range(1, 65)):
        # Calculate shared y-limit for all plots
        y_limit = get_y_limits(data, key)
        
        # Create separate plots with matching y-axes
        plot_single_distribution(data, key, "learned_pdf", y_limit)
        plot_single_distribution(data, key, "true_pdf", y_limit)
        plot_single_distribution(data, key, "multinomial", y_limit)

if __name__ == "__main__":
    # Specify your JSON file path
    json_file = "output/data-mixture-of-gaussians-v2.json"
    create_separate_distribution_plots(json_file)