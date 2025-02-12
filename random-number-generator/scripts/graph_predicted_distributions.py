import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import json
import os
import argparse

# Font and style settings
font_path = 'data/Times_New_Roman.ttf'
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

def plot_distribution_comparison(true_dist, pred_dist, tvd, ax, title, ymax=0.20):
    """
    Plot histogram comparison between true and predicted distributions
    """
    # Create bin edges and centers
    bin_edges = np.linspace(-1, 1, 21)  # 11 points to create 10 bins from 0 to 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Center of each bin
    width = 0.1  # Width of bars (matches bin width)
    
    # Plot bars for predicted distribution
    ax.bar(bin_centers, pred_dist, width, color='tab:blue', alpha=0.7, 
          label=f"Predicted PMF\nTVD = {tvd:.4f}")
    
    # Plot dots for true distribution
    ax.scatter(bin_centers, true_dist, color='tab:green', s=100, zorder=3,
              label="True PMF", marker='o')
    
    # Set x-axis limits and ticks
    ax.set_xlim(-1, 1)
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([f'{x:.1f}' for x in bin_edges], rotation=45)
    
    ax.set_ylim(0.0, ymax)
    ax.grid(True, linewidth=0.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title(title, fontsize=12)

def process_json_file(json_path, output_dir, test_idx):
    """
    Process a single JSON file and create plots
    """
    # Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract data from first key (assuming structure as provided)
    experiment_data = data[test_idx]
    
    # Get distributions
    true_dist = experiment_data['true_distribution']
    pred_distributions = experiment_data['predicted_distributions']
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # File name without extension for title
    file_name = os.path.basename(json_path).replace('.json', '')

    y_max = 1.2 * max(true_dist)
    # Plot for min_tvd
    plot_distribution_comparison(
        true_dist,
        pred_distributions['min_tvd']['distribution'],
        pred_distributions['min_tvd']['tvd'],
        axes[0],
        f"{file_name}\nMin TVD",
        ymax=y_max
    )
    
    # Plot for median_tvd
    plot_distribution_comparison(
        true_dist,
        pred_distributions['median_tvd']['distribution'],
        pred_distributions['median_tvd']['tvd'],
        axes[1],
        f"{file_name}\nMedian TVD",
        ymax=y_max
    )
    
    # Plot for max_tvd
    plot_distribution_comparison(
        true_dist,
        pred_distributions['max_tvd']['distribution'],
        pred_distributions['max_tvd']['tvd'],
        axes[2],
        f"{file_name}\nMax TVD",
        ymax=y_max
    )
    
    # Add a single shared y-axis label
    fig.text(0.04, 0.5, 'Probability Mass', va='center', rotation='vertical', fontsize=14)
    
    # Add a single shared x-axis label
    fig.text(0.5, 0.02, 'Value', ha='center', fontsize=14)
    
    # Adjust layout
    plt.tight_layout(pad=1.0, rect=[0.05, 0.08, 1, 0.95])
    
    # Save plot
    output_path = os.path.join(output_dir, f"test_idx_{test_idx}_{file_name}_pred_dists.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {output_path}")
    plt.close()

def main(json_dir, output_dir, test_idx):
    """
    Process all JSON files in the specified directory
    """
    # Ensure the directory exists
    if not os.path.exists(json_dir):
        raise ValueError(f"Directory {json_dir} does not exist")
    
    # Process each JSON file
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            process_json_file(json_path, output_dir, test_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metric plots.")
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing the JSON files')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to write the image')
    args = parser.parse_args()
    
    for test_idx in ["0", "1", "2", "3"]:
        main(args.input_dir, args.output_dir, test_idx)