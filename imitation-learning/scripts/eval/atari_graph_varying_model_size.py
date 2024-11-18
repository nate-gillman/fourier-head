import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator, LogFormatter
import numpy as np
import json
import os

from atari_expert_scores import get_human_normalized_score

# Keep original font settings
font_path = 'scripts/eval/Times_New_Roman.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

import sys
for path in sys.path:
    if path.endswith("/imitation-learning/scripts/eval"):
        sys.path.append(path.replace("/scripts/eval", "/"))

from mingpt.scale_configs import SCALE_CONFIGS

def get_returns_for_model(base_dir, model_type):
    """Calculate average of maximal returns for a given model directory."""
    returns_path = os.path.join(base_dir, model_type, "returns_dict.json")
    
    if not os.path.exists(returns_path):
        return None
    
    with open(returns_path, "r") as f:
        returns_dict = json.load(f)
    
    # Get maximum return for each run
    max_returns = []
    for returns in returns_dict.values():
        max_returns.append(max(returns))

    max_returns = np.asarray(max_returns)
    max_returns_normalized = 100*get_human_normalized_score("seaquest", max_returns)

    max_returns_mean = np.mean(max_returns_normalized)
    max_returns_std = np.std(max_returns_normalized)

    return max_returns_mean, max_returns_std

def analyze_seaquest_directories(ignore_indices=None):
    """
    Analyze returns across Seaquest-{i} directories for both Linear and Fourier-14 models.
    
    Args:
        ignore_indices (list or set): Indices to ignore in the analysis
    """
    if ignore_indices is None:
        ignore_indices = set()
    else:
        ignore_indices = set(ignore_indices)
    
    linear_returns = []
    fourier_returns = []
    linear_returns_std = []
    fourier_returns_std = []
    x_values = []  # Will store M_log_params values
    
    # Collect data for each directory
    for i in range(11):  # Only going up to 10 as per the provided SCALE_CONFIGS
        if i in ignore_indices:
            continue
            
        directory = f"output/Seaquest-{i}"
        if os.path.exists(directory):
            linear_return, linear_return_std = get_returns_for_model(directory, "linear")
            fourier_return, fourier_return_std = get_returns_for_model(directory, "fourier_14")
            
            if linear_return is not None and fourier_return is not None:
                linear_returns.append(linear_return)
                fourier_returns.append(fourier_return)
                linear_returns_std.append(linear_return_std)
                fourier_returns_std.append(fourier_return_std)
                x_values.append(SCALE_CONFIGS[i]["M_log_params"])
    
    return x_values, linear_returns, fourier_returns, linear_returns_std, fourier_returns_std

def create_graph(x_values, linear_returns, fourier_returns, linear_returns_std, fourier_returns_std, output_fname):
    """Create a graph comparing Linear and Fourier-14 returns."""
    fig, ax = plt.subplots(figsize=(8, 4))

    DIVIDE_STD_BY = (1/0.67)  # this gives 50% confidence interval
    
    # Plot linear returns
    ax.plot(x_values, linear_returns, c="tab:red", label="Linear")
    # Plot standard deviation tunnel for linear returns
    ax.fill_between(
        x_values, 
        np.array(linear_returns) - np.array(linear_returns_std)/DIVIDE_STD_BY, 
        np.array(linear_returns) + np.array(linear_returns_std)/DIVIDE_STD_BY, 
        color="tab:red", 
        alpha=0.2
    )

    # Plot fourier returns
    ax.plot(x_values, fourier_returns, c="tab:blue", label="Fourier-14")
    # Plot standard deviation tunnel for fourier returns
    ax.fill_between(
        x_values, 
        np.array(fourier_returns) - np.array(fourier_returns_std)/DIVIDE_STD_BY, 
        np.array(fourier_returns) + np.array(fourier_returns_std)/DIVIDE_STD_BY, 
        color="tab:blue", 
        alpha=0.2
    )
    
    # Set labels and title
    ax.set_xlabel("Model Parameters (Millions)", fontsize=16)
    ax.set_ylabel("Average Normalized Returns", fontsize=16)
    fig.suptitle("Impact of Model Size on Decision Transformer Returns: Seaquest", fontsize=16, y=0.95)
    
    # Configure grid and axis
    ax.grid(True, linewidth=0.3)
    
    # Set x-axis limits with padding
    min_x = 0.25
    max_x = 2.05
    ax.set_xlim(min_x, max_x)
    
    # Set major ticks every 0.1
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    
    # Add minor gridlines
    ax.grid(True, which='major', linestyle='-', alpha=0.6)
    
    # Add legend
    ax.legend(loc="lower right")
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the figure
    plt.savefig(output_fname, dpi=300)

def main():
    # Collect and analyze data, ignoring index 5
    x_values, linear_returns, fourier_returns, linear_returns_std, fourier_returns_std = analyze_seaquest_directories(ignore_indices=[5])
    
    # Create and save the graph
    output_fname = "scripts/eval/atari_graph_varying_model_size.png"
    create_graph(x_values, linear_returns, fourier_returns, linear_returns_std, fourier_returns_std, output_fname)

if __name__ == "__main__":
    main()