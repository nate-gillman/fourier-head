import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import json
import os

from atari_expert_scores import get_human_normalized_score
from scipy.ndimage import gaussian_filter1d

# Keep original font settings
font_path = 'scripts/eval/Times_New_Roman.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

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

def analyze_seaquest_directories():
    """Analyze returns across Seaquest-{i} directories for both Linear and Fourier-14 models."""
    linear_returns = []
    fourier_returns = []
    linear_returns_std = []
    fourier_returns_std = []
    indices = []
    
    # Collect data for each directory
    for i in range(11):
        directory = f"output/Seaquest-{i}"
        if os.path.exists(directory):
            linear_return, linear_return_std = get_returns_for_model(directory, "linear")
            fourier_return, fourier_return_std = get_returns_for_model(directory, "fourier_14")
            
            if linear_return is not None and fourier_return is not None:
                linear_returns.append(linear_return)
                fourier_returns.append(fourier_return)
                linear_returns_std.append(linear_return_std)
                fourier_returns_std.append(fourier_return_std)
                indices.append(i)
    
    return indices, linear_returns, fourier_returns, linear_returns_std, fourier_returns_std

def create_graph(indices, linear_returns, fourier_returns, linear_returns_std, fourier_returns_std, output_fname):
    """Create a graph comparing Linear and Fourier-14 returns."""
    fig, ax = plt.subplots(figsize=(8, 4))

    DIVIDE_STD_BY = (1/0.67) # this gives 50% confidence interval
    
    # Plot linear returns
    ax.plot(indices, linear_returns, c="tab:red", label="Linear")
    # Plot standard deviation tunnel for linear returns
    ax.fill_between(
        indices, 
        np.array(linear_returns) - np.array(linear_returns_std)/DIVIDE_STD_BY, 
        np.array(linear_returns) + np.array(linear_returns_std)/DIVIDE_STD_BY, 
        color="tab:red", 
        alpha=0.2
    )

    # Plot fourier returns
    ax.plot(indices, fourier_returns, c="tab:blue", label="Fourier-14")
    # Plot standard deviation tunnel for fourier returns
    ax.fill_between(
        indices, 
        np.array(fourier_returns) - np.array(fourier_returns_std)/DIVIDE_STD_BY, 
        np.array(fourier_returns) + np.array(fourier_returns_std)/DIVIDE_STD_BY, 
        color="tab:blue", 
        alpha=0.2
    )
    
    # Set labels and title
    ax.set_xlabel("Model Size", fontsize=16)
    ax.set_ylabel("Average Maximum Returns", fontsize=16)
    fig.suptitle("Impact of Model Size on Decision Transformer Returns: Seaquest", fontsize=16, y=0.95)
    
    # Configure grid and axis
    ax.grid(True, linewidth=0.3)
    ax.set_xlim(-0.25,10.25)  # Slightly wider than 0-18 for better visibility
    ax.xaxis.set_major_locator(MultipleLocator(1))  # Set x-ticks to integers
    
    # Add legend
    ax.legend(loc="lower right")
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the figure
    plt.savefig(output_fname, dpi=300)

def main():
    # Collect and analyze data
    indices, linear_returns, fourier_returns, linear_returns_std, fourier_returns_std = analyze_seaquest_directories()
    
    # Create and save the graph
    output_fname = "scripts/eval/atari_graph_varying_model_size.png"
    create_graph(indices, linear_returns, fourier_returns, linear_returns_std, fourier_returns_std, output_fname)

if __name__ == "__main__":
    main()