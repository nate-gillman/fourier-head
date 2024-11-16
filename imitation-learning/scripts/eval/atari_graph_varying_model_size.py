import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import json
import os

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
    
    return np.mean(max_returns)

def analyze_seaquest_directories():
    """Analyze returns across Seaquest-{i} directories for both Linear and Fourier-14 models."""
    linear_returns = []
    fourier_returns = []
    indices = []
    
    # Collect data for each directory
    for i in range(19):  # 0 to 18 inclusive
        directory = f"output/Seaquest-{i}"
        if os.path.exists(directory):
            linear_return = get_returns_for_model(directory, "linear")
            fourier_return = get_returns_for_model(directory, "fourier_14")
            
            if linear_return is not None and fourier_return is not None:
                linear_returns.append(linear_return)
                fourier_returns.append(fourier_return)
                indices.append(i)
    
    return indices, linear_returns, fourier_returns

def create_graph(indices, linear_returns, fourier_returns, output_fname):
    """Create a graph comparing Linear and Fourier-14 returns."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot both returns
    ax.plot(indices, linear_returns, c="red", label="Linear")
    ax.plot(indices, fourier_returns, c="tab:blue", label="Fourier-14")
    
    # Set labels and title
    ax.set_xlabel("Model Size", fontsize=16)
    ax.set_ylabel("Average Maximum Returns", fontsize=16)
    fig.suptitle("Impact of Model Size on Decision Transformer Returns", fontsize=16, y=0.95)
    
    # Configure grid and axis
    ax.grid(True, linewidth=0.3)
    ax.set_xlim(-1, 19)  # Slightly wider than 0-18 for better visibility
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
    indices, linear_returns, fourier_returns = analyze_seaquest_directories()
    
    # Create and save the graph
    output_fname = "scripts/eval/atari_graph_varying_model_size.png"
    create_graph(indices, linear_returns, fourier_returns, output_fname)

if __name__ == "__main__":
    main()