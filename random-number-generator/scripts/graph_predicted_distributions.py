import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import json
import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple

# Configure matplotlib settings
def setup_matplotlib():
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

@dataclass
class DistributionData:
    true_distribution: List[float]
    predicted_distribution: List[float]
    tvd: float
    color: str
    title: str

class DistributionPlotter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        setup_matplotlib()

    def plot_single_distribution(self, data: DistributionData, ax, ymax: float):
        """Plot a single distribution comparison on given axis."""
        bin_edges = np.linspace(-1, 1, 21)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = 0.1

        ax.bar(bin_centers, data.predicted_distribution, width, 
               color=data.color, alpha=0.7,
               label=f"Predicted PMF\nTVD = {data.tvd:.4f}")
        
        ax.scatter(bin_centers, data.true_distribution, 
                  color='tab:green', s=100, zorder=3,
                  label="True PMF", marker='o')
        
        ax.set_xlim(-1, 1)
        ax.set_xticks(bin_edges)
        ax.set_xticklabels([f'{x:.1f}' for x in bin_edges], rotation=45)
        ax.set_ylim(0.0, ymax)
        ax.grid(True, linewidth=0.5)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title(data.title, fontsize=12)

    def create_figure(self, n_plots: int = 3) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Create a figure with specified number of subplots."""
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        fig.text(0.04, 0.5, 'Probability Mass', va='center', rotation='vertical', fontsize=14)
        fig.text(0.5, 0.02, 'Value', ha='center', fontsize=14)
        plt.tight_layout(pad=1.0, rect=[0.05, 0.08, 1, 0.95])
        return fig, axes

    def save_figure(self, fig: plt.Figure, filename: str):
        """Save figure and close it."""
        output_path = os.path.join(self.output_dir, filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved as: {output_path}")
        plt.close(fig)

class DistributionAnalyzer:
    def __init__(self, json_dir: str):
        self.json_dir = json_dir

    def load_json(self, filename: str, test_idx: str) -> dict:
        """Load JSON file and extract test data."""
        with open(os.path.join(self.json_dir, filename), 'r') as f:
            return json.load(f)[test_idx]

    def get_color_for_file(self, filename: str) -> str:
        """Determine color based on filename."""
        if filename.startswith("original"):
            return "tab:purple"
        elif "freqs-0" in filename:
            return "tab:red"
        return "tab:blue"

    def load_baseline_and_freq0(self, test_idx: str) -> Tuple[dict, dict]:
        """Load baseline and frequency 0 data."""
        baseline_data = None
        freq0_data = None
        
        for filename in os.listdir(self.json_dir):
            if not filename.endswith('.json'):
                continue
                
            if filename.startswith('original'):
                baseline_data = self.load_json(filename, test_idx)
            elif filename == 'epochs-16-freqs-0-aggregated.json':
                freq0_data = self.load_json(filename, test_idx)
        
        if baseline_data is None or freq0_data is None:
            raise ValueError("Could not find baseline or freq0 data files")
            
        return baseline_data, freq0_data

def process_frequency_comparison(test_idx: str, json_dir: str, output_dir: str):
    """Process and plot frequency comparisons."""
    analyzer = DistributionAnalyzer(json_dir)
    plotter = DistributionPlotter(output_dir)
    
    # Load baseline data
    baseline_data, freq0_data = analyzer.load_baseline_and_freq0(test_idx)
    true_dist = baseline_data['true_distribution']
    ymax = 1.2 * max(true_dist)
    
    # Process each frequency
    for freq in range(1, 13):
        freqn_filename = f'epochs-16-freqs-{freq}-aggregated.json'
        freqn_path = os.path.join(json_dir, freqn_filename)
        
        if not os.path.exists(freqn_path):
            print(f"Warning: Could not find data for frequency {freq}")
            continue
        
        freqn_data = analyzer.load_json(freqn_filename, test_idx)
        
        # Create distributions for plotting
        if freq == 1:
            model_name_str = f"Fourier Head ({freq} freq)"
        elif freq > 1:
            model_name_str = f"Fourier Head ({freq} freqs)"
        distributions = [
            DistributionData(
                true_distribution=true_dist,
                predicted_distribution=baseline_data['predicted_distributions']['median_tvd']['distribution'],
                tvd=baseline_data['predicted_distributions']['median_tvd']['tvd'],
                color="tab:orange",
                title="Llama-3.1-8B-instruct"
            ),
            DistributionData(
                true_distribution=true_dist,
                predicted_distribution=freq0_data['predicted_distributions']['median_tvd']['distribution'],
                tvd=freq0_data['predicted_distributions']['median_tvd']['tvd'],
                color="tab:red",
                title="Llama-3.1-8B-instruct + LoRA"
            ),
            DistributionData(
                true_distribution=true_dist,
                predicted_distribution=freqn_data['predicted_distributions']['median_tvd']['distribution'],
                tvd=freqn_data['predicted_distributions']['median_tvd']['tvd'],
                color="tab:blue",
                title=f"Llama-3.1-8B-instruct + LoRA + {model_name_str})"
            )
        ]
        
        # Create and save plot
        fig, axes = plotter.create_figure(3)
        for ax, dist in zip(axes, distributions):
            plotter.plot_single_distribution(dist, ax, ymax)
        
        output_filename = f"method_comparison_test_idx_{test_idx}_epochs-16-freqs-{freq}-aggregated.png"
        plotter.save_figure(fig, output_filename)

def process_min_median_max(test_idx: str, json_dir: str, output_dir: str):
    """Process and plot min/median/max comparisons for each file."""
    analyzer = DistributionAnalyzer(json_dir)
    plotter = DistributionPlotter(output_dir)
    
    for filename in os.listdir(json_dir):
        if not filename.endswith('.json'):
            continue
            
        data = analyzer.load_json(filename, test_idx)
        true_dist = data['true_distribution']
        ymax = 1.2 * max(true_dist)
        color = analyzer.get_color_for_file(filename)
        
        # Create distributions for min/median/max
        distributions = []
        for dist_type in ['min_tvd', 'median_tvd', 'max_tvd']:
            distributions.append(DistributionData(
                true_distribution=true_dist,
                predicted_distribution=data['predicted_distributions'][dist_type]['distribution'],
                tvd=data['predicted_distributions'][dist_type]['tvd'],
                color=color,
                title=f"{os.path.splitext(filename)[0]}\n{dist_type.replace('_', ' ').title()}"
            ))
        
        # Create and save plot
        fig, axes = plotter.create_figure(3)
        for ax, dist in zip(axes, distributions):
            plotter.plot_single_distribution(dist, ax, ymax)
            
        output_filename = f"min_median_max_test_idx_{test_idx}_{os.path.splitext(filename)[0]}.png"
        plotter.save_figure(fig, output_filename)

def main():
    parser = argparse.ArgumentParser(description="Generate distribution comparison plots.")
    parser.add_argument('--input_dir', type=str, required=True, 
                      help='Directory containing the JSON files')
    parser.add_argument('--output_dir', type=str, required=True, 
                      help='Directory to write the images')
    args = parser.parse_args()
    
    for test_idx in ["0", "1", "2", "3"]:
        process_frequency_comparison(test_idx, args.input_dir, args.output_dir)
        process_min_median_max(test_idx, args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()