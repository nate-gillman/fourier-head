import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import json
import glob
import os
import argparse

# Match original font settings
font_path = 'data/Times_New_Roman.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

FONTSIZE = 19

def load_json_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_metrics(data, metric_type, key="0"):
    """Extract metrics based on the specified type"""
    if metric_type == "tvd":
        return data[key]["tvd_error_mean"], data[key]["tvd_error_sem"]
    elif metric_type == "num_unique_samples":
        return data[key]["num_unique_samples_mean"], data[key]["num_unique_samples_sem"]

def build_graph(data_dir, output_fname, metric_type, max_num_freqs, test_idx):
    # Get all json files in directory
    json_files = glob.glob(os.path.join(data_dir, "*-aggregated.json"))
    
    # Initialize lists to store data points
    freqs = []
    means = []
    sems = []

    # Load baseline data (no fine-tuning)
    baseline_file = os.path.join(data_dir, "original-model-baseline-aggregated.json")
    baseline_data = load_json_data(baseline_file)
    baseline_mean, baseline_sem = extract_metrics(baseline_data, metric_type, test_idx)
    
    # Load LoRA data (freqs-0)
    LoRA_file = os.path.join(data_dir, "epochs-16-freqs-0-aggregated.json")
    LoRA_data = load_json_data(LoRA_file)
    LoRA_mean, LoRA_sem = extract_metrics(LoRA_data, metric_type, test_idx)
    
    # Load LoRA + Fourier Head Data; process each frequency file...
    for json_file in json_files:
        if "freqs-0-" not in json_file and "baseline" not in json_file:
            # Extract frequency number from filename
            freq = int(json_file.split("-freqs-")[-1].split("-")[0])
            data = load_json_data(json_file)
            mean, sem = extract_metrics(data, metric_type, test_idx)
            
            freqs.append(freq)
            means.append(mean)
            sems.append(sem)

    # Sort by frequency
    sorted_indices = np.argsort(freqs)
    freqs = np.array(freqs)[sorted_indices]
    means = np.array(means)[sorted_indices]
    sems = np.array(sems)[sorted_indices]

    # Create plot with same style as original
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Set title and axis labels based on metric type
    if metric_type == "tvd":
        title = "Distance Between Predicted Distribution and True Distribution"
        ylabel = "TVD Error"
        ylim = None
    elif metric_type == "num_unique_samples":  # num_unique_samples
        title = "Quantity of Samples that are Unique Values"
        ylabel = "Quantity of Samples"
        min_y = min([baseline_mean] + [LoRA_mean] + means.tolist())
        max_y = max([baseline_mean] + [LoRA_mean] + means.tolist())
        ylim = (min_y-5, max_y+20)
    else:
        raise NotImplementedError
    
    # Global title
    fig.suptitle(title, fontsize=26, y=0.99)
    
    # Plot baseline datapoint as horizontal line
    ax.axhline(y=baseline_mean, color='tab:purple', linestyle='-.', label='Llama-3.1-8B-instruct')
    ax.fill_between(freqs,
                    baseline_mean - baseline_sem,
                    baseline_mean + baseline_sem,
                    color='tab:purple', alpha=0.2)

    # Plot LoRA datapoint as horizontal line
    ax.axhline(y=LoRA_mean, color='tab:red', linestyle='--', label='Llama-3.1-8B-instruct + LoRA')
    ax.fill_between(freqs,
                    LoRA_mean - LoRA_sem,
                    LoRA_mean + LoRA_sem,
                    color='tab:red', alpha=0.2)
    
    # Plot LoRA + Fourier Head datapoints
    ax.plot(freqs, means, c="tab:blue", label='Llama-3.1-8B-instruct + LoRA + Fourier Head')
    ax.fill_between(freqs,
                    means - sems,
                    means + sems,
                    color='tab:blue', alpha=0.2)
    
    # Match original axis styling
    ax.set_xlabel("Fourier Frequencies", fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.grid(True, linewidth=0.3)
    ax.set_xlim((1, max_num_freqs))
    if ylim:
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    
    # Match original legend styling
    ax.legend(loc="upper right", fontsize=FONTSIZE)
    ax.tick_params(axis='x', labelsize=FONTSIZE)
    ax.tick_params(axis='y', labelsize=FONTSIZE)
    
    # Match original layout adjustments
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    plt.savefig(output_fname, dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metric plots.")
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing the JSON files')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to write the image')
    parser.add_argument('--metric', type=str, required=True, choices=['tvd', 'num_unique_samples'],
                        help='Metric to plot: tvd (Total Variation Distance) or num_unique_samples')
    parser.add_argument('--max_num_freqs', type=int, required=True,
                        help='Largest value to graph on the x axis')
    args = parser.parse_args()

    # Set output filename based on metric type
    os.makedirs(args.output_dir, exist_ok=True)
    
    for test_idx in ["0", "1", "2", "3"]:
        output_fname = os.path.join(args.output_dir, f"test_idx_{test_idx}_graph_metrics_{args.metric}.png")
        build_graph(args.input_dir, output_fname, args.metric, args.max_num_freqs, test_idx)
        print(f'Graph saved as: {output_fname}')