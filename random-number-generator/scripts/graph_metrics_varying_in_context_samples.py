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
    elif metric_type == "containment":
        return data[key]["containment_mean"], data[key]["containment_sem"]

def build_graph(data_dir, output_fname, metric_type, freqs_to_graph, max_num_in_context_samples_per_prompt, test_idx):
    
    # Load baseline data (no fine-tuning)
    baseline_means, baseline_sems = [], []
    for i in range(0, max_num_in_context_samples_per_prompt + 1):
        baseline_file = os.path.join(data_dir, f"{i:02d}_in_context_samples_per_prompt", "original-model-baseline-aggregated.json")
        baseline_data = load_json_data(baseline_file)
        baseline_mean, baseline_sem = extract_metrics(baseline_data, metric_type, test_idx)
        baseline_means.append(baseline_mean), baseline_sems.append(baseline_sem)
    baseline_means = np.asarray(baseline_means)
    baseline_sems = np.asarray(baseline_sems)
    
    # Load LoRA data (freqs = 0)
    LoRA_means, LoRA_sems = [], []
    for i in range(0, max_num_in_context_samples_per_prompt + 1):
        LoRA_file = os.path.join(data_dir, f"{i:02d}_in_context_samples_per_prompt", "epochs-16-freqs-0-aggregated.json")
        LoRA_data = load_json_data(LoRA_file)
        LoRA_mean, LoRA_sem = extract_metrics(LoRA_data, metric_type, test_idx)
        LoRA_means.append(LoRA_mean), LoRA_sems.append(LoRA_sem)
    LoRA_means = np.asarray(LoRA_means)
    LoRA_sems = np.asarray(LoRA_sems)

    # Load LoRA + Fourier Head data (freqs > 0)
    means, sems = {}, {}
    for freq in freqs_to_graph:
        # Initialize lists to store data points
        means_ = []
        sems_ = []
        for i in range(0, max_num_in_context_samples_per_prompt + 1):
            # Get all json files in directory
            json_files = glob.glob(os.path.join(data_dir, f"{i:02d}_in_context_samples_per_prompt", "*-aggregated.json"))
            # Load LoRA + Fourier Head Data; process each frequency file...
            for json_file in json_files:
                if f"-freqs-{freq}-aggregated.json" in json_file:
                    # Extract frequency number from filename
                    data = load_json_data(json_file)
                    mean, sem = extract_metrics(data, metric_type, test_idx)
                    means_.append(mean)
                    sems_.append(sem)
                continue
        means[freq] = np.asarray(means_)
        sems[freq] = np.asarray(sems_)

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

        min_y = min(min(baseline_means), min(LoRA_means))
        for freq in freqs_to_graph:
            min_y = min(min_y, min(means[freq]))
        max_y = max(max(baseline_means), max(LoRA_means))
        for freq in freqs_to_graph:
            max_y = max(max_y, max(means[freq]))

        ylim = (min_y-5, max_y+20)

    elif metric_type == "containment":
        title = "Fraction of Samples Contained in [0,1]"
        ylabel = "Containment"
        ylim = None
    else:
        raise NotImplementedError
    
    # Global title
    fig.suptitle(title, fontsize=26, y=0.99)
    x_axis = np.arange(0, max_num_in_context_samples_per_prompt + 1)
    
    # Plot baseline datapoint as horizontal line
    ax.plot(x_axis, baseline_means, c="tab:orange", linestyle='-.', label='Llama-3.1-8B-instruct')
    ax.fill_between(x_axis,
                    baseline_means - baseline_sems,
                    baseline_means + baseline_sems,
                    color='tab:orange', alpha=0.2)

    # Plot LoRA datapoint as horizontal line
    ax.plot(x_axis, LoRA_means, color='tab:red', linestyle='--', label='Llama-3.1-8B-instruct + LoRA')
    ax.fill_between(x_axis,
                    LoRA_means - LoRA_sems,
                    LoRA_means + LoRA_sems,
                    color='tab:red', alpha=0.2)
    
    # Plot LoRA + Fourier Head datapoints
    for freq in freqs_to_graph:
        ax.plot(x_axis, means[freq], c="tab:blue", label=f'Llama-3.1-8B-instruct + LoRA + Fourier Head ({freq} freqs)')
        ax.fill_between(x_axis,
                        means[freq] - sems[freq],
                        means[freq] + sems[freq],
                        color='tab:blue', alpha=0.2)
    
    # Match original axis styling
    ax.set_xlabel("Quantity of In-Context Samples in Prompt", fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.grid(True, linewidth=0.3)
    ax.set_xlim((0, max_num_in_context_samples_per_prompt))
    if ylim:
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    
    # Match original legend styling
    ax.legend(loc="upper right", fontsize=11)
    ax.tick_params(axis='x', labelsize=FONTSIZE)
    ax.tick_params(axis='y', labelsize=FONTSIZE)
    
    # Match original layout adjustments
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    plt.savefig(output_fname, dpi=300)
    plt.close()

# Define a custom argument type for a list of strings
def list_of_ints(arg):
    the_list = [int(elt) for elt in arg.split(',')]
    return the_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metric plots.")
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing the JSON files')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to write the image')
    parser.add_argument('--metric', type=str, required=True, choices=['tvd', 'num_unique_samples', 'containment'],
                        help='Metric to plot: tvd (Total Variation Distance) or num_unique_samples or containment')
    parser.add_argument('--freq', type=int, required=True,
                        help='The frequency to graph')
    parser.add_argument('--max_num_in_context_samples_per_prompt', type=int, required=True,
                        help='Largest value to graph on the x axis')
    args = parser.parse_args()

    # Set output filename based on metric type
    os.makedirs(args.output_dir, exist_ok=True)
    
    for test_idx in ["0", "1", "2", "3", "agg"]:
        output_fname = os.path.join(args.output_dir, f"test_idx_{test_idx}_metric_{args.metric}_freq_{args.freq}.png")
        build_graph(args.input_dir, output_fname, args.metric, [args.freq], args.max_num_in_context_samples_per_prompt, test_idx)
        print(f'Graph saved as: {output_fname}')