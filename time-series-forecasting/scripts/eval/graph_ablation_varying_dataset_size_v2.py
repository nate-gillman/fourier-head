import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import json
import os

font_path = '../imitation-learning/scripts/eval/Times_New_Roman.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

CHECKPOINT = [
    "20000", "40000", "60000", "80000", "100000", 
    "120000", "140000", "160000", "180000", "200000"
][9] # 2, 3, 7!

def get_largest_checkpoint_idx(lst):

    lst_ints = [int(elt) for elt in lst]
    ckpt = str(max(lst_ints))

    return ckpt

def gather_metrics():
    linear_runs = [f"run-{i}" for i in range(15, 20)]
    fourier_runs = [f"run-{i}" for i in range(20, 25)]
    
    metrics = {
        "linear": {"MASE": [], "WQL": []},
        "fourier": {"MASE": [], "WQL": []}
    }
    
    # Gather Linear metrics
    print("Gathering metrics for linear runs...")
    for run in linear_runs:
        file_path = f"output/11-15-dataset-size-ablations-v2/{run}/eval_dict-zero-shot.json"
        with open(file_path, "r") as f:
            data = json.load(f)
        ckpt = get_largest_checkpoint_idx(data.keys())
        print(ckpt)
        metrics["linear"]["MASE"].append(data[ckpt]["MASE"])
        metrics["linear"]["WQL"].append(data[ckpt]["WQL"])
    
    # Gather Fourier metrics
    print("Gathering metrics for Fourier runs...")
    for run in fourier_runs:
        file_path = f"output/11-15-dataset-size-ablations-v2/{run}/eval_dict-zero-shot.json"
        with open(file_path, "r") as f:
            data = json.load(f)
        ckpt = get_largest_checkpoint_idx(data.keys())
        print(ckpt)
        metrics["fourier"]["MASE"].append(data[ckpt]["MASE"])
        metrics["fourier"]["WQL"].append(data[ckpt]["WQL"])
    
    return metrics

def build_graph(metric_name, linear_values, fourier_values, output_fname):
    fig, ax = plt.subplots(figsize=(6, 3))
    
    x_positions = list(range(1, 6))
    
    # Plot individual points
    ax.plot(x_positions, linear_values, 'o-', color='tab:red', label="Linear head")
    ax.plot(x_positions, fourier_values, 'o-', color='tab:blue', label="Fourier head")
    
    ax.set_ylabel(metric_name, fontsize=16)
    ax.grid(True, linewidth=0.3)
    ax.set_xticks(x_positions)
    ax.set_xlabel("-2 + log_10(dataset size), i.e. sizes are 1000, 10000, ...", fontsize=16)
    ax.legend(loc="best")
    
    plt.tight_layout()
    plt.savefig(output_fname, dpi=300)
    plt.close()

def main():
    metrics = gather_metrics()
    
    # Create MASE graph
    build_graph(
        "MASE",
        metrics["linear"]["MASE"],
        metrics["fourier"]["MASE"],
        "scripts/eval/graph_ablation_varying_dataset_size_MASE.png"
    )
    
    # Create WQL graph
    build_graph(
        "WQL",
        metrics["linear"]["WQL"],
        metrics["fourier"]["WQL"],
        "scripts/eval/graph_ablation_varying_dataset_size_WQL.png"
    )

if __name__ == "__main__":
    main()