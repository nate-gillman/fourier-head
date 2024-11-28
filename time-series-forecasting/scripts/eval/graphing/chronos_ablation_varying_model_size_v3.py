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

import sys
for pth in sys.path:
    if pth.endswith("time-series-forecasting/src"):
        sys.path.append(pth.replace("/src", "/scripts/train"))
        break

from t5_scaling_configs_v2 import t5_scaling_configs

MAX_RUN_IDX = 2

run_idxs = {
    "fourier" : {
        0 : [0],
        1 : [1],
        2 : [2],
        3 : [3],
    },
    "linear" : {
        0 : [4],
        1 : [5],
        2 : [6],
        3 : [7],
    },

}

def get_largest_checkpoint_idx(lst):
    lst_ints = [int(elt) for elt in lst]
    ckpt = str(max(lst_ints))
    return ckpt

def gather_metrics(base_path):
    metrics = {
        "linear": {"MASE": [], "WQL": [], "MASE_std": [], "WQL_std": [], "num_params" : []},
        "fourier": {"MASE": [], "WQL": [], "MASE_std": [], "WQL_std": [], "num_params" : []}
    }
    
    # Process each model type (linear and fourier)
    for model_type in ["linear", "fourier"]:
        # Process each dataset size
        for dataset_size_idx in range(MAX_RUN_IDX+1):

            if dataset_size_idx in range(4):

                run_indices = run_idxs[model_type][dataset_size_idx]
                num_params = t5_scaling_configs[dataset_size_idx]["num_params"]

                metrics[model_type]["num_params"].append(num_params)
                
                # Store metrics for this dataset size
                mase_values = []
                wql_values = []
                
                # Process each run for this dataset size
                for idx in run_indices:
                    file_path = f"{base_path}/run-{idx}/eval_dict-zero-shot.json"
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        ckpt = get_largest_checkpoint_idx(data.keys())
                        mase_values.append(data[ckpt]["MASE"])
                        wql_values.append(data[ckpt]["WQL"])
                    except FileNotFoundError:
                        print(f"Warning: Could not find file {file_path}")
                        continue
                
                # Calculate mean and std (if we have multiple runs)
                if len(mase_values) > 0:
                    metrics[model_type]["MASE"].append(np.mean(mase_values))
                    metrics[model_type]["WQL"].append(np.mean(wql_values))
                    
                    # For the last dataset size (10000000), we won't have std dev
                    if len(mase_values) > 1:
                        metrics[model_type]["MASE_std"].append(np.std(mase_values))
                        metrics[model_type]["WQL_std"].append(np.std(wql_values))
                    else:
                        metrics[model_type]["MASE_std"].append(0)
                        metrics[model_type]["WQL_std"].append(0)
            
            elif dataset_size_idx == 4:

                # hard code to results from the paper
                metrics[model_type]["MASE_std"].append(0)
                metrics[model_type]["WQL_std"].append(0)

                if model_type == "linear":
                    metrics[model_type]["MASE"].append(0.883)
                    metrics[model_type]["WQL"].append(0.750)
                elif model_type == "fourier":
                    metrics[model_type]["MASE"].append(0.852)
                    metrics[model_type]["WQL"].append(0.749)
            
    return metrics

def build_graph(metric_name, x_positions, linear_values, fourier_values, linear_std, fourier_std, output_fname):

    fig, ax = plt.subplots(figsize=(6, 3))

    DIVIDE_STD_BY = (1/0.67)  # this gives 50% confidence interval

    # Plot linear metric
    ax.plot(x_positions, linear_values, c="tab:red", label="Linear head")
    # Plot standard deviation tunnel for linear metric

    # Plot fourier metric
    ax.plot(x_positions, fourier_values, c="tab:blue", label="Fourier head")
    # Plot standard deviation tunnel for linear metric

    ax.set_xscale('log')

    ax.set_yscale('log')
    yticks = [0.8, 1.0, 1.2, 1.4]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:.1f}' for y in yticks])
    
    ax.set_ylabel(metric_name, fontsize=16)
    ax.grid(True, linewidth=0.3)
    
    # Set custom x-axis ticks and labels
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xticks(x_positions)
    x_labels = [r'1.25M', r'2.5M', r'5M', r'10M', r'20M'][:MAX_RUN_IDX+1]
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Quantity of Model Parameters (Millions)", fontsize=16)
    ax.legend(loc="best")
    
    plt.tight_layout()
    plt.savefig(output_fname, dpi=300)
    plt.close()

def main():

    base_path = "output/11-27-pt4-ablation-model-size"

    metrics = gather_metrics(base_path)

    x_positions = [1250000, 2500000, 5000000, 10000000, 20000000][:MAX_RUN_IDX+1]
    metrics["fourier"]["num_params"] = x_positions
    
    # Create MASE graph
    build_graph(
        "MASE",
        metrics["fourier"]["num_params"], # same for both
        metrics["linear"]["MASE"],
        metrics["fourier"]["MASE"],
        metrics["linear"]["MASE_std"],
        metrics["fourier"]["MASE_std"],
        "scripts/eval/graphing/chronos_ablation_varying_model_size_v3_MASE.png",
    )
    
    # Create WQL graph
    build_graph(
        "WQL",
        metrics["fourier"]["num_params"], # same for both
        metrics["linear"]["WQL"],
        metrics["fourier"]["WQL"],
        metrics["linear"]["WQL_std"],
        metrics["fourier"]["WQL_std"],
        "scripts/eval/graphing/chronos_ablation_varying_model_size_v3_WQL.png"
    )

if __name__ == "__main__":
    main()