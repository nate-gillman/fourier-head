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

idxs = {
    "linear" : {
        0 : {
            "run_idxs" : [0],
            "num_params" : 16
        },
        1 : {
            "run_idxs" : [2],
            "num_params" : 20
        },
        2 : {
            "run_idxs" : [3],
            "num_params" : 46
        },
        # 3 : {
        #     "run_idxs" : [4],
        #     "num_params" : 200
        # },
    },
    "fourier" : {
        0 : {
            "run_idxs" : [1],
            "num_params" : 16
        },
        1 : {
            "run_idxs" : [6],
            "num_params" : 20
        },
        2 : {
            "run_idxs" : [10],
            "num_params" : 46
        },
        # 3 : {
        #     "run_idxs" : [5],
        #     "num_params" : 200
        # },
    },
}

def get_largest_checkpoint_idx(lst):
    lst_ints = [int(elt) for elt in lst]
    ckpt = str(max(lst_ints))
    return ckpt

def gather_metrics():
    metrics = {
        "linear": {"MASE": [], "WQL": [], "MASE_std": [], "WQL_std": [], "num_params" : []},
        "fourier": {"MASE": [], "WQL": [], "MASE_std": [], "WQL_std": [], "num_params" : []}
    }
    
    # Process each model type (linear and fourier)
    for model_type in ["linear", "fourier"]:
        # Process each dataset size
        for dataset_size_idx in [0, 1, 2]:
            run_indices = idxs[model_type][dataset_size_idx]["run_idxs"]
            num_params = idxs[model_type][dataset_size_idx]["num_params"]

            metrics[model_type]["num_params"].append(num_params)
            
            # Store metrics for this dataset size
            mase_values = []
            wql_values = []
            
            # Process each run for this dataset size
            for idx in run_indices:
                file_path = f"output/run-{idx}/eval_dict-zero-shot.json"
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
            
    return metrics

def build_graph(metric_name, x_positions, linear_values, fourier_values, linear_std, fourier_std, output_fname):

    fig, ax = plt.subplots(figsize=(6, 3))

    DIVIDE_STD_BY = (1/0.67)  # this gives 50% confidence interval

    

    # Plot linear metric
    ax.plot(x_positions, linear_values, c="tab:red", label="Linear head")
    # Plot standard deviation tunnel for linear metric
    ax.fill_between(
        x_positions, 
        np.array(linear_values) - np.array(linear_std)/DIVIDE_STD_BY, 
        np.array(linear_values) + np.array(linear_std)/DIVIDE_STD_BY, 
        color="tab:red", 
        alpha=0.2
    )

    # Plot fourier metric
    ax.plot(x_positions, fourier_values, c="tab:blue", label="Fourier head")
    # Plot standard deviation tunnel for linear metric
    ax.fill_between(
        x_positions, 
        np.array(fourier_values) - np.array(fourier_std)/DIVIDE_STD_BY, 
        np.array(fourier_values) + np.array(fourier_std)/DIVIDE_STD_BY, 
        color="tab:blue", 
        alpha=0.2
    )
    
    ax.set_ylabel(metric_name, fontsize=16)
    ax.grid(True, linewidth=0.3)
    
    # Set custom x-axis ticks and labels
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    # x_labels = [r'16M', r'20M', r'46M']
    ax.xaxis.set_major_locator(MultipleLocator(10))
    # ax.set_xticklabels(x_labels)
    ax.set_xlabel("Quantity of Model Parameters (Millions)", fontsize=16)
    ax.legend(loc="best")
    
    plt.tight_layout()
    plt.savefig(output_fname, dpi=300)
    plt.close()

def main():
    metrics = gather_metrics()
    
    # Create MASE graph
    build_graph(
        "MASE",
        metrics["fourier"]["num_params"], # same for both
        metrics["linear"]["MASE"],
        metrics["fourier"]["MASE"],
        metrics["linear"]["MASE_std"],
        metrics["fourier"]["MASE_std"],
        "scripts/eval/graphing/chronos_ablation_varying_model_size_MASE.png"
    )
    
    # Create WQL graph
    build_graph(
        "WQL",
        metrics["fourier"]["num_params"], # same for both
        metrics["linear"]["WQL"],
        metrics["fourier"]["WQL"],
        metrics["linear"]["WQL_std"],
        metrics["fourier"]["WQL_std"],
        "scripts/eval/graphing/chronos_ablation_varying_model_size_WQL.png"
    )

if __name__ == "__main__":
    main()