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

X_POSITIONS = [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]

idxs_v1 = {
    "linear" : {
        20000 : [25],
        40000 : [35],
        60000 : [43],
        80000 : [27],
        100000 : [39],
        120000 : [33],
        140000 : [42],
        160000 : [49],
    },
    "fourier" : {
        20000 : [20],
        40000 : [30],
        60000 : [40],
        80000 : [21],
        100000 : [32],
        120000 : [26],
        140000 : [38],
        160000 : [44],
    },
}

idxs = {
    "linear" : {
        20000 : [8, 9, 10, 11],
        40000 : [20, 21, 22, 23],
        60000 : [32, 33, 34, 35],
        80000 : [44, 45, 46, 48],
        100000 : [56, 57, 58, 59],
        120000 : [72, 73, 74, 75],
        140000 : [16, 17, 18, 19],
        160000 : [38, 40, 42, 43],
        180000 : [61, 63, 65, 67],
        200000 : [76, 77, 78, 79],
    },
    "fourier" : {
        20000 : [0, 2, 4, 6],
        40000 : [12, 13, 14, 15],
        60000 : [25, 27, 29, 31],
        80000 : [36, 37, 39, 41],
        100000 : [50, 52, 54, 55],
        120000 : [60, 62, 64, 66],
        140000 : [1, 3, 5, 7],
        160000 : [24, 26, 28, 30],
        180000 : [47, 49, 51, 53],
        200000 : [69, 69, 70, 71],
    },
}

def get_largest_checkpoint_idx(lst):
    lst_ints = [int(elt) for elt in lst]
    ckpt = str(max(lst_ints))
    return ckpt

def gather_metrics(base_path):
    metrics = {
        "linear": {"MASE": [], "WQL": [], "MASE_std": [], "WQL_std": []},
        "fourier": {"MASE": [], "WQL": [], "MASE_std": [], "WQL_std": []}
    }
    
    # Process each model type (linear and fourier)
    for model_type in ["linear", "fourier"]:
        # Process each dataset size
        for dataset_size in X_POSITIONS:
            run_indices = idxs[model_type][dataset_size]
            
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
            
    return metrics

def build_graph(metric_name, linear_values, fourier_values, linear_std, fourier_std, output_fname):
    fig, ax = plt.subplots(figsize=(6, 3))
    
    x_positions = X_POSITIONS

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
    ax.set_xticks(x_positions)
    x_labels = [r'20k', r'40k', r'60k', r'80k', r'100k', r'120k', r'140k', r'160k', r'180k', r'200k']
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Dataset size", fontsize=16)
    ax.legend(loc="best")
    
    plt.tight_layout()
    plt.savefig(output_fname, dpi=300)
    plt.close()

def main():
    base_path = "output/11-26-pt4-ablation-dataset-size-smaller"
    metrics = gather_metrics(base_path)
    
    # Create MASE graph
    build_graph(
        "MASE",
        metrics["linear"]["MASE"],
        metrics["fourier"]["MASE"],
        metrics["linear"]["MASE_std"],
        metrics["fourier"]["MASE_std"],
        "scripts/eval/graphing/chronos_ablation_varying_dataset_size_v2_MASE.png"
    )
    
    # Create WQL graph
    build_graph(
        "WQL",
        metrics["linear"]["WQL"],
        metrics["fourier"]["WQL"],
        metrics["linear"]["WQL_std"],
        metrics["fourier"]["WQL_std"],
        "scripts/eval/graphing/chronos_ablation_varying_dataset_size_v2_WQL.png"
    )

if __name__ == "__main__":
    main()