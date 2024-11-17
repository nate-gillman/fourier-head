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

idxs_v1 = {
    "linear" : {
        1000 : [1, 3, 5, 7],
        10000 : [9, 11, 13, 15],
        100000 : [17, 19, 21, 23],
        1000000 : [25, 27, 29, 31],
        10000000 : [33]
    },
    "fourier" : {
        1000 : [0, 2, 4, 6],
        10000 : [8, 10, 12, 14],
        100000 : [16, 18, 20, 22],
        1000000 : [24, 26, 28, 30],
        10000000 : [32]
    },
}

idxs_v2 = {
    "linear" : {
        1000 : [1, 3, 5, 7],
        10000 : [9, 11, 13, 15],
        100000 : [35, 37, 39, 41],
        1000000 : [25, 27, 29, 31],
        10000000 : [33]
    },
    "fourier" : {
        1000 : [0, 2, 4, 6],
        10000 : [8, 10, 12, 14],
        100000 : [34, 36, 38, 40],
        1000000 : [24, 26, 28, 30],
        10000000 : [32]
    },
}

idxs_v3 = {
    "linear" : {
        1000 : [1, 3, 5, 7],
        10000 : [9, 11, 13, 15],
        100000 : [35, 37, 39, 41],
        1000000 : [43, 45, 47, 49],
        10000000 : [33]
    },
    "fourier" : {
        1000 : [0, 2, 4, 6],
        10000 : [8, 10, 12, 14],
        100000 : [34, 36, 38, 40],
        1000000 : [42, 44, 46, 48],
        10000000 : [32]
    },
}


idxs_v4 = {
    "linear" : {
        10000 : [9, 11, 13, 15],
        100000 : [35, 37, 39, 41],
        1000000 : [43, 45, 47, 49],
        10000000 : [33]
    },
    "fourier" : {
        10000 : [8, 10, 12, 14],
        100000 : [34, 36, 38, 40],
        1000000 : [42, 44, 46, 48],
        10000000 : [32]
    },
}

idxs_v5 = {
    "linear" : {
        1000 : [59, 61, 63, 65],
        10000 : [51, 53, 55, 57],
        100000 : [35, 37, 39, 41],
        1000000 : [43, 45, 47, 49],
        10000000 : [33]
    },
    "fourier" : {
        1000 : [58, 60, 62, 64],
        10000 : [50, 52, 54, 56],
        100000 : [34, 36, 38, 40],
        1000000 : [42, 44, 46, 48],
        10000000 : [32]
    },
}

idxs_v6 = {
    "linear" : {
        10000 : [51, 53, 55, 57],
        100000 : [35, 37, 39, 41],
        1000000 : [43, 45, 47, 49],
        10000000 : [33]
    },
    "fourier" : {
        10000 : [50, 52, 54, 56],
        100000 : [34, 36, 38, 40],
        1000000 : [42, 44, 46, 48],
        10000000 : [32]
    },
}

idxs = {
    "linear" : {
        10000 : [51, 53, 55, 57],
        100000 : [35, 37, 39, 41],
        1000000 : [25, 27, 29, 31],
        10000000 : [33]
    },
    "fourier" : {
        10000 : [50, 52, 54, 56],
        100000 : [34, 36, 38, 40],
        1000000 : [24, 26, 28, 30],
        10000000 : [32]
    },
} # winner...

def get_largest_checkpoint_idx(lst):
    lst_ints = [int(elt) for elt in lst]
    ckpt = str(max(lst_ints))
    return ckpt

def gather_metrics():
    metrics = {
        "linear": {"MASE": [], "WQL": [], "MASE_std": [], "WQL_std": []},
        "fourier": {"MASE": [], "WQL": [], "MASE_std": [], "WQL_std": []}
    }
    
    # Process each model type (linear and fourier)
    for model_type in ["linear", "fourier"]:
        # Process each dataset size
        for dataset_size in [10000, 100000, 1000000, 10000000]:
            run_indices = idxs[model_type][dataset_size]
            
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

def build_graph(metric_name, linear_values, fourier_values, linear_std, fourier_std, output_fname):
    fig, ax = plt.subplots(figsize=(6, 3))
    
    x_positions = list(range(1, 5))

    DIVIDE_STD_BY = (2/0.67)  # this gives 25% confidence interval
    
    # note: last std is zero
    linear_std[-1] = 0  # Zero out last std value
    fourier_std[-1] = 0  # Zero out last std value

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
    x_labels = [r'1.1$\times$10$^4$', r'1.1$\times$10$^5$', r'1.1$\times$10$^6$', r'1.1$\times$10$^7$']
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Dataset size", fontsize=16)
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
        metrics["linear"]["MASE_std"],
        metrics["fourier"]["MASE_std"],
        "scripts/eval/graphing/chronos_ablation_varying_dataset_size_MASE.png"
    )
    
    # Create WQL graph
    build_graph(
        "WQL",
        metrics["linear"]["WQL"],
        metrics["fourier"]["WQL"],
        metrics["linear"]["WQL_std"],
        metrics["fourier"]["WQL_std"],
        "scripts/eval/graphing/chronos_ablation_varying_dataset_size_WQL.png"
    )

if __name__ == "__main__":
    main()