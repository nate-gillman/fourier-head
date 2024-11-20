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

from atari_expert_scores import get_human_normalized_score

def get_returns_metrics(game_name, returns_local):

    returns_best = []
    for returns in returns_local.values():
        returns_best.append(max(returns))

    returns_best = np.asarray(returns_best)

    # normalize
    returns_best = 100*get_human_normalized_score(game_name.lower(), returns_best)

    returns_mean = returns_best.mean()
    returns_std = returns_best.std()

    return returns_mean, returns_std


def gather_returns_and_smoothness(game_name, sizes):

    DIVIDE_STD_BY = (1/0.67) # this gives 50% confidence interval

    vals = {
        "linear_returns"        : [],
        "linear_returns_err"    : [],
        "fourier_returns"       : [],
        "fourier_returns_err"   : []
    }

    for size in sizes:

        # get linear returns
        returns_local_path = f"output/{game_name}-dataset-{size}/linear/returns_dict.json"
        with open(returns_local_path, "r") as f:
            returns_local = json.load(f)
        
        return_baseline, return_std = get_returns_metrics(game_name, returns_local)
        return_baseline_err = return_std / DIVIDE_STD_BY

        vals["linear_returns"].append(return_baseline)
        vals["linear_returns_err"].append(return_baseline_err)

        # get fourier returns
        returns_fourier, returns_fourier_err = [], []
        smoothness_fourier, smoothness_fourier_err = [], []
        freq = 14

        # GET RETURNS STATISTICS
        returns_local_path = f"output/{game_name}-dataset-{size}/fourier_{freq}/returns_dict.json"
        with open(returns_local_path, "r") as f:
            returns_local = json.load(f)

        return_mean, return_std = get_returns_metrics(game_name, returns_local)
        return_std = return_std / DIVIDE_STD_BY
        vals["fourier_returns"].append(return_mean)
        vals["fourier_returns_err"].append(return_std)


    return vals


def build_graph(metric_name, linear_values, fourier_values, linear_err, fourier_err, output_fname):
    fig, ax = plt.subplots(figsize=(6, 3))
    
    x_positions = list(range(1, 6))

    # Plot fourier metric
    ax.plot(x_positions, fourier_values, c="tab:blue", label="Fourier head")
    # Plot standard deviation tunnel for linear metric
    ax.fill_between(
        x_positions, 
        np.array(fourier_values) - np.array(fourier_err), 
        np.array(fourier_values) + np.array(fourier_err), 
        color="tab:blue", 
        alpha=0.2
    )
    
    # Plot linear metric
    ax.plot(x_positions, linear_values, c="tab:red", label="Linear head")
    # Plot standard deviation tunnel for linear metric
    ax.fill_between(
        x_positions, 
        np.array(linear_values) - np.array(linear_err), 
        np.array(linear_values) + np.array(linear_err), 
        color="tab:red", 
        alpha=0.2
    )

    ax.set_ylabel(metric_name, fontsize=16)
    ax.grid(True, linewidth=0.3)
    
    # Set custom x-axis ticks and labels
    ax.set_xticks(x_positions)
    x_labels = [r'5$\times$10$^2$', r'5$\times$10$^3$', r'5$\times$10$^4$', r'5$\times$10$^5$', r'5$\times$10$^6$']
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Dataset size", fontsize=16)
    ax.legend(loc="best")
    
    plt.tight_layout()
    plt.savefig(output_fname, dpi=300)
    plt.close()

def main():
    sizes = [50, 500, 5000, 50000, 500000]
    vals = gather_returns_and_smoothness("Seaquest", sizes)
    
    # Create normalized returns graph
    build_graph(
        "Normalized returns",
        vals["linear_returns"],
        vals["fourier_returns"],
        vals["linear_returns_err"],
        vals["fourier_returns_err"],
        "scripts/eval/atari_graph_varying_dataset_size.png"
    )

if __name__ == "__main__":
    main()