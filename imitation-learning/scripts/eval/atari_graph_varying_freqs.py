import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import json
import argparse

from atari_expert_scores import get_human_normalized_score

font_path = 'scripts/eval/Times_New_Roman.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

def get_frequencies_for_game(game_name):
    if game_name in ['Seaquest', "Gravitar", "BankHeist", "DoubleDunk"]:
        return list(range(2, 34, 2))  # [2,4,6,8,...,32]
    elif game_name in ["Centipede"]:
        return list(range(4, 36, 4))  # [4,8,12,...,32]
    elif game_name == "Boxing" : 
        return [4, 8, 12, 16]
    else:
        return [8, 16]

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


def gather_returns_and_smoothness(game_name, use_originally_reported_baseline=True):

    DIVIDE_STD_BY = (1/0.67) # this gives 50% confidence interval

    vals = {}

    # STEP 1: GET BASELINE METRICS
    if use_originally_reported_baseline:
        # Linear baseline values for each graph
        return_baseline = 2.53
        return_baseline_err = 0.63/DIVIDE_STD_BY
        smoothness_baseline = 0.48
        smoothness_baseline_err = 0.14/DIVIDE_STD_BY
    else:

        # GET RETURNS STATISTICS
        returns_local_path = f"output/{game_name}/linear/returns_dict.json"
        with open(returns_local_path, "r") as f:
            returns_local = json.load(f)
        return_baseline, return_std = get_returns_metrics(game_name, returns_local)
        return_baseline_err = return_std / DIVIDE_STD_BY

        # GET SMOOTHNESS STATISTICS
        smoothness_local_path = f"output/{game_name}/linear/smoothness_dict.json"
        with open(smoothness_local_path, "r") as f:
            smoothness_local = json.load(f)
        smoothness_baseline = smoothness_local["L2"]["mean"]
        smoothness_std = smoothness_local["L2"]["std"]
        smoothness_baseline_err = smoothness_std / DIVIDE_STD_BY
        # smoothness_fourier.append(smoothness_mean)
        # smoothness_fourier_err.append(smoothness_std/DIVIDE_STD_BY)

    vals["baseline"] = {
        "return"           : return_baseline,
        "return_err"       : return_baseline_err,
        "smoothness"        : smoothness_baseline,
        "smoothness_err"    : smoothness_baseline_err
    }

    # STEP 2: GET FOURIER METRICS
    returns_fourier, returns_fourier_err = [], []
    smoothness_fourier, smoothness_fourier_err = [], []
    freqs = get_frequencies_for_game(game_name)
    for freq in freqs:

        # GET RETURNS STATISTICS
        returns_local_path = f"output/{game_name}/fourier_{freq}/returns_dict.json"
        with open(returns_local_path, "r") as f:
            returns_local = json.load(f)
        returns_mean, returns_std = get_returns_metrics(game_name, returns_local)
        returns_fourier.append(returns_mean)
        returns_fourier_err.append(returns_std)

        # GET SMOOTHNESS STATISTICS
        smoothness_local_path = f"output/{game_name}/fourier_{freq}/smoothness_dict.json"
        with open(smoothness_local_path, "r") as f:
            smoothness_local = json.load(f)
        smoothness_mean = smoothness_local["L2"]["mean"]
        smoothness_std = smoothness_local["L2"]["std"]
        smoothness_fourier.append(smoothness_mean)
        smoothness_fourier_err.append(smoothness_std/DIVIDE_STD_BY)

    
    vals["fourier"] = {
        "returns"           : returns_fourier,
        "returns_err"       : returns_fourier_err,
        "smoothness"        : smoothness_fourier,
        "smoothness_err"    : smoothness_fourier_err
    }

    return vals, freqs


def build_graphs(vals, freqs, output_fname, title=""):

    return_baseline         = vals["baseline"]["return"]
    return_baseline_err     = vals["baseline"]["return_err"]
    smoothness_baseline     = vals["baseline"]["smoothness"]
    smoothness_baseline_err = vals["baseline"]["smoothness_err"]

    returns_fourier         = vals["fourier"]["returns"]
    returns_fourier_err     = vals["fourier"]["returns_err"]
    smoothness_fourier      = vals["fourier"]["smoothness"]
    smoothness_fourier_err  = vals["fourier"]["smoothness_err"]


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))  # Two side-by-side graphs

    # Global title for both graphs
    fig.suptitle(title, fontsize=24, y=0.95)

    # Plot the first graph (Returns)
    ax1.plot(freqs, returns_fourier, c="tab:blue", label="Fourier head")
    ax1.fill_between(freqs, 
                     np.array(returns_fourier) - np.array(returns_fourier_err), 
                     np.array(returns_fourier) + np.array(returns_fourier_err), 
                     color="tab:blue", alpha=0.2)  # Standard deviation tunnel for returns

    # Add red horizontal line for the baseline and tunnel for the baseline
    ax1.axhline(y=return_baseline, color='red', linestyle='--', label="Linear head")
    ax1.fill_between(freqs, 
                     return_baseline - return_baseline_err, 
                     return_baseline + return_baseline_err, 
                     color='red', alpha=0.2)  # Standard deviation tunnel for baseline

    ax1.set_xlabel("Fourier Frequencies", fontsize=16)
    ax1.set_ylabel("Normalized Returns", fontsize=16)
    ax1.grid(True, linewidth=0.3)
    ax1.set_xlim(freqs[0], freqs[-1])  # Set the x-axis limits to match the new tick values
    ax1.xaxis.set_major_locator(MultipleLocator(2))  # Set x-ticks to 2, 4, 6, ..., 32
    ax1.legend(loc="lower right")

    # Plot the second graph (Smoothness)
    ax2.plot(freqs, smoothness_fourier, c="tab:blue", label="Fourier head")
    ax2.fill_between(freqs, 
                     np.array(smoothness_fourier) - np.array(smoothness_fourier_err), 
                     np.array(smoothness_fourier) + np.array(smoothness_fourier_err), 
                     color="tab:blue", alpha=0.2)  # Standard deviation tunnel for smoothness

    # Add red horizontal line for the baseline and tunnel for the baseline
    ax2.axhline(y=smoothness_baseline, color='red', linestyle='--', label="Linear head")
    ax2.fill_between(freqs, 
                     smoothness_baseline - smoothness_baseline_err, 
                     smoothness_baseline + smoothness_baseline_err, 
                     color='red', alpha=0.2)  # Standard deviation tunnel for baseline

    ax2.set_xlabel("Fourier Frequencies", fontsize=16)
    ax2.set_ylabel("Smoothness", fontsize=16)
    ax2.grid(True, linewidth=0.3)
    ax2.set_xlim(freqs[0], freqs[-1])  # Set the x-axis limits to match the new tick values
    ax2.xaxis.set_major_locator(MultipleLocator(2))  # Set x-ticks to 2, 4, 6, ..., 32
    ax2.legend(loc="lower right")

    # Adjust layout to keep spacing consistent
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(output_fname, dpi=300)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'game', 
        type=str, 
        help='Game name (e.g. DoubleDunk or Seaquest)'
    )
    args = parser.parse_args()

    if args.game == "Seaquest":
        use_originally_reported_baseline = True
    else:
        use_originally_reported_baseline = False

    vals, freqs = gather_returns_and_smoothness(
        args.game,
        use_originally_reported_baseline=use_originally_reported_baseline
    )

    build_graphs(
        vals,
        freqs, 
        output_fname=f"scripts/eval/atari_graph_varying_freqs_{args.game}.png",
        title=f"Impact of Varying Fourier Frequencies on Decision Transformer: {args.game}"
    )

if __name__ == "__main__":
    main()

