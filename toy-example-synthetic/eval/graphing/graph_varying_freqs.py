import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import sys
import argparse
import os

for path in sys.path:
    if path.endswith("/eval/graphing"):
        sys.path.append(path.replace("/eval/graphing", "/"))

from eval.aggregate import aggregate
        
font_path = 'eval/graphing/Times_New_Roman.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

DIVIDE_STD_BY = (1/0.67)  # this gives 50% confidence interval

FONTSIZE = 19

def build_graphs(data, baseline_values, baseline_stds, gmm_values, gmm_stds, output_fname, title=""):
    fig, axs = plt.subplots(1, 2, figsize=(20, 4))  # Three vertical graphs
    # Global title for all graphs
    fig.suptitle(title, fontsize=26, y=0.99)
    # X-axis values (2, 4, 6, ..., 32)
    generations = list(range(2, 22, 2))
    for i, (metric, values) in enumerate(data.items()):
        if metric == 'MSE': 
            continue
        if metric == 'Smoothness':
            i = i-1

        for gamma, (gamma_values, gamma_std) in values.items():
            if gamma == 'gamma0':
                axs[i].plot(generations, gamma_values, c="tab:orange", label='Fourier, no regularization')
                axs[i].fill_between(generations,
                                    np.array(gamma_values) - np.array(gamma_std),
                                    np.array(gamma_values) + np.array(gamma_std),
                                    color='tab:orange', alpha=0.2)
            if gamma == 'gamma1':
                axs[i].axhline(y=baseline_values[metric], color='tab:red', linestyle='--', label="Linear baseline")
                axs[i].fill_between(generations,
                                baseline_values[metric] - baseline_stds[metric],
                                baseline_values[metric] + baseline_stds[metric],
                                color='tab:red', alpha=0.2)  # Red tunnel for the baseline

                axs[i].axhline(y=gmm_values[metric], color='tab:purple', linestyle='--', label="GMM head")  
                axs[i].fill_between(generations,
                                gmm_values[metric] - gmm_stds[metric],
                                gmm_values[metric] + gmm_stds[metric],
                                color='tab:purple', alpha=0.2)  # Red tunnel for the baseline
                axs[i].plot(generations, gamma_values, c="tab:blue", label='Fourier, with regularization')
                axs[i].fill_between(generations,
                                    np.array(gamma_values) - np.array(gamma_std),
                                    np.array(gamma_values) + np.array(gamma_std),
                                    color='tab:blue', alpha=0.2)
            


        axs[i].set_xlabel("Fourier Frequencies", fontsize=FONTSIZE)
        axs[i].set_ylabel(metric, fontsize=FONTSIZE)
        axs[i].grid(True, linewidth=0.3)
        axs[i].set_xlim(2, 20)
        axs[i].xaxis.set_major_locator(MultipleLocator(2))
        #axs[i].legend(loc="upper right",fontsize=FONTSIZE)
        # Retrieve current handles and labels
        if i == 0:
            handles, labels = axs[i].get_legend_handles_labels()

            # Define a new order for the labels (e.g., [Line 3, Line 1, Line 2])
            order = [1, 2, 0, 3]

            # Reorder handles and labels
            reordered_handles = [handles[i] for i in order]
            reordered_labels = [labels[i] for i in order]

            # Pass reordered handles and labels to legend
            axs[i].legend(reordered_handles, reordered_labels, loc="upper right",fontsize=FONTSIZE)
            
        axs[i].tick_params(axis='x', labelsize=FONTSIZE)
        axs[i].tick_params(axis='y', labelsize=FONTSIZE)

    # Adjust layout to keep spacing consistent
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_fname)
    plt.show()


def main(output_dir):
    # Set dataset
    title_dict = {"gaussian": "Gaussian", "gmm": "GMM", "gmm2": "GMM-2", "beta": "Beta"}
    for dataset in ["gaussian", "gmm2", "beta"]:
        data, baseline_values, baseline_stds, gmm_values, gmm_stds = aggregate(output_dir, [dataset], verbose=False)[0]
        # Adjust standard deviations
        baseline_stds = {key: value / DIVIDE_STD_BY for key, value in baseline_stds.items()}
        gmm_stds = {key: value / DIVIDE_STD_BY for key, value in gmm_stds.items()}
        for key in data:
            for gamma in data[key]:
                data[key][gamma][1] =[std / DIVIDE_STD_BY for std in data[key][gamma][1]]
       
        build_graphs(
            data,
            baseline_values,
            baseline_stds,
            gmm_values,
            gmm_stds,
            output_fname=os.path.join(output_dir, f"toy_{dataset}_graph_varying_freqs.png"),
            title=f"Impact of Varying Fourier Frequencies on {title_dict[dataset]} dataset"
        )

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse input arguments.")
        
    # Adding arguments
    parser.add_argument('--dir', type=str, required=True, help='Specify output dir (string)')
    # Parsing arguments
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.dir)
    print(f'Graphs saved to dir: {args.dir}')
    
