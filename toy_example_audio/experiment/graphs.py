import re
import json
from typing import List, Tuple, Dict, Any

from pathlib import Path
from itertools import pairwise
from datetime import datetime, timezone

import matplotlib.pyplot as plt
font_path = '../imitation-learning/scripts/eval/Times_New_Roman.ttf'
import matplotlib.font_manager as fm
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

import numpy as np

from experiment.load_data import get_bins
from experiment.training import softmax

import sys
sys.path.append("..")
from smoothness_metric import get_smoothness_metric



from config import OUTPUT_FOLDER, GRAPH_OUTPUT_FOLDER

def plot_histogram_with_line(data: np.ndarray, data_true: np.ndarray, ax: plt.Axes, title: str, label: str, bins: Dict[int, List[int]]) -> None:
    # get_smoothness = get_l2_smoothness_measurement_function()
    bin_names = [f"{int(np.median(np.array(x)))}" for x in bins.values()]
    for i in range(len(bin_names)):
        if i % 2 == 1:
            bin_names[i] = ""
    bin_edges = np.linspace(0, 11, data.shape[0] + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot histogram
    color = "tab:red"
    if label == "Fourier":
        color = "tab:blue"

    smoothness = get_smoothness_metric(data[np.newaxis, ...])["L2"]["mean"]
    ax.bar(bin_centers, data, width=0.3333, color=color, alpha=0.7, label=f"Predicted {label} PMF\nSmoothness = {smoothness:.4f}")

    # ---- WHITESPACE CONTROLS -----
    # Fix y-axis limits
    ax.set_ylim(0.0, 0.15)
    y = data[data_true] + 0.02 # adjust constant here to change how much higher green dot is
    # ------------------------------

    ax.set_yticklabels([])
    ax.scatter(x=bin_centers[data_true], y=y, color="tab:green", label="True Label")
    ax.xaxis.set_ticks(bin_centers)
    ax.tick_params(axis='x', labelrotation=0)
    ax.set_xticklabels(bin_names, fontsize=7)
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(title, fontsize=14)

def load_eval_pair(label_ids_path: Path, logits_path: Path) -> Dict[str, np.ndarray]:
    y_true = np.load(label_ids_path)
    y_pred_dist = softmax(np.load(logits_path))
    return {
        'y_true': y_true,
        'y_pred_dist': y_pred_dist
    }

def get_frequency_to_result_map(output_folder: Path = OUTPUT_FOLDER) -> Dict[int, Dict[str, Any]]:
    run_directories = [x for x in output_folder.iterdir() if (x.is_dir() and x.name.isdigit())]
    frequency_to_results_map = {}
    exp = re.compile(r"^run-[0-9]+")
    for run_directory in run_directories:
        # the number of frequencies is the top level directory name
        num_frequencies = int(run_directory.name)
        
        # look for subdirectories of the form `run-x`, where x is a value that increments with each new run
        # take only the most recently ran one, ranked by the time it was created
        latest_run_path = max([x for x in run_directory.iterdir() if x.is_dir() and exp.match(x.name)], key=lambda y: y.stat().st_ctime)

        latest_run_logs = list((latest_run_path / "ast_classifier").rglob("**/trainer_state.json"))
        log_history = []
        for log_file in latest_run_logs:
            trainer_state = json.loads(log_file.read_text())
            log_history += trainer_state['log_history']

        log_history = list(sorted(log_history, key=lambda x: x['step']))
        latest_eval = [x for x in log_history if "eval_f1" in x][-1]
        latest_run_info = json.loads((latest_run_path / "info.json").read_text())
        train_function_args = json.loads(latest_run_info['start_training_function_arguments_json_string'])
        
        # get all the numpy files saved within that run as a list of PosixPaths, order them by their prefix
        # the prefix is the epoch at which they were produced. Example filename: 
        # 0-eval_pred_label_ids-num_freq-1.npy, 0-eval_pred_predictions-num_freq-1.npy (both these were saved at end of 1st epoch)
        # put into tuples of (gt label_ids, predictions)
        saved_evals = list(pairwise(sorted(latest_run_path.glob("*.npy"), key=lambda x: int(x.name.split("-")[0]))))
        frequency_to_results_map[num_frequencies] = {
            'evals': saved_evals,
            'log_history': log_history,
            'latest_eval_metrics': latest_eval,
            'info': latest_run_info,
            # extract some useful information from the run info to top level of this object
            'lm_head_trainable_params': latest_run_info['lm_head_trainable_params'],
            'audio_sample_rate': latest_run_info['sampling_rate'],
            'bins': {int(label): bpm_bucket_vals for label, bpm_bucket_vals in latest_run_info['bins'].items()},
            'dataset_split_style': train_function_args['dataset_split_style'],
        }

    return frequency_to_results_map

def pct_change(new_val: float, old_val: float) -> float:
    return (new_val / old_val) - 1

def create_graph() -> None: 
    # --- LOAD TRAINING ARTIFACTS ---
    frequency_to_results_map = get_frequency_to_result_map()

    linear_eval = frequency_to_results_map[0]['latest_eval_metrics']
    fourier_eval_1 = frequency_to_results_map[1]['latest_eval_metrics']
    fourier_eval_2 = frequency_to_results_map[2]['latest_eval_metrics']

    improvement_over_baseline = pct_change(fourier_eval_1['eval_f1'], linear_eval['eval_f1'])
    assert improvement_over_baseline >= 1.18

    if not frequency_to_results_map:
        raise RuntimeError("No training artifacts exist. Run `toy_example_audio/scripts/reproduce.py` to create them.")

    linear_info = frequency_to_results_map[0]
    fourier_info = frequency_to_results_map[2]
    linear_results = frequency_to_results_map[0]['evals'][-1]
    fourier_results = frequency_to_results_map[2]['evals'][-1]

    fourier_sm = softmax(np.load(fourier_results[1]))
    linear_sm = softmax(np.load(linear_results[1]))
    true_fourier = np.load(fourier_results[0])
    true_linear = np.load(linear_results[0])

    # check some properties of the run that must be true for a more apples-to-apples comparison
    np.testing.assert_equal(true_fourier, true_linear)
    assert fourier_info['bins'] == linear_info['bins']
    assert fourier_info['audio_sample_rate'] == linear_info['audio_sample_rate']
    assert fourier_info['dataset_split_style'] == linear_info['dataset_split_style']
    
    # this part may not be true in general, but in our experiment (this setup's default config) it is
    assert fourier_info['lm_head_trainable_params'] < linear_info['lm_head_trainable_params']

    # --- CREATE THE GRAPH IMAGE ---
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, dpi=300)
    fig.set_size_inches(8, 2)

    # select the sample ID
    sample_id = 499
    
    # set axis titles
    axes[0].set_ylabel("Mass")

    axes[1].set_xlabel("BPM")
    axes[0].set_xlabel("BPM")

    # plot histograms
    bins = fourier_info['bins']
    plot_histogram_with_line(
        fourier_sm[sample_id], true_fourier[sample_id], axes[1], "", "Fourier", bins
    )
    plot_histogram_with_line(
        linear_sm[sample_id], true_linear[sample_id], axes[0], "", "Linear", bins
    )
    plt.tight_layout()

    # save the graph to disk
    GRAPH_OUTPUT_FOLDER.mkdir(exist_ok=True)
    # timestamp each output so we don't lose any graphs accidentally
    # from: https://stackoverflow.com/a/48779287
    tz = timezone.utc
    ft = "%Y-%m-%dT%H:%M:%S%z"
    t = datetime.now(tz=tz).strftime(ft).replace("+", "_").replace(":", "_")
    plt.savefig(GRAPH_OUTPUT_FOLDER / f"fig1_{t}.png", dpi=300, bbox_inches='tight', pad_inches=0.02)

if __name__ == "__main__":
    create_graph()