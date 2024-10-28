import json
import os

from typing import Callable
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers.trainer_utils import denumpify_detensorize

import sys
for path in sys.path:
    if path.endswith("/time-series-forecasting/scripts/eval"):
        sys.path.append(path.replace("/time-series-forecasting/scripts/eval", ""))

from smoothness_metric import get_smoothness_metric


def compute_and_save_smoothness_dict_in_dir(dir):

    print(f"Beginning compute_and_save_smoothness_dict_in_dir for dir {dir}...")

    # STEP 1: compute smoothness for each dataset

    multinomials_dir = os.path.join(dir, "multinomials")
    npy_fnames = sorted(os.listdir(multinomials_dir))
    npy_paths = [os.path.join(multinomials_dir, npy_fname) for npy_fname in npy_fnames]

    smoothness_dict = {}
    dataset_name_to_npy_path = {}
    for npy_path in npy_paths:

        print(f"   ...beginning {npy_path}")

        assert npy_path.endswith(".npy")
        dataset_name = npy_path.split("/")[-1].split(".")[0] # e.g. 'ETTh'
        dataset_name_to_npy_path[dataset_name] = npy_path

        multinomials = np.load(npy_path) # (num_timesteps_total, 4096)
        smoothness_metrics = get_smoothness_metric(multinomials)
        smoothness_metrics["num_multinomials"] = multinomials.shape[0]

        smoothness_dict[dataset_name] = smoothness_metrics

    # STEP 2: compute aggregated smoothness, and aggregated standard deviations
    # NOTE: we're assuming equal weighting between each of them!!

    L2_means, L2_stds, nums_multinomials = [], [], []
    for dataset_name in smoothness_dict.keys():
        smoothness_dict_local = smoothness_dict[dataset_name]

        L2_means.append(smoothness_dict_local["L2"]["mean"])
        L2_stds.append(smoothness_dict_local["L2"]["std"])
        nums_multinomials.append(smoothness_dict_local["num_multinomials"])

    L2_means = np.asarray(L2_means)
    L2_stds = np.asarray(L2_stds)
    nums_multinomials = np.asarray(nums_multinomials)

    aggregate_metrics = {
        "mean" : L2_means.mean(),
        "std" : np.linalg.norm(L2_stds) / np.sqrt(L2_stds.shape[0]),
        "num_multinomials" : nums_multinomials.sum()
    }

    # STEP 3: write to disk
    smoothness_metrics_dict_path = os.path.join(dir, "eval_dict-smoothness.json")

    smoothness_dict["aggregate_metrics"] = aggregate_metrics
    metrics_dict = denumpify_detensorize(smoothness_dict)

    with open(smoothness_metrics_dict_path, "w") as fp:
        json.dump( metrics_dict, fp, indent=4)

    print(f"compute_and_save_smoothness_dict_in_dir finished for dir = {smoothness_metrics_dict_path}")

    return None

def main():

    model_dir = "/".join(sys.argv[1].split("/")[:-1])
    compute_and_save_smoothness_dict_in_dir(model_dir)

if __name__ == "__main__":
    main()