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
    if path.endswith("/toy-example-synthetic/eval"):
        sys.path.append(path.replace("/toy-example-synthetic/eval", "/"))


for path in sys.path:
    if path.endswith("/toy-example-synthetic"):
        sys.path.append(path.replace("/toy-example-synthetic", "/"))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from smoothness_metric import get_smoothness_metric


def compute_and_save_smoothness_dict_in_dir(dir):

    print(f"Beginning compute_and_save_smoothness_dict_in_dir for dir {dir}...")

    # STEP 1: load all the numpy files in the dir

    npy_fnames = sorted(os.listdir(dir))
    npy_paths = []
    for npy_fname in npy_fnames:
        if npy_fname.endswith(".npy"):
            npy_paths.append(os.path.join(dir, npy_fname))

    npy_arrs = []
    for npy_path in npy_paths:
        npy_arr = np.load(npy_path, allow_pickle=True)
        npy_arrs.append(npy_arr)

    multinomials = np.concatenate(npy_arrs, axis=0) # (num_multinomials, 18)
    
    # STEP 2: compute smoothness metric for all of them
    smoothness_metrics = get_smoothness_metric(multinomials)
    smoothness_metrics["num_multinomials"] = float(multinomials.shape[0])

    # STEP 3: write metric dict to disk
    smoothness_metrics_dict_path = os.path.join(dir, "smoothness_dict.json")
    with open(smoothness_metrics_dict_path, "w") as fp:
        json.dump(smoothness_metrics, fp, indent=4)

    print(f"compute_smoothness_dict finished for dir = {smoothness_metrics_dict_path}")

    return None

def main():
    dataset_list = ['gaussian', 'gmm', 'gmm2']
    freqs_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    gamma_list = ['0.0', '1e-06']

    model_dirs = [
    f"graphing/sweep_output/{dataset}/fourier/{gamma}/{freq}" 
    for dataset in dataset_list
    for gamma in gamma_list 
    for freq in freqs_list
]
    model_dirs.append("graphing/sweep_output/gaussian/linear/0.0/0/")
    model_dirs.append("graphing/sweep_output/gmm/linear/0.0/0/")
    model_dirs.append("graphing/sweep_output/gmm2/linear/0.0/0/")

    for dir in model_dirs:
        compute_and_save_smoothness_dict_in_dir(dir)

if __name__ == "__main__":
    main()