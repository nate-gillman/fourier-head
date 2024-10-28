import json
import os
import numpy as np

import sys
for path in sys.path:
    if path.endswith("/imitation-learning/scripts/eval"):
        sys.path.append(path.replace("/imitation-learning/scripts/eval", "/"))

from smoothness_metric import get_smoothness_metric

def compute_and_save_smoothness_dict_in_dir(dir):

    print(f"Beginning compute_and_save_smoothness_dict_in_dir for dir {dir}...")

    # STEP 1: load all the numpy files in the dir

    npy_fnames = sorted(os.listdir(dir))
    npy_paths = []
    for npy_fname in npy_fnames:
        if npy_fname.endswith(".npy") and not npy_fname.startswith("test"):
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

    MULTINOMIAL_DIRS = [
        "output/linear",
        "output/fourier_2",
        "output/fourier_4",
        "output/fourier_6",
        "output/fourier_8",
        "output/fourier_10",
        "output/fourier_12",
        "output/fourier_14",
        "output/fourier_16",
        "output/fourier_18",
        "output/fourier_20",
        "output/fourier_22",
        "output/fourier_24",
        "output/fourier_26",
        "output/fourier_28",
        "output/fourier_30",
        "output/fourier_32",
    ]

    for dir in MULTINOMIAL_DIRS:
        compute_and_save_smoothness_dict_in_dir(dir)

if __name__ == "__main__":
    # demo_get_smoothness_metrics()
    main()