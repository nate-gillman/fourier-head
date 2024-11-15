import json
import os
import numpy as np
import argparse

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

    parser = argparse.ArgumentParser(description='Compute smoothness metrics for multinomial data')
    parser.add_argument(
        'game', 
        type=str, 
        help='Game name (e.g., DoubleDunk or Seaquest)'
    )
    args = parser.parse_args()

    MULTINOMIAL_DIRS = [
        f"output/{args.game}/linear",
        f"output/{args.game}/fourier_2",
        f"output/{args.game}/fourier_4",
        f"output/{args.game}/fourier_6",
        f"output/{args.game}/fourier_8",
        f"output/{args.game}/fourier_10",
        f"output/{args.game}/fourier_12",
        f"output/{args.game}/fourier_14",
        f"output/{args.game}/fourier_16",
        f"output/{args.game}/fourier_18",
        f"output/{args.game}/fourier_20",
        f"output/{args.game}/fourier_22",
        f"output/{args.game}/fourier_24",
        f"output/{args.game}/fourier_26",
        f"output/{args.game}/fourier_28",
        f"output/{args.game}/fourier_30",
        f"output/{args.game}/fourier_32",
    ]

    for dir in MULTINOMIAL_DIRS:
        try:
            compute_and_save_smoothness_dict_in_dir(dir)
            print(f"...compute_and_save_smoothness_dict_in_dir finished for dir: {dir}")
        except:
            print(f"...Skipped dir: {dir}")
            pass

if __name__ == "__main__":
    main()