from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from gluonts.dataset.arrow import ArrowWriter
import datasets
from tqdm import tqdm
import os
import sys
import random

def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    start_times: Optional[Union[List[np.datetime64], np.ndarray]] = None,
    compression: str = "lz4",
    subset_size: int = 100,
    random_seed: Optional[int] = None
):
    if start_times is None:
        # Set an arbitrary start time
        start_times = [np.datetime64("2000-01-01 00:00", "s")] * len(time_series)

    assert len(time_series) == len(start_times)

    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Create indices for all time series
    all_indices = list(range(len(time_series)))
    
    # Randomly select subset_size indices
    selected_indices = random.sample(all_indices, min(subset_size, len(time_series)))
    
    dataset = []
    for idx in tqdm(selected_indices):
        ts = time_series[idx]
        start = start_times[idx]
        dataset.append({"start": start, "target": np.asarray(ts["target"])})

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )    

if __name__ == "__main__":

    subset_size_tsmixup = int(sys.argv[1])
    subset_size_kernelsynth = int(sys.argv[2])
    random_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42  # Default seed is 42

    cache_dir = "data"
    os.makedirs(cache_dir, exist_ok=True)

    ds_kernelsynth = datasets.load_dataset(
        "autogluon/chronos_datasets", 
        "training_corpus_kernel_synth_1m",
        streaming=False, 
        split="train", 
        cache_dir=cache_dir
    )
    convert_to_arrow(
        f"data/kernelsynth-data-{subset_size_kernelsynth}-seed-{random_seed}.arrow", 
        time_series=ds_kernelsynth, 
        subset_size=subset_size_kernelsynth,
        random_seed=random_seed
    )

    ds_tsmixup = datasets.load_dataset(
        "autogluon/chronos_datasets", 
        "training_corpus_tsmixup_10m", 
        streaming=False, 
        split="train", 
        cache_dir=cache_dir
    )
    convert_to_arrow(
        f"data/tsmixup-data-{subset_size_tsmixup}-seed-{random_seed}.arrow", 
        time_series=ds_tsmixup, 
        subset_size=subset_size_tsmixup,
        random_seed=random_seed
    )