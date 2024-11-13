from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from gluonts.dataset.arrow import ArrowWriter
import datasets
from tqdm import tqdm
import os
import sys

def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    start_times: Optional[Union[List[np.datetime64], np.ndarray]] = None,
    compression: str = "lz4",
    subset_size=100
):
    if start_times is None:
        # Set an arbitrary start time
        start_times = [np.datetime64("2000-01-01 00:00", "s")] * len(time_series)

    assert len(time_series) == len(start_times)

    dataset = []
    num_found = 0
    for ts, start in tqdm(zip(time_series, start_times)):
        num_found += 1
        dataset.append({"start": start, "target": np.asarray(ts["target"])})
        if num_found >= subset_size:
            break

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )    

if __name__ == "__main__":

    subset_size_tsmixup = int(sys.argv[1])
    subset_size_kernelsynth = int(sys.argv[2])

    cache_dir = "data"
    os.makedirs(cache_dir, exist_ok=True)

    ds_kernelsynth = datasets.load_dataset(
        "autogluon/chronos_datasets", 
        "training_corpus_kernel_synth_1m",
        streaming=False, 
        split="train", 
        cache_dir=cache_dir
    )
    convert_to_arrow(f"data/kernelsynth-data-{subset_size_kernelsynth}.arrow", time_series=ds_kernelsynth, subset_size=subset_size_kernelsynth)

    ds_tsmixup = datasets.load_dataset(
        "autogluon/chronos_datasets", 
        "training_corpus_tsmixup_10m", 
        streaming=False, 
        split="train", 
        cache_dir=cache_dir
    )
    convert_to_arrow(f"data/tsmixup-data-{subset_size_tsmixup}.arrow", time_series=ds_tsmixup, subset_size=subset_size_tsmixup)
    
