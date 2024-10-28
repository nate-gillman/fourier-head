from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from gluonts.dataset.arrow import ArrowWriter
import datasets
from tqdm import tqdm
import os

def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    start_times: Optional[Union[List[np.datetime64], np.ndarray]] = None,
    compression: str = "lz4",
):
    if start_times is None:
        # Set an arbitrary start time
        start_times = [np.datetime64("2000-01-01 00:00", "s")] * len(time_series)

    assert len(time_series) == len(start_times)

    dataset = []
    for ts, start in tqdm(zip(time_series, start_times)):
        dataset.append({"start": start, "target": np.asarray(ts["target"])})

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


def get_mini_chunk(dataset, start_idx, num_ts):

    time_series = []
    for idx in tqdm(range(start_idx, min(len(dataset), start_idx+num_ts))):
        time_series.append(dataset[idx])

    return time_series

    

if __name__ == "__main__":

    cache_dir = "data"
    os.makedirs(cache_dir, exist_ok=True)

    ds_kernelsynth = datasets.load_dataset("autogluon/chronos_datasets", "training_corpus_kernel_synth_1m", streaming=False, split="train", cache_dir=cache_dir)
    convert_to_arrow("data/kernelsynth-data.arrow", time_series=ds_kernelsynth)

    ds_tsmixup = datasets.load_dataset("autogluon/chronos_datasets", "training_corpus_tsmixup_10m", streaming=False, split="train", cache_dir=cache_dir)
    convert_to_arrow("data/tsmixup-data.arrow", time_series=ds_tsmixup)
    
