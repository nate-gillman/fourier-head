import re
from typing import Dict, List, Any, Tuple, Literal
from filelock import FileLock
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from datasets import Dataset, Audio, ClassLabel, Features, DatasetDict, load_dataset


def get_dataset_from_hugging_face_hub() -> Dataset:
    # there are 6 parts in parquet files: https://huggingface.co/datasets/meganwei/syntheory/tree/main/tempos
    # approx 1.2 GB at rest
    ds = load_dataset("meganwei/syntheory", "tempos", streaming=False, cache_dir="data")['train']
    return ds.select_columns(['audio', 'bpm'])


def get_bins(start: int, end: int, bin_size: int) -> Dict[int, List[int]]:
    bins = defaultdict(list)
    for i in range(start, end + 1):
        bin_index = (i - start) // bin_size
        bins[bin_index].append(i)
    return bins


def get_bin_for_value(value: int, start: int, bin_size: int) -> int:
    bin_index = (value - start) // bin_size
    return bin_index


def get_dataset(
    bin_size: int = 15,
    test_size_percent: float = 0.3,
    split_style: Literal["standard", "odds_evens"] = "standard",
    random_seed: int = 42,
) -> Tuple[Dataset, Dict[str, Any]]:
    ds = get_dataset_from_hugging_face_hub()

    all_bpms = ds["bpm"]
    lb = min(all_bpms)
    ub = max(all_bpms)
    bins = get_bins(lb, ub, bin_size)

    # a bit akward but the class label's name is an integer as a string
    class_labels = [f"[{x[0]}, {x[-1]}]" for x in bins.values()]

    # the ID is just auto assigned sequential
    class_labels_to_id = {x: i for i, x in enumerate(class_labels)}
    id_to_class_labels = {i: x for i, x in enumerate(class_labels)}

    # order matters here
    labels = [get_bin_for_value(x, lb, bin_size) for x in all_bpms]
    dataset = ds.add_column(name="labels", column=labels)

    # do a typical test train split with shuffling
    if split_style == "standard":
        dataset_splits = dataset.shuffle(seed=random_seed).train_test_split(
            test_size=test_size_percent
        )
    elif split_style == "odds_evens":
        # train on odds, test on evens
        idx_odd = [i for i, x in enumerate(all_bpms) if x % 2 == 1]
        idx_even = [i for i, x in enumerate(all_bpms) if x % 2 != 1]
        train_set = dataset.select(idx_odd).shuffle(seed=random_seed)
        test_set = dataset.select(idx_even).shuffle(seed=random_seed)
        dataset_splits = DatasetDict({"train": train_set, "test": test_set})
    else:
        raise ValueError(f"Invalid split style, got: {split_style}")

    # return dataset object and some summary information we need later
    return dataset_splits, {
        "num_labels": len(class_labels),
        "label2id": class_labels_to_id,
        "id2label": id_to_class_labels,
        "bins": bins,
    }


def get_next_path(
    base_fname: str,
    base_dir: Path,
    file_type: str = "yaml",
    separator: str = "-",
    lock_file: str = "next_path.lock",
) -> Path:
    """
    Gets the next available path in a directory. For example, if `base_fname="results"`
    and `base_dir` has files ["results-0.yaml", "results-1.yaml"], this function returns
    "results-2.yaml".
    """
    lock_path = base_dir / lock_file
    lock = FileLock(lock_path, timeout=10)

    try:
        with lock:
            if file_type == "":
                # Directory
                items = filter(
                    lambda x: x.is_dir()
                    and re.match(f"^{base_fname}{separator}\\d+$", x.stem),
                    base_dir.glob("*"),
                )
            else:
                # File
                items = filter(
                    lambda x: re.match(f"^{base_fname}{separator}\\d+$", x.stem),
                    base_dir.glob(f"*.{file_type}"),
                )
            run_nums = list(
                map(lambda x: int(x.stem.replace(base_fname + separator, "")), items)
            ) + [-1]

            next_num = max(run_nums) + 1
            fname = f"{base_fname}{separator}{next_num}" + (
                f".{file_type}" if file_type != "" else ""
            )

            result = base_dir / fname

            if result.is_dir():
                # make this directory so that future invocations of this function do
                # not collide
                result.mkdir(parents=True)

            # lock is released when this code is reached
            return result
    except TimeoutError:
        raise RuntimeError(f"Could not acquire lock on {lock_path}")
