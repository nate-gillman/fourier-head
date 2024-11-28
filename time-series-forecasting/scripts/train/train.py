# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import logging
import os
import re
import sys
import json
import itertools
import random
from filelock import FileLock
from copy import deepcopy
from pathlib import Path
import tempfile
from functools import partial
from typing import List, Iterator, Optional, Dict, Literal
import wandb

import typer
import yaml
from typer_config import use_yaml_config
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    T5Config,
    TrainingArguments,
)
import accelerate
import gluonts
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Cyclic, Map, Filter
from gluonts.transform import (
    FilterTransformation,
    TestSplitSampler,
    ValidationSplitSampler,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    MissingValueImputation,
    LeavesMissingValues,
    LastValueImputation,
)

from chronos import ChronosConfig, ChronosTokenizer
import shutil
import time

app = typer.Typer(pretty_exceptions_enable=False)

import sys

for pth in sys.path:
    if pth.endswith("scripts/train"):
        sys.path.append(pth.replace("scripts/train", ""))
        break
for pth in sys.path:
    if pth.endswith("time-series-forecasting/src"):
        sys.path.append(pth.replace("/src", "/scripts/train"))
        break

from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.model.evaluation import evaluate_forecasts

from chronos import ChronosConfig, ChronosTokenizer, ChronosPipeline, ChronosModel
from transformers.trainer_utils import EvalLoopOutput
from scripts.eval.evaluate import (
    load_and_split_dataset,
    generate_sample_forecasts,
)
import pandas as pd
import yaml
from transformers.trainer_utils import denumpify_detensorize
from scipy.stats import gmean

from src.chronos.t5 import T5ForConditionalGeneration
from src.chronos.trainer import Trainer

from t5_scaling_configs import t5_scaling_configs


def is_main_process() -> bool:
    """
    Check if we're on the main process.
    """
    if not dist.is_torchelastic_launched():
        return True
    return int(os.environ["RANK"]) == 0


def log_on_main(msg: str, logger: logging.Logger, log_level: int = logging.INFO):
    """
    Log the given message using the given logger, if we're on the main process.
    """
    if is_main_process():
        logger.log(log_level, msg)


def get_training_job_info() -> Dict:
    """
    Returns info about this training job.
    """
    job_info = {}

    # CUDA info
    job_info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        job_info["device_count"] = torch.cuda.device_count()

        job_info["device_names"] = {
            idx: torch.cuda.get_device_name(idx)
            for idx in range(torch.cuda.device_count())
        }
        job_info["mem_info"] = {
            idx: torch.cuda.mem_get_info(device=idx)
            for idx in range(torch.cuda.device_count())
        }

    # DDP info
    job_info["torchelastic_launched"] = dist.is_torchelastic_launched()

    if dist.is_torchelastic_launched():
        job_info["world_size"] = dist.get_world_size()

    # Versions
    job_info["python_version"] = sys.version.replace("\n", " ")
    job_info["torch_version"] = torch.__version__
    job_info["numpy_version"] = np.__version__
    job_info["gluonts_version"] = gluonts.__version__
    job_info["transformers_version"] = transformers.__version__
    job_info["accelerate_version"] = accelerate.__version__

    return job_info


def save_training_info(ckpt_path: Path, training_config: Dict):
    """
    Save info about this training job in a json file for documentation.
    """
    assert ckpt_path.is_dir()
    default = lambda o: f"(JSON not serializable): {repr(o)}"
    with open(ckpt_path / "training_info.json", "w") as fp:
        json.dump(
            {"training_config": training_config, "job_info": get_training_job_info()},
            fp,
            indent=4,
            default=default
        )


def get_next_path(
    base_fname: str,
    base_dir: Path,
    file_type: str = "yaml",
    separator: str = "-",
    lock_file: str = "next_path.lock"
):
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
                    lambda x: x.is_dir() and re.match(f"^{base_fname}{separator}\\d+$", x.stem),
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
        raise RuntimeError("Could not acquire lock on {lock_path}")


def load_model(
    model_id="google/t5-efficient-tiny",
    model_type="seq2seq",
    vocab_size=4096,
    random_init=False,
    tie_embeddings=False,
    pad_token_id=0,
    eos_token_id=1,
    lm_head_type="linear",
    fourier_kwargs: Optional[dict] = None,
) -> T5ForConditionalGeneration:
    """
    Load the specified HuggingFace model, adjusting the vocabulary
    size, special token IDs, and initialization options.

    This allows to set a model up for training on a new vocabulary
    of tokens.
    """
    
    assert model_type in ["seq2seq", "causal"]

    if random_init:
        try:
            log_on_main("Using random initialization", logger)
        except:
            pass

        config = AutoConfig.from_pretrained(model_id)
        config.vocab_size = vocab_size
        config.lm_head_type = lm_head_type
        config.fourier_kwargs = fourier_kwargs
        if isinstance(config, T5Config):
            # The default initializer_factor (1.0) in transformers is too large
            config.initializer_factor = 0.05
        config.tie_word_embeddings = tie_embeddings

        if fourier_kwargs["base_model_size_idx"] >= 0:
            # update them!!
            config_updates = t5_scaling_configs[fourier_kwargs["base_model_size_idx"]]

            config.d_ff                 = config_updates["d_ff"]
            config.d_kv                 = config_updates["d_kv"]
            config.d_model              = config_updates["d_model"]
            config.num_heads            = config_updates["num_heads"]
            config.num_decoder_layers   = config_updates["num_decoder_layers"]
            config.num_layers           = config_updates["num_layers"]

        model = T5ForConditionalGeneration(config)
        print(f"num_params = {sum(p.numel() for p in model.parameters())}")


    else:
        try:
            log_on_main(f"Using pretrained initialization from {model_id}", logger)
        except:
            pass

        # does not work for fourier quite yet
        try:
            model = T5ForConditionalGeneration.from_pretrained(model_id)
        except AttributeError:
            raise NotImplementedError(
                "random_init=False does not support fourier model."
            )

    model.resize_token_embeddings(vocab_size)

    model.config.pad_token_id = model.generation_config.pad_token_id = pad_token_id
    model.config.eos_token_id = model.generation_config.eos_token_id = eos_token_id

    return model


def has_enough_observations(
    entry: dict, min_length: int = 0, max_missing_prop: float = 1.0
) -> bool:
    """
    Check if the given entry has enough observations in the ``"target"`` attribute.

    Parameters
    ----------
    entry
        The data entry (dictionary) to be tested.
    min_length
        The minimum length the ``"target"`` attribute must have.
    max_missing_prop
        The maximum proportion of missing data allowed in the ``"target"``
        attribute.
    """
    if (
        len(entry["target"]) >= min_length
        and np.isnan(entry["target"]).mean() <= max_missing_prop
    ):
        return True
    return False


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffle entries from an iterable by temporarily accumulating them
    in an intermediate buffer.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_length
        Size of the buffer use to shuffle entries from the base dataset.
    """

    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    """
    Mix-in class that datasets can inherit from to get
    shuffling functionality.
    """

    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class ChronosDataset(IterableDataset, ShuffleMixin):
    """
    Dataset wrapper, using a ``ChronosTokenizer`` to turn data from a time series
    into a HuggingFace-compatible set of ``input_ids``, ``attention_mask`` and
    ``labels``.

    Entries from the original datasets are assumed to have a ``"start"`` attribute
    (of type ``pd.Period``), and a ``"target"`` attribute (of type ``np.ndarray``).

    Parameters
    ----------
    datasets
        Datasets containing the original time series data.
    probabilities
        In training mode, data will be sampled from each of the original datasets
        with these probabilities.
    tokenizer
        Tokenizer to be used to turn sequences of real numbers into token IDs.
    context_length
        Samples context will be limited to this length.
    prediction_length
        Samples labels will be limited to this length.
    drop_prob
        In training mode, observations from a sample will be turned into ``np.nan``,
        i.e. turned into missing values, with this probability.
    min_past
        Data samples will be considered only if there's at least ``min_past``-many
        historical observations.
    mode
        One of ``"training"``, ``"validation"``, or ``"test"``.
    np_dtype
        Numpy float data type.
    """

    def __init__(
        self,
        datasets: list,
        probabilities: List[float],
        tokenizer: ChronosTokenizer,
        context_length: int = 512,
        prediction_length: int = 64,
        drop_prob: float = 0.2,
        min_past: Optional[int] = None,
        model_type: str = "seq2seq",
        imputation_method: Optional[MissingValueImputation] = None,
        mode: str = "training",
        np_dtype=np.float32,
    ) -> None:
        super().__init__()

        assert len(probabilities) == len(datasets)
        assert mode in ("training", "validation", "test")
        assert model_type in ("seq2seq", "causal")

        self.datasets = datasets
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.drop_prob = drop_prob if model_type == "seq2seq" else 0.0
        self.min_past = min_past or prediction_length
        self.model_type = model_type
        self.imputation_method = imputation_method or LeavesMissingValues()
        self.mode = mode
        self.np_dtype = np_dtype

    def preprocess_entry(self, entry: dict, mode: str) -> dict:
        entry = {f: entry[f] for f in ["start", "target"]}
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)
        assert entry["target"].ndim == 1, f"got {entry['target'].ndim=}, expected 1"

        if self.model_type == "causal":
            # Causal models do not play nice with missing values, so it is
            # recommended to use an imputation method, e.g., LastValueImputation
            entry["target"] = self.imputation_method(entry["target"])

        if mode == "training" and self.drop_prob > 0:
            target = entry["target"].copy()
            drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
            mask = np.random.choice(
                [True, False], size=len(target), p=[drop_p, 1 - drop_p]
            )
            target[mask] = np.nan
            entry["target"] = target

        return entry

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "test", "validation"]

        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
        )

    def create_training_data(self, data):
        data = Cyclic(data)
        split_transform = self._create_instance_splitter(
            "training"
        ) + FilterTransformation(
            condition=lambda entry: (~np.isnan(entry["past_target"])).sum() > 0
        )
        data = split_transform.apply(data, is_train=True)
        return data

    def create_test_data(self, data):
        data = self._create_instance_splitter("test").apply(data, is_train=False)
        return data

    def create_validation_data(self, data):
        data = self._create_instance_splitter("validation").apply(data, is_train=False)
        return data

    def to_hf_format(self, entry: dict) -> dict:
        past_target = torch.tensor(entry["past_target"]).unsqueeze(0)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(
            past_target
        )
        future_target = torch.tensor(entry["future_target"]).unsqueeze(0)
        labels, labels_mask = self.tokenizer.label_input_transform(future_target, scale)
        labels[labels_mask == 0] = -100

        if self.model_type == "causal":
            # The InstanceSplitter pads time series on the left to be equal to the
            # context_length. However, certain models (e.g., GPT2) with absolute
            # position embeddings should not be trained with left padding.
            # The following piece of code moves padding from left to right.

            assert input_ids.shape[-1] == entry["past_is_pad"].shape[0]

            # Find the index where padding starts
            pad_start_idx = np.searchsorted(1 - entry["past_is_pad"], 1)
            padded_input_ids, obs_input_ids = torch.tensor_split(
                input_ids, [pad_start_idx], dim=-1
            )
            padded_attention_mask, obs_attention_mask = torch.tensor_split(
                attention_mask, [pad_start_idx], dim=-1
            )

            # Move padding to the right
            input_ids = torch.cat(
                [
                    obs_input_ids,
                    labels,
                    padded_input_ids,
                ],
                axis=-1,
            )
            attention_mask = torch.cat(
                [
                    obs_attention_mask,
                    labels_mask,
                    padded_attention_mask,
                ],
                axis=-1,
            )

            # labels for causal models are same as the input_ids.
            # Internally transformers shifts the labels by one during training.
            labels = input_ids.clone()
            input_ids[~attention_mask] = self.tokenizer.config.pad_token_id
            labels[~attention_mask] = -100

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def __iter__(self) -> Iterator:
        preprocessed_datasets = [
            Map(
                partial(self.preprocess_entry, mode=self.mode),
                dataset,
            )
            for dataset in self.datasets
        ]

        if self.mode == "training":
            iterables = [
                self.create_training_data(dataset) for dataset in preprocessed_datasets
            ]
        elif self.mode == "test":
            iterables = [
                self.create_test_data(dataset) for dataset in preprocessed_datasets
            ]
        else:
            iterables = [
                self.create_validation_data(dataset)
                for dataset in preprocessed_datasets
            ]

        worker_info = get_worker_info()
        if worker_info is None:
            probs = list(self.probabilities)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iterables = list(itertools.islice(iterables, worker_id, None, num_workers))
            probs = list(
                itertools.islice(self.probabilities, worker_id, None, num_workers)
            )

        probs = [prob / sum(probs) for prob in probs]

        iterators = list(map(iter, iterables))
        if self.mode == "training":
            while True:
                idx = np.random.choice(range(len(iterators)), p=probs)
                try:
                    yield self.to_hf_format(next(iterators[idx]))
                except StopIteration:
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [prob / sum(probs) for prob in probs]
        else:
            for entry in itertools.chain(*iterators):
                yield self.to_hf_format(entry)


def agg_relative_score(model_df: pd.DataFrame, baseline_df: pd.DataFrame):
    relative_score = model_df.drop("model", axis="columns") / baseline_df.drop(
        "model", axis="columns"
    )
    return relative_score.agg(gmean)


def update_evaluation_loop(
    trainer: Trainer,
    tokenizer,
    chronos_config,
    eval_config_path,
    we_are_training_on_subset_of_train=False
):
    """
    This overwrites Trainer.evaluation_loop with a custom evaluation function. This is necessary because the default
    one in the Trainer class is only set up to handle evaluation with teacher forcing (yuck) and it seems like the
    huggingface team doesn't intend to fix this, see e.g. https://github.com/huggingface/transformers/issues/23763
    """
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    trainer.eval_dict_path = os.path.join(trainer.args.output_dir, "eval_dict.json")

    trainer.pipeline = ChronosPipeline(
        tokenizer=tokenizer,
        model=ChronosModel(config=chronos_config, model=trainer.model),
    )
    trainer.chronos_config = chronos_config
    trainer.eval_config_path = eval_config_path
    trainer.we_are_training_on_subset_of_train = we_are_training_on_subset_of_train
    trainer.we_are_training_on_subset_of_train_subset_size = 16

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Evaluation loop, updated to work with Chronos
        """

        start_time = time.time()
        temperature = None
        top_k = None
        top_p = None

        with open(self.eval_config_path) as fp:
            backtest_configs = yaml.safe_load(fp)

        result_rows = []
        for config in backtest_configs:
            dataset_name = config["name"]
            prediction_length = config["prediction_length"]

            logger.info(f"Loading {dataset_name}")

            subset_size = np.inf
            if self.we_are_training_on_subset_of_train:
                subset_size = self.we_are_training_on_subset_of_train_subset_size
            test_data = load_and_split_dataset(backtest_config=config, subset_size=subset_size)

            logger.info(
                f"Generating forecasts for {dataset_name} "
                f"({len(test_data.input)} time series)"
            )
            sample_forecasts = generate_sample_forecasts(
                test_data.input,
                pipeline=self.pipeline,
                prediction_length=prediction_length,
                batch_size=self.args.eval_batch_size,
                num_samples=self.chronos_config.num_samples,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            logger.info(f"Evaluating forecasts for {dataset_name}")

            metrics = (
                evaluate_forecasts(
                    sample_forecasts,
                    test_data=test_data,
                    metrics=[
                        MASE(),
                        MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
                    ],
                    batch_size=5000,
                )
                .reset_index(drop=True)
                .to_dict(orient="records")
            )
            # print("------UPDATE_EVALUATION_LOOP-------")
            log_metric_object_as = {dataset_name: {'metrics': next(iter(metrics), {}), 'prediction_length': prediction_length }}

            if np.isnan(log_metric_object_as[dataset_name]["metrics"]['MASE[0.5]']) or np.isnan(log_metric_object_as[dataset_name]["metrics"]['mean_weighted_sum_quantile_loss']):
                import pdb; pdb.set_trace()
                print("----"*20,"----"*20, f"Got some NANs on dataset {dataset_name}. Setting those metric values to 1.")
                print("test_data: ", test_data)
                print("sample_forecasts: ", sample_forecasts)

            print(log_metric_object_as)
            wandb.log(log_metric_object_as)
            # print("-----------------------------------")
            result_rows.append(
                {"dataset": dataset_name, "model": "chronos_model_id", **metrics[0]}
            )

        results_df = (
            pd.DataFrame(result_rows)
            .rename(
                {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"},
                axis="columns",
            )
            .sort_values(by="dataset")
        )

        baseline_df = pd.read_csv("scripts/eval/seasonal-naive-zero-shot.csv").set_index("dataset")

        agg_score_df = agg_relative_score(results_df.set_index("dataset"), baseline_df)

        metrics_dict = agg_score_df.to_dict()

        # add metrics in each dataset
        results_dict_per_dataset = results_df.set_index('dataset')[['MASE', 'WQL']].T.to_dict()
        metrics_dict.update(results_dict_per_dataset)

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics_dict)

        # dump the metrics into the eval_dict json
        try:
            with open(self.eval_dict_path, "r") as f:
                eval_metrics = json.load(f)
        except IOError:
            print(f"initializing {self.eval_dict_path}")
            eval_metrics = {}

        if str(self.state.global_step) not in eval_metrics.keys():
            eval_metrics[str(self.state.global_step)] = metrics
        else:
            eval_metrics[str(self.state.global_step)].update(metrics)

        with open(self.eval_dict_path, "w") as fp:
            json.dump(eval_metrics, fp, indent=4)

        print(f"    Eval loop finished. Total time: {(time.time() - start_time)/60} minutes")

        # just return something dummy to fit the contract
        return EvalLoopOutput(
            predictions=np.zeros(4), label_ids=None, metrics=metrics, num_samples=0
        )

    trainer.evaluation_loop = (
        lambda dataloader,
        description,
        prediction_loss_only,
        ignore_keys,
        metric_key_prefix: evaluation_loop(
            trainer,
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )
    )

    return trainer


@app.command()
@use_yaml_config(param_name="config")
def main(
    training_data_paths: str,
    eval_config_paths: str,
    probability: Optional[str] = None,
    context_length: int = 512,
    prediction_length: int = 64,
    min_past: int = 64,
    max_steps: int = 200_000,
    save_steps: int = 50_000,
    eval_steps: int = 50_000,
    log_steps: int = 500,
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 32,
    learning_rate: float = 1e-3,
    optim: str = "adamw_torch_fused",
    shuffle_buffer_length: int = 100,
    gradient_accumulation_steps: int = 2,
    model_id: str = "google/t5-efficient-tiny",
    model_type: str = "seq2seq",
    random_init: bool = False,
    tie_embeddings: bool = False,
    output_dir: str = "./output/",
    tf32: bool = True,
    torch_compile: bool = True,
    tokenizer_class: str = "MeanScaleUniformBins",
    tokenizer_kwargs: str = "{'low_limit': -15.0, 'high_limit': 15.0}",
    n_tokens: int = 4096,
    n_special_tokens: int = 2,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    use_eos_token: bool = True,
    max_grad_norm: float = 1.0,
    lm_head_type: str = "linear", # options: [linear, fourier]
    fourier_kwargs: str = "{'fourier_num_frequences': 16}",
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    dataloader_num_workers: int = 1,
    max_missing_prop: float = 0.9,
    num_samples: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    weight_decay: float = 0.0,
):
    """This is just so that the training function can be called from other code, but the setup with
    the @app.command / yaml config magic is preserved.
    """
    start_training(
        training_data_paths,
        eval_config_paths,
        probability,
        context_length,
        prediction_length,
        min_past,
        max_steps,
        save_steps,
        eval_steps,
        log_steps,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        learning_rate,
        optim,
        shuffle_buffer_length,
        gradient_accumulation_steps,
        model_id,
        model_type,
        random_init,
        tie_embeddings,
        output_dir,
        tf32,
        torch_compile,
        tokenizer_class,
        tokenizer_kwargs,
        n_tokens,
        n_special_tokens,
        pad_token_id,
        eos_token_id,
        use_eos_token,
        max_grad_norm,
        lm_head_type,
        fourier_kwargs,
        lr_scheduler_type,
        warmup_ratio,
        dataloader_num_workers,
        max_missing_prop,
        num_samples,
        temperature,
        top_k,
        top_p,
        seed,
        weight_decay,
    )


def start_training(
    # the default arguments have been edited, the defaults are different for
    # main. Below are config that work by default for the fourier project.
    training_data_paths: str,
    eval_config_paths: str,
    probability: Optional[str] = None,
    context_length: int = 512,
    prediction_length: int = 64,
    min_past: int = 60,
    max_steps: int = 200_000,
    save_steps: int = 50_000,
    eval_steps: int = 50_000,
    log_steps: int = 1_000,
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 32,
    learning_rate: float = 1e-3,
    optim: str = "adamw_torch_fused",
    shuffle_buffer_length: int = 100_000,
    gradient_accumulation_steps: int = 1,
    model_id: str = "google/t5-efficient-mini",
    model_type: str = "seq2seq",
    random_init: bool = True,
    tie_embeddings: bool = True,
    output_dir: str = "./output/",
    tf32: bool = True,
    torch_compile: bool = True,
    tokenizer_class: str = "MeanScaleUniformBins",
    tokenizer_kwargs: str = "{'low_limit': -15.0, 'high_limit': 15.0}",
    n_tokens: int = 4096,
    n_special_tokens: int = 2,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    use_eos_token: bool = True,
    max_grad_norm: float = 1.0,
    lm_head_type: str = "linear", # options: [linear, fourier]
    fourier_kwargs: str = "{'fourier_num_frequences': 16}",
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    dataloader_num_workers: int = 1,
    max_missing_prop: float = 0.9,
    num_samples: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    weight_decay: float = 0.0,
    # new arguments for wandb hparam sweeps
    log_in_wandb: bool = False,
    wandb_project: str = "",
    wandb_watch: Literal["", "all", "gradients"] = "",
):
    if not isinstance(training_data_paths, list):
        training_data_paths = ast.literal_eval(training_data_paths)
    assert isinstance(training_data_paths, list)

    if not isinstance(eval_config_paths, list):
        eval_config_paths = ast.literal_eval(eval_config_paths)
    assert isinstance(eval_config_paths, list)
    assert(len(eval_config_paths) == 1) # only one eval set is implemented at the moment

    if isinstance(probability, str):
        probability = ast.literal_eval(probability)
    elif probability is None:
        probability = [1.0 / len(training_data_paths)] * len(training_data_paths)
    assert isinstance(probability, list)

    if isinstance(tokenizer_kwargs, str):
        tokenizer_kwargs = ast.literal_eval(tokenizer_kwargs)
    assert isinstance(tokenizer_kwargs, dict)

    if isinstance(fourier_kwargs, str):
        fourier_kwargs = ast.literal_eval(fourier_kwargs)
    assert isinstance(fourier_kwargs, dict)

    # do this first before anything, before any local variables are declared
    # this is also saved at the end of training, see 'raw_training_config'. We save
    # this first for parity with the CLI version of the training procedure, which
    # requires a config yaml file that is copied to the run folder.
    training_args = deepcopy(locals())
    do_not_save_params = ["log_in_wandb", "wandb_project", "wandb_watch"]
    for c in do_not_save_params:
        training_args.pop(c)
    in_memory_yaml = yaml.dump(training_args)

    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    if tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        # TF32 floating point format is available only on NVIDIA GPUs
        # with compute capability 8 and above. See link for details.
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
        log_on_main(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
            logger,
        )
        tf32 = False

    if seed is None:
        seed = random.randint(0, 2**32)
    log_on_main(f"Using SEED: {seed}", logger)
    transformers.set_seed(seed=seed)
    assert model_type in ["seq2seq", "causal"]

    raw_training_config = deepcopy(locals())
    output_dir = Path(output_dir)
    output_dir = get_next_path("run", base_dir=output_dir, file_type="")

    log_on_main(f"Logging dir: {output_dir}", logger)
    log_on_main(
        f"Loading and filtering {len(training_data_paths)} datasets "
        f"for training: {training_data_paths}",
        logger,
    )

    log_on_main(
        f"Mixing probabilities: {probability}",
        logger,
    )

    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=min_past + prediction_length,
                max_missing_prop=max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h"),
        )
        for data_path in training_data_paths
    ]

    log_on_main("Initializing model", logger)

    model: T5ForConditionalGeneration = load_model(
        model_id=model_id,
        model_type=model_type,
        vocab_size=n_tokens,
        random_init=random_init,
        tie_embeddings=tie_embeddings,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        lm_head_type=lm_head_type,
        fourier_kwargs=fourier_kwargs,
    )

    chronos_config = ChronosConfig(
        tokenizer_class=tokenizer_class,
        tokenizer_kwargs=tokenizer_kwargs,
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_eos_token=use_eos_token,
        model_type=model_type,
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Add extra items to model config so that it's saved in the ckpt
    model.config.chronos_config = chronos_config.__dict__
    chronos_tokenizer = chronos_config.create_tokenizer()
    shuffled_train_dataset = ChronosDataset(
        datasets=train_datasets,
        probabilities=probability,
        tokenizer=chronos_tokenizer,
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=min_past,
        model_type=model_type,
        imputation_method=LastValueImputation() if model_type == "causal" else None,
        mode="training",
    ).shuffle(shuffle_buffer_length=shuffle_buffer_length)

    # Define training args
    # https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
    # see the docs on wandb + hf transformers: https://docs.wandb.ai/guides/integrations/huggingface#-next-level-logging-in-few-lines
    if log_in_wandb:
        os.environ["WANDB_PROJECT"] = wandb_project
        if wandb_watch:
            os.environ["WANDB_WATCH"] = wandb_watch
            # log the gradients and compute graph of the Linear, Fourier, and Guassian LM Head that we
            # place at the end of T5.
            wandb.watch(
                model.lm_head, log=wandb_watch, log_freq=log_steps, log_graph=True
            )
    else:
        # kind of interesting, we need to explicitly disable wandb logging if argument is false
        # somewhere downstream in the hf transformers library, if we do not do this, it will
        # start logging if some wandb-related environment variables are set to truthy values.
        wandb.init(mode="disabled")

    report_to = "wandb" if log_in_wandb else None
    # can add max grad norm here...
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        optim=optim,
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=log_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_safetensors=False,
        save_steps=save_steps,
        report_to=report_to,
        max_steps=max_steps,
        max_grad_norm=max_grad_norm,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_accumulation_steps=10,
        dataloader_num_workers=dataloader_num_workers,
        tf32=tf32,  # remove this if not using Ampere GPUs (e.g., A100)
        # turn `torch_compile` to False if want to skip dynamic graph compilation, helpful
        # for interactive debugging
        # torch_compile=torch_compile,
        torch_compile=False,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    temp_dir = None
    try:
        # this was invoked in the CLI and was given a file
        config_idx = sys.argv.index("--config")
        src_config: str = sys.argv[config_idx + 1]
    except ValueError:
        # this was not invoked with the CLI and there is no --config file
        # but we have all the arguments as an in-memory yaml file
        temp_dir = tempfile.TemporaryDirectory()
        temp_conf = Path(temp_dir.name) / "config.yaml"
        temp_conf.write_text(in_memory_yaml)
        src_config: str = str(temp_conf.absolute())

    # copy the config file to the output_dir, for reproducibility
    dst_config = os.path.join(output_dir, src_config.split("/")[-1])
    dummy_eval_dataset = {} # this ensures that eval doesn't happen during training loop
    if torch.cuda.device_count() == 1:
        os.makedirs(output_dir, exist_ok=True)
        shutil.copyfile(src_config, dst_config)
    elif torch.cuda.device_count() > 1:
        if torch.distributed.get_rank() == 0:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copyfile(src_config, dst_config)

    if temp_dir:
        # if the config was a temporary file, remove temporary scratch locations
        temp_dir.cleanup()

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=shuffled_train_dataset,
        eval_dataset=dummy_eval_dataset,  # it's not actually this, don't worry... we just need to pass SOMETHING in. TODO: create a dummy dataset to pass instead...
    )

    WE_ARE_TRAINING_ON_SUBSET_OF_TRAIN = "0.arrow" in training_data_paths[0]
    eval_config_path = eval_config_paths[0]
    training_data_paths = None
    trainer = update_evaluation_loop(
        trainer, chronos_tokenizer, chronos_config, eval_config_path, 
        we_are_training_on_subset_of_train=WE_ARE_TRAINING_ON_SUBSET_OF_TRAIN
    )
    log_on_main("Training", logger)

    trainer.train()

    if is_main_process():
        model.save_pretrained(output_dir / "checkpoint-final")
        save_training_info(
            output_dir / "checkpoint-final", training_config=raw_training_config
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    app()
