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
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import List, Iterator, Optional, Dict, Literal

import typer
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

app = typer.Typer(pretty_exceptions_enable=False)

import sys
for pth in sys.path:
    if pth.endswith("scripts/eval"):
        sys.path.append(pth.replace("scripts/eval", ""))
        break

from src.chronos.trainer import Trainer
from scripts.train.train import load_model, ChronosDataset, agg_relative_score, log_on_main, has_enough_observations

from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.model.evaluation import evaluate_forecasts

from chronos import ChronosConfig, ChronosTokenizer, ChronosPipeline, ChronosModel
from transformers.trainer_utils import EvalLoopOutput
from scripts.eval.evaluate import load_and_split_dataset, generate_sample_forecasts 
import pandas as pd
import yaml
from transformers.trainer_utils import denumpify_detensorize
from scipy.stats import gmean
import time
import wandb

def run_evaluation(trainer, step_number) -> None:
    """
    Evaluation loop, updated to work with Chronos
    """

    # dump the metrics into the eval_dict json
    try:
        with open(trainer.eval_dict_path, "r") as f:
            eval_metrics = json.load(f)
    except IOError:
        print(f"initializing {trainer.eval_dict_path}")
        eval_metrics = {}
    if step_number in eval_metrics.keys():
        return None
    
    print("COMPUTING METRICS FOR step_number = ", step_number)

    start_time = time.time()
    temperature = None
    top_k = None
    top_p = None

    with open(trainer.eval_config_path) as fp:
        backtest_configs = yaml.safe_load(fp)

    result_rows = []
    for config in backtest_configs:
        dataset_name = config["name"]

        prediction_length = config["prediction_length"]

        logger.info(f"Loading {dataset_name}")
        test_data = load_and_split_dataset(backtest_config=config)

        logger.info(
            f"Generating forecasts for {dataset_name} "
            f"({len(test_data.input)} time series)"
        )
        sample_forecasts = generate_sample_forecasts(
            test_data.input,
            pipeline=trainer.pipeline,
            prediction_length=prediction_length,
            batch_size=trainer.args.eval_batch_size,
            num_samples=trainer.chronos_config.num_samples,
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

    if trainer.eval_config_path == "scripts/eval/configs/zero-shot.yaml":
        baseline_df = pd.read_csv(
            "scripts/eval/seasonal-naive-zero-shot.csv"
        ).set_index("dataset")
    else:
        raise NotImplementedError

    agg_score_df = agg_relative_score(results_df.set_index("dataset"), baseline_df)

    metrics_dict = agg_score_df.to_dict()

    # add metrics in each dataset
    results_dict_per_dataset = results_df.set_index('dataset')[['MASE', 'WQL']].T.to_dict()
    metrics_dict.update(results_dict_per_dataset)

    # To be JSON-serializable, we need to remove numpy types or zero-d tensors
    metrics = denumpify_detensorize(metrics_dict)

    # dump the metrics into the eval_dict json
    try:
        with open(trainer.eval_dict_path, "r") as f:
            eval_metrics = json.load(f)
    except IOError:
        print(f"initializing {trainer.eval_dict_path}")
        eval_metrics = {}

    if str(step_number) not in eval_metrics.keys():
        eval_metrics[str(step_number)] = metrics
    else:
        eval_metrics[str(step_number)].update(metrics)

    with open(trainer.eval_dict_path, "w") as fp:
        json.dump(eval_metrics, fp, indent=4)

    print(f"    Eval loop finished. Total time: {(time.time() - start_time)/60} minutes")


    return None




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
    lm_head_type: str = "linear", # options: [linear, fourier, gaussian]
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
    start_evaluation(
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


def start_evaluation(
    # the default arguments have been edited, the defaults are different for
    # main. Below are config that work by default for the fourier project. They were
    # copied from scripts/training/configs/07-05/chronos-t5-mini-linear.yaml
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
    lm_head_type: str = "linear", # options: [linear, fourier, gaussian]
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
    print("eval_config_paths:", eval_config_paths)

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

    if tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        tf32 = False

    if seed is None:
        seed = random.randint(0, 2**32)

    transformers.set_seed(seed=seed)
    assert model_type in ["seq2seq", "causal"]

    raw_training_config = deepcopy(locals())

    # uses config file from output_dir to find the checkpoint dirs
    config_idx = sys.argv.index("--config")
    src_config = sys.argv[config_idx+1]
    output_dir = Path("/".join(src_config.split("/")[:-1]))

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

    model = load_model(
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

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=shuffled_train_dataset,
        eval_dataset=shuffled_train_dataset # it's not actually this, don't worry... we just need to pass SOMETHING in. TODO: create a dummy dataset to pass instead...
    )
    eval_config_path = eval_config_paths[0]
    # trainer = update_evaluation_loop(trainer, chronos_tokenizer, chronos_config, eval_config_path)
    # log_on_main("Training", logger)

    if "zero-shot" in eval_config_path:
        trainer.eval_dict_path = os.path.join(trainer.args.output_dir, "eval_dict-zero-shot.json")
    else:
        raise NotImplementedError

    trainer.chronos_config = chronos_config   
    trainer.eval_config_path = eval_config_path

    # import time; time.sleep(60*120) # sleep two hours
    
    for _ in range(20):

        checkpoint_dirs = [name for name in os.listdir(output_dir) if name.startswith("checkpoint-") and not name.endswith("-final")]
        checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x : int(x.split("-")[1]))
        print("checkpoint_dirs found: ", checkpoint_dirs)
        
        for checkpoint_name in checkpoint_dirs[::-1]:

            start_time = time.time()
            print("checkpoint_name:", checkpoint_name)

            checkpoint_path = os.path.join(output_dir, checkpoint_name, "pytorch_model.bin")
            checkpoint = torch.load(checkpoint_path)
            trainer.model.load_state_dict(checkpoint)

            trainer.pipeline = ChronosPipeline(
                tokenizer=chronos_tokenizer,
                model=ChronosModel(config=chronos_config, model=trainer.model)
            )

            step_number = checkpoint_name.split("-")[-1]
            run_evaluation(trainer, step_number)
            print(f"    Runtime for evaluating this checkpoint: {(time.time() - start_time)/60} minutes")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    app()
