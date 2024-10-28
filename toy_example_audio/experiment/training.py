"""
Adapted from: https://renumics.com/blog/how-to-fine-tune-the-audio-spectrogram-transformer
Thank you / credit to Marius Steger for the helpful guide.
"""
# allow import of smoothness metric and fourier head
import sys
sys.path.append("..")

# ----
import random
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
import warnings
import json
import torch
import wandb
import numpy as np
import evaluate
from datasets import Audio, ClassLabel, load_dataset, Dataset
from transformers import (
    ASTFeatureExtractor,
    ASTConfig,
    ASTForAudioClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction

from fourier_head import Fourier_Head
from smoothness_metric import get_l2_smoothness_measurement_function

from experiment.load_data import get_dataset, get_next_path
from config import SWEEP_ARTIFACT_FOLDER, AUDIO_REPO_ROOT, PROJECT_SEED

# set seeds on several sources of variation. Even with all this there might be some non-determinism. 
set_seed(PROJECT_SEED)

# --------------- CONFIG --------------
# this is trained on: https://research.google.com/audioset/
HF_PRETRAINED_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
# -------------------------------------


def setup_feature_extrator(feature_extractor: ASTFeatureExtractor, dataset: Dataset) -> ASTFeatureExtractor:
    # calculate values for normalization
    # we set normalization to False in order to calculate the mean + std of the dataset
    feature_extractor.do_normalize = False
    mean = []
    std = []
    # get features of input audio dist for normalization
    for i, sample in enumerate(dataset["train"]):
        audio_input = sample["audio"]["array"]
        cur_mean = np.mean(audio_input)
        cur_std = np.std(audio_input)
        mean.append(cur_mean)
        std.append(cur_std)
    feature_extractor.mean = np.mean(mean)
    feature_extractor.std = np.mean(std)
    feature_extractor.do_normalize = True
    return feature_extractor


def preprocess_audio(batch, model_input_name: str, sr: int, feature_extractor: ASTFeatureExtractor) -> Dict[str, Any]:
    """
    Define a transformation over the test set samples.
    """
    # get the audio from the batch, don't do any data augmentation
    wavs = [audio["array"] for audio in batch[model_input_name]]

    # inputs are spectrograms as torch.tensors now
    inputs = feature_extractor(wavs, sampling_rate=sr, return_tensors="pt")
    output_batch = {
        model_input_name: inputs.get(model_input_name),
        "labels": list(batch["labels"]),
    }
    return output_batch


def softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.amax(x, axis=1, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=1, keepdims=True)


def reduce_bucket(bucket_bounds: Tuple[int, int]) -> float:
    a, b = bucket_bounds
    return (a + b) / 2


def start_training(
    freeze_encoder: bool = False,
    hf_pretrained_model_name: str = HF_PRETRAINED_MODEL_NAME,
    dataset_bin_size: int = 15,
    fourier_num_frequencies: int = 1,
    fourier_regularization_gamma: float = 0.000001,
    num_train_epochs: int = 2,
    learning_rate: float = 5e-5,
    eval_every_n_epochs: int = 1,
    save_every_n_epochs: int = 1,
    log_every_n_steps: int = 20,
    wandb_project: str = "",
    output_dir: Optional[Path] = None,
    dataset_split_style: str = "standard",
) -> None:
    # gets a copy of the arguments passed to training
    start_training_function_arguments = json.dumps({**locals()}, default=repr)

    # --- CONFIGURE DATASET AND FEATURE EXTRACTION ---
    # we define which pretrained model we want to use and instantiate a feature extractor
    with warnings.catch_warnings(action='ignore', category=UserWarning):
        feature_extractor = ASTFeatureExtractor.from_pretrained(hf_pretrained_model_name)

    model_input_name = feature_extractor.model_input_names[0]  # this is 'input_values'
    SAMPLING_RATE = feature_extractor.sampling_rate

    dataset, dataset_info = get_dataset(
        bin_size=dataset_bin_size,
        split_style=dataset_split_style,
        random_seed=PROJECT_SEED,
    )
    feature_extractor = setup_feature_extrator(feature_extractor, dataset)
    dataset = dataset.rename_column("audio", model_input_name)
    dataset = dataset.cast_column(
        model_input_name, Audio(sampling_rate=SAMPLING_RATE)
    )

    # set transformations for each sample, this just runs feature extraction over the
    # waveforms for the AST model.
    dataset["train"].set_transform(
        lambda x: preprocess_audio(
            x, model_input_name, SAMPLING_RATE, feature_extractor, 
        ),
        output_all_columns=False,
    )
    dataset["test"].set_transform(
        lambda x: preprocess_audio(
            x, model_input_name, SAMPLING_RATE, feature_extractor
        ),
        output_all_columns=False,
    )

    # --- CONFIGURE MODEL TO FINETUNE ---
    # load pretrained model
    config = ASTConfig.from_pretrained(hf_pretrained_model_name)

    # add configuration
    config.num_labels = dataset_info["num_labels"]
    config.label2id = dataset_info["label2id"]
    config.id2label = dataset_info["id2label"]

    # initialize the model with the updated configuration
    model = ASTForAudioClassification.from_pretrained(
        hf_pretrained_model_name, config=config, ignore_mismatched_sizes=True
    )
    model.init_weights()
    if freeze_encoder:
        # freeze the weights of the base model
        for name, param in model.base_model.named_parameters():
            param.requires_grad = False

    # --- CONFIGURE LM HEAD ---
    # this is so sweep is simpler, but makes it a bit harder to understand after the fact
    #  -1: gaussian
    #   0: linear
    # >=1: fourier
    if fourier_num_frequencies < -1:
        raise ValueError(
            f"Invalid Number of Fourier Frequencies. Got: {fourier_num_frequencies}"
        )

    if fourier_num_frequencies == -1:
        # gaussian
        raise NotImplementedError("Truncated Gaussian is not implemented.")
    elif fourier_num_frequencies == 0:
        # linear
        pass
    else:
        # fourier
        model.classifier = Fourier_Head(
            dim_input=config.hidden_size,
            dim_output=config.num_labels,
            num_frequencies=fourier_num_frequencies,
            regularizion_gamma=fourier_regularization_gamma,
        )

    # --- CONFIGURE METRICS ---
    accuracy = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")
    _mse = evaluate.load("mse")
    
    def mse(a: np.ndarray, b: np.ndarray) -> float:
        with warnings.catch_warnings(action="ignore", category=FutureWarning):
            # suppresses the warning: FutureWarning: 'squared' is deprecated in version 1.4 and 
            # will be removed in 1.6. To calculate the root mean squared error, use 
            # the function'root_mean_squared_error'
            return _mse.compute(
                predictions=a, references=b
            )["mse"]

    AVERAGE = "macro" if config.num_labels > 2 else "binary"
    get_smoothness = get_l2_smoothness_measurement_function()

    total_evals = 0
    def compute_metrics(
        eval_pred: EvalPrediction,
        config: ASTConfig,
        fourier_num_frequencies: int,
        save_npy_to: Optional[Path] = None,
    ) -> Dict[str, Any]:
        nonlocal total_evals
        # this is numpy array at this point
        # (bs, n_classes)

        # Note that the baseline AST model with a linear layer at the end does not automatically apply a 
        # softmax, so eval_pred.predictions are logits before softmax. Source: 
        # https://huggingface.co/docs/transformers/v4.45.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput
        # This is just to verify and make clear that we are not applying softmax twice for any model configuration.
        logits: np.ndarray = eval_pred.predictions
        sm = softmax(logits)
        smoothness = np.array([get_smoothness(x) for x in sm])
        predictions = np.argmax(logits, axis=1)
        metrics: Dict[str, Any] = accuracy.compute(
            predictions=predictions, references=eval_pred.label_ids
        )
        metrics.update(
            precision.compute(
                predictions=predictions, references=eval_pred.label_ids, average=AVERAGE
            )
        )
        metrics.update(
            recall.compute(
                predictions=predictions, references=eval_pred.label_ids, average=AVERAGE
            )
        )
        metrics.update(
            f1.compute(
                predictions=predictions, references=eval_pred.label_ids, average=AVERAGE
            )
        )

        # differnece between the index of the labels
        mse_by_labels = mse(predictions, eval_pred.label_ids)

        # produces tuples of (lower_bound, upper_bound), both inclusive, for each bucket
        real_bucket_bounds_in_unit = [
            tuple(eval(config.id2label[x])) for x in eval_pred.label_ids
        ]
        pred_bucket_bounds_in_unit = [
            tuple(eval(config.id2label[x])) for x in predictions
        ]
        real_approx_vals_for_in_unit = [
            reduce_bucket(x) for x in real_bucket_bounds_in_unit
        ]
        pred_approx_vals_for_in_unit = [
            reduce_bucket(x) for x in pred_bucket_bounds_in_unit
        ]

        # differnece between the MIDPOINT of a bucket in the original unit in our problem
        mse_for_actuals = mse(pred_approx_vals_for_in_unit, real_approx_vals_for_in_unit)

        metrics.update(
            {
                "smoothness_mean": np.mean(smoothness),
                "smoothness_stdev": np.std(smoothness),
                "mse_by_label": mse_by_labels,
                "mse_by_bpm": mse_for_actuals,
            }
        )

        if output_dir:
            # save: eval_pred.predictions
            logit_save_path = (
                save_npy_to
                / f"{total_evals}-eval_pred_predictions-num_freq-{fourier_num_frequencies}.npy"
            )
            np.save(logit_save_path, logits, allow_pickle=False)

            # save: eval_pred.label_ids
            gt_save_path = (
                save_npy_to
                / f"{total_evals}-eval_pred_label_ids-num_freq-{fourier_num_frequencies}.npy"
            )
            np.save(gt_save_path, eval_pred.label_ids)

        total_evals += 1
        return metrics

    total_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    total_lm_head_params = sum(
        param.numel()
        for param in model.classifier.parameters()
        if param.requires_grad
    )

    # --- CONFIGURE REPORTING ---
    report_to = None
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        wandb_watch = "all"
        os.environ["WANDB_WATCH"] = wandb_watch
        wandb.watch(
            model.classifier,
            log=wandb_watch,
            log_freq=log_every_n_steps,
            log_graph=True,
        )
        report_to = "wandb"
        wandb.config.update(
            {
                "total_trainable_params": total_params,
                "lm_head_trainable_params": total_lm_head_params,
            }
        )
    else:
        # wandb project settings not given, disable
        wandb.init(mode="disabled")
        os.environ["WANDB_DISABLED"] = "true"

    # log to a specific place but default to the root of the repo in /runs and /logs
    base_output_dir = Path(output_dir or AUDIO_REPO_ROOT)
    base_output_dir = get_next_path("run", base_dir=base_output_dir, file_type="")
    artifacts_path = Path(base_output_dir)

    # --- CONFIGURE TRAINING ---
    training_args = TrainingArguments(
        output_dir=artifacts_path / "ast_classifier",
        logging_dir=artifacts_path / "ast_classifier",
        report_to=report_to,
        learning_rate=learning_rate,
        push_to_hub=False,
        num_train_epochs=num_train_epochs,
        seed=PROJECT_SEED,
        data_seed=PROJECT_SEED,
        # save and eval every k epochs
        # can be either: "epoch" (no 's') or "steps"
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_steps=eval_every_n_epochs,
        save_steps=eval_every_n_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_strategy="steps",
        logging_steps=log_every_n_steps,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=lambda x: compute_metrics(
            x,
            config,
            fourier_num_frequencies=fourier_num_frequencies,
            save_npy_to=base_output_dir,
        ),
    )

    # a lot of the same information is logged into the ../ast_classifier/checkpoint-n/... folders. This is more
    # human readable. 
    run_info = {
        # if an argument is not json serializable, this string representation may lose some information
        "start_training_function_arguments_json_string": start_training_function_arguments,
        "hf_transformers_training_args_json_string": training_args.to_json_string(),
        "sampling_rate": int(SAMPLING_RATE),
        "total_trainable_params": int(total_params),
        "lm_head_trainable_params": int(total_lm_head_params),
        "bins": dataset_info["bins"],
    }

    info_path = artifacts_path / "info.json"
    info_path.write_text(json.dumps(run_info))

    # --- TRAIN THE MODEL ---
    trainer.train()
