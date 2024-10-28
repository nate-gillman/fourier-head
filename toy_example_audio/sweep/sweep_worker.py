import os
import traceback
import argparse
import collections
from collections import defaultdict
from pathlib import Path
from typing import List

from json import loads

import numpy as np
import yaml
import wandb

from experiment.training import start_training
from config import SWEEP_ARTIFACT_FOLDER
from sweep.util import use_770_permissions


def start(
    sweep_id: str, sweep_name: str, wandb_project_name: str, base_config_yaml_path: str
) -> None:
    with use_770_permissions():
        with open(base_config_yaml_path, "r") as f:
            base_training_config = yaml.safe_load(f)

        # the hyperparameters assigned to this sweep will be in the below object
        # they are auto-assigned by wandb server at read time
        job_config = wandb.config

        output_dir = (SWEEP_ARTIFACT_FOLDER / wandb_project_name) / sweep_name

        run_settings = dict(job_config)
        run_settings["output_dir"] = str(output_dir.absolute())

        # overwrite the base training configuration with the values defined in the sweep
        # but preserve all other settings
        for k, v in run_settings.items():
            base_training_config[k] = v

        # this defines what the run will use as its arguments, and will be written
        # to its own yaml file in the run directory, along with logging artifacts, etc.
        train_conf = base_training_config

        start_training(**train_conf, wandb_project=wandb_project_name)


def wrapped_start(
    sweep_id: str, sweep_name: str, wandb_project_name: str, base_config_yaml_path: str
) -> None:
    # start the job, but catch any exception and log it in wandb with a traceback.
    try:
        wandb.init()
        start(sweep_id, sweep_name, wandb_project_name, base_config_yaml_path)
    except Exception as e:
        wandb.log({"error": str(e), "traceback": traceback.format_exc()})
        raise e


if __name__ == "__main__":
    """
    This is meant to be called by sweep_runner.py.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--sweep_name", type=str, required=True)
    parser.add_argument("--wandb_project_name", type=str, required=True)
    parser.add_argument(
        "--base_config_yaml_path",
        type=str,
        required=True,
        help="A yaml file that defines the base training settings. Every parameter not defined / changed in the sweep is preserved.",
    )
    args = parser.parse_args()

    # interesting bug, see: https://github.com/wandb/wandb/issues/5272#issuecomment-1881950880
    # only applies when doing hyperparameter sweeps with slurm.
    os.environ["WANDB_DISABLE_SERVICE"] = "True"

    # get the wandb sweep ID and launch the agent to perform some runs
    # do only 1 run within this script. This is the recommended configuration by wandb, see more
    # information here: https://docs.wandb.ai/guides/sweeps/faq#how-should-i-run-sweeps-on-slurm
    wandb.agent(
        args.sweep_id,
        project=args.wandb_project_name,
        # bit awkward, there is apparently no way to recover the sweep ID from the config
        # https://community.wandb.ai/t/retrieving-sweep-id-when-starting-sweep-from-cl/2921/2
        function=lambda: wrapped_start(
            args.sweep_id,
            args.sweep_name,
            args.wandb_project_name,
            args.base_config_yaml_path,
        ),
        count=1,
    )
