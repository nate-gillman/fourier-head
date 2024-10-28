import os
import sys
import subprocess
from typing import Tuple, Literal, Any, Dict
from pathlib import Path
import argparse
import yaml
import time
import warnings

import wandb

from sweep.util import use_770_permissions
from config import AUDIO_REPO_ROOT, SWEEP_ARTIFACT_FOLDER

# the location of the script that will execute a single run in the sweep. This script must accept
# only 2 arguments, which are: the wandb sweep ID (issued by wandb API) and the wandb project name (defined
# by the person running this sweep).
SWEEP_WORKER_SCRIPT_FILEPATH = Path(__file__).parent / "sweep_worker.py"

# below is the slurm job template that will be submitted with sbatch.
SLURM_JOB_BASE = r"""
#!/bin/bash
#SBATCH -p {slurm_partition} --gres=gpu:1
#SBATCH --constraint={hardware}
#SBATCH --exclude=gpu2106,gpu2108,gpu2112,gpu2114,gpu2115,gpu2116
#SBATCH -N 1
#SBATCH --mem={memory_in_gb}G
#SBATCH -t {total_time_string}
#SBATCH -J {wandb_project_name}_{wandb_sweep_id}
#SBATCH -e {logs_parent_directory}/{wandb_project_name}/slurm_logs/{wandb_sweep_id}-%j.err
#SBATCH -o {logs_parent_directory}/{wandb_project_name}/slurm_logs/{wandb_sweep_id}-%j.out

# Move to correct working directory
cd {repo_directory}
export PYTHONPATH=.

# SET UP COMPUTING ENV
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Load anaconda module, and other modules
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh

# Activate virtual environment
conda activate {conda_env_name}

# Run the script
python {sweep_worker_script_filepath} --sweep_id {wandb_sweep_id} --sweep_name {wandb_sweep_name} --wandb_project_name {wandb_project_name} --base_config_yaml_path {base_config_yaml_path}
"""


def get_slurm_job_script_for_workers(
    wandb_project_name: str,
    wandb_sweep_id: str,
    wandb_sweep_name: str,
    conda_env_name: str,
    sweep_worker_script_filepath: Path,
    base_config_yaml_path: Path,
    repo_directory: Path = AUDIO_REPO_ROOT,
    # partition can also be gpu-he or any other partition available to the user running the sweep
    # TODO: these can be CLI arguments
    slurm_partition: Literal["3090-gcondo", "a6000-gcondo", "gpu-he"] = "3090-gcondo",
    hardware: Literal[
        "a6000|geforce3090", "geforce3090", "a6000"
    ] = "a6000|geforce3090",
    memory_in_gb: str = "31",
    total_time_string: str = "24:00:00",
    logs_parent_directory: str = r"/users/%u/scratch/logs/wandb_sweeps",
) -> str:
    return SLURM_JOB_BASE.format(
        conda_env_name=conda_env_name,
        repo_directory=str(repo_directory.absolute()),
        slurm_partition=slurm_partition,
        hardware=hardware,
        memory_in_gb=memory_in_gb,
        total_time_string=total_time_string,
        wandb_project_name=wandb_project_name,
        wandb_sweep_id=wandb_sweep_id,
        wandb_sweep_name=wandb_sweep_name,
        logs_parent_directory=logs_parent_directory,
        sweep_worker_script_filepath=str(sweep_worker_script_filepath.absolute()),
        base_config_yaml_path=str(base_config_yaml_path.absolute()),
    ).strip()


def get_sweep_config_filename_from_sweep_config_name(sweep_config_key: str) -> str:
    # no spaces allowed
    cleaned_name = sweep_config_key.replace(" ", "_")
    return f"{cleaned_name}.txt"


def get_wandb_sweep_id(
    sweep_enclosing_dir: Path,
    sweep_config: Dict[str, Any],
    wandb_org_name: str,
    wandb_project_name: str,
) -> Tuple[str, int]:
    """Given the directory where we want to store artifacts for the sweep and its unique config name,
    get a wandb ID for it. Assumes that a project under the given org already exists.

    If there already exists a file that is named to track the settings of the sweep, we will read those
    settings from that file. Otherwise, we request a new sweep from wandb with the specified settings. Note
    that once a sweep is created, its settings CANNOT be edited. This means if we wish to add a new hyperparameter
    to the sweep, we must make a new sweep.
    """
    sweep_name = sweep_config["name"]
    sweep_info_file = (
        sweep_enclosing_dir
        / get_sweep_config_filename_from_sweep_config_name(sweep_name)
    )

    if sweep_info_file.exists() and sweep_info_file.is_file():
        # a sweep with this same name already exists, assuming a unique mapping from sweep file name and config
        sweep_id = sweep_info_file.read_text().strip()
        print(
            f"A wandb sweep ID already existed, using sweep id = {sweep_id}, wandb_project = {wandb_project_name}"
        )
    else:
        # create a new sweep with this configuration and project name
        sweep_id = wandb.sweep(sweep=sweep_config, project=wandb_project_name)

        print(
            f"A wandb sweep ID did NOT exist, created sweep id = {sweep_id}, wandb_project = {wandb_project_name}"
        )
        sweep_info_file.write_text(sweep_id)

    api = wandb.Api()
    sweep = api.sweep(f"{wandb_org_name}/{wandb_project_name}/sweeps/{sweep_id}")
    num_runs_final = sweep.expected_run_count

    if num_runs_final is None:
        warnings.warn(
            "Unable to fetch / determine the number of estimated runs for this sweep from wandb... defaulting to 10",
            UserWarning,
        )
        num_runs_final = 10

    return sweep_id, num_runs_final


def submit_sbatch_script(script_path: Path) -> str:
    """
    Submits the SLURM job script and returns the job ID.
    """
    result = subprocess.run(
        f"sbatch {script_path.absolute()}",
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output = result.stdout
    # The job ID is usually in the output string like: "Submitted batch job 123456"
    job_id = output.strip().split()[-1]
    return job_id


def check_job_status(job_id: str) -> str:
    """
    Checks the status of the SLURM job using its job ID.
    """
    result = subprocess.run(
        f"squeue -j {job_id} -o %T",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output = result.stdout.strip()
    if "RUNNING" in output:
        return "RUNNING"
    elif "PENDING" in output:
        return "PENDING"
    else:
        return "NONE"


if __name__ == "__main__":
    """Define a sweep configuration to run in wandb. Call an agent script to run a sweep within the sweep ID.

    Run this script with: 

        python scripts/training/sweep_starter.py [...args]


    For example:

        python scripts/training/sweep_starter.py --sweep_definition_yaml_path scripts/training/configs/example_sweep.yaml --wandb_project_name chronos_sweep --wandb_org_name megan-wei --conda_env_name chronos-doover  --base_config_yaml_path scripts/training/configs/07-05/chronos-t5-mini-fourier.yaml

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep_definition_yaml_path",
        type=str,
        required=True,
        help="The wandb sweep definition. See here for its structure: https://docs.wandb.ai/guides/sweeps/define-sweep-configuration",
    )
    parser.add_argument(
        "--base_config_yaml_path",
        type=str,
        required=True,
        help="A yaml file that defines the base training settings. Every parameter not defined / changed in the sweep is preserved.",
    )
    parser.add_argument("--wandb_project_name", type=str, required=True)
    parser.add_argument("--wandb_org_name", type=str, required=True)
    parser.add_argument("--conda_env_name", type=str, required=True)
    parser.add_argument("--partition", type=str, required=True)
    parser.add_argument(
        "--hardware", type=str, required=False, default="a6000|geforce3090"
    )
    parser.add_argument(
        "--total_time_string", type=str, required=False, default="24:00:00"
    )
    parser.add_argument("--wait_sequential", type=bool, required=False, default=False)
    args = parser.parse_args()

    wandb_org_name = args.wandb_org_name
    base_config_yaml_path = Path(args.base_config_yaml_path)
    wandb_project_name = args.wandb_project_name
    wandb_sweep_definition_yaml_path = args.sweep_definition_yaml_path
    conda_env_name = args.conda_env_name
    slurm_partition = args.partition
    hardware = args.hardware
    wait_sequential = args.wait_sequential
    total_time_string = args.total_time_string

    with open(wandb_sweep_definition_yaml_path, "r") as f:
        sweep_config = yaml.safe_load(f)
        # ignore this field, the file is the program that runs the agent is hardcoded in our
        # setup
        sweep_config.pop("program", "")

    # ensure all files written have unrestricted user permissions
    with use_770_permissions():
        # make the sweeps directory if it doesn't exist, otherwise just leave it alone
        wandb_sweep_name = sweep_config["name"]
        sweep_enclosing_dir = (
            SWEEP_ARTIFACT_FOLDER / wandb_project_name
        ) / wandb_sweep_name
        sweep_enclosing_dir.mkdir(parents=True, exist_ok=True)

        # get or create the wandb sweep ID
        sweep_id, num_runs_remaining = get_wandb_sweep_id(
            sweep_enclosing_dir, sweep_config, wandb_org_name, wandb_project_name
        )

        # create script file to run
        script_contents = get_slurm_job_script_for_workers(
            wandb_project_name=wandb_project_name,
            wandb_sweep_id=sweep_id,
            wandb_sweep_name=wandb_sweep_name,
            conda_env_name=conda_env_name,
            sweep_worker_script_filepath=SWEEP_WORKER_SCRIPT_FILEPATH,
            base_config_yaml_path=base_config_yaml_path,
            logs_parent_directory=os.path.join(os.getcwd(), "output/wandb_sweeps"),
            slurm_partition=slurm_partition,
            hardware=hardware,
            total_time_string=total_time_string,
        )

        # write the slurm jobs to a shell script, change permissions of that file
        sweep_invocation_script = (
            sweep_enclosing_dir / f"start_sweep_for_{wandb_project_name}.sh"
        )
        sweep_invocation_script.write_text(script_contents)
        subprocess.run(
            f"chmod u+x {sweep_invocation_script.absolute()}", shell=True
        )  # might not be necessary
        print(f"Saved sweep run invocation script to: {sweep_invocation_script}")

        # Enqueue this script to slurm scheduler n times
        for _ in range(num_runs_remaining):
            job_id = submit_sbatch_script(sweep_invocation_script)
            print(f"Submitted job with ID: {job_id}")

            time.sleep(30)

            # Check job status
            status = check_job_status(job_id)
            print(f"Job {job_id} status: {status}")
            sys.stdout.flush()

            # Optionally, wait until the job starts or gets queued
            if wait_sequential:
                print("status:", status)
                while status == "PENDING":
                    time.sleep(30)  # Wait for 10 seconds before checking again
                    status = check_job_status(job_id)
                    print(f"    Job {job_id} status: {status}")
                    sys.stdout.flush()

            # wait at least 10 seconds in between jobs being launched to make sure that no two jobs end up queued at the same time
            time.sleep(30)
