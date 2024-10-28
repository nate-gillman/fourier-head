#!/bin/bash
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2106,gpu2108,gpu2112,gpu2114,gpu2115,gpu2116
#SBATCH -N 1
#SBATCH --mem=31G
#SBATCH -t 24:00:00
#SBATCH -J fourier_head_audio_sweep_starter
#SBATCH -e /oscar/scratch/%u/fourier_head_logs/starter-%j.err
#SBATCH -o /oscar/scratch/%u/fourier_head_logs/starter-%j.out
set -e

# cd to the location of this file on disk
cd "$(dirname "$0")"

# Load anaconda module, and other modules
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh

# Activate virtual environment, see toy_example_audio/environment.yaml
conda activate fourier_head_audio

# Run the sweep starter script
python sweep/sweep_starter.py \
    --sweep_definition_yaml sweep_configurations/fourier_head_audio/sweep.yaml \
    --wandb_project_name my_wandb_project_name \
    --wandb_org_name my_org_name \
    --conda_env_name fourier_head_audio \
    --base_config_yaml_path sweep_configurations/fourier_head_audio/base.yaml \
    --partition 3090-gcondo