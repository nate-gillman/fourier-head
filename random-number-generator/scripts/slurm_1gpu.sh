#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --constraint=a6000|l40s|geforce3090
#SBATCH --exclude=gpu2112,gpu2114,gpu2116,gpu2108,gpu2106
#SBATCH -N 1 # gives one node, makes sure cpu cores are on same node
#SBATCH -c 1 # num CPU cores
#SBATCH --mem=31G
#SBATCH -t 12:00:00
#SBATCH -e output/slurm_logs/%j.err
#SBATCH -o output/slurm_logs/%j.out
#SBATCH --mail-user=nate_gillman@brown.edu
#SBATCH --mail-type=ALL

# Activate virtual environment
# Load anaconda module, and other modules
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
CONDA_ENV_DIR=/oscar/data/superlab/users/nates_stuff/fourier-head/random-number-generator/llama-fourier-head-env
conda activate $CONDA_ENV_DIR

# Move to correct working directory
HOME_DIR=/oscar/data/superlab/users/nates_stuff/fourier-head/random-number-generator
cd ${HOME_DIR}

# experiment script here...

num_epochs=16
# nums_freqs=(1 2 3)
# nums_freqs=(4 5 6)
# nums_freqs=(7 8 9)
nums_freqs=(10 11 12)
nums_in_context_samples_per_prompt=(0)
seeds=(42 43 44 45 46 47 48 49 50 51)
for num_freqs in "${nums_freqs[@]}"; do
    for num_in_context_samples_per_prompt in "${nums_in_context_samples_per_prompt[@]}"; do
        data_dir=data/$(printf "%02d" $num_in_context_samples_per_prompt)_in_context_samples
        for seed in "${seeds[@]}"; do
            sh scripts/experiment_LoRA.sh $num_in_context_samples_per_prompt $data_dir $num_epochs $num_freqs $seed
        done
    done
done