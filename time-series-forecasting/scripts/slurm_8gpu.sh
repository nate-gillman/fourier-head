#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p gpu-he --gres=gpu:8
#SBATCH --constraint=geforce3090|a6000|l40s
#SBATCH -N 1 # gives one node, makes sure cpu cores are on same node
#SBATCH -c 4 # num CPU cores
#SBATCH --mem=31G
#SBATCH -t 72:00:00
#SBATCH -e output/slurm_logs/%j.err
#SBATCH -o output/slurm_logs/%j.out
#SBATCH --mail-user=nate_gillman@brown.edu
#SBATCH --mail-type=ALL

# SET UP COMPUTING ENV
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

module load mesa
module load boost/1.80.0
module load patchelf
module load glew
module load cuda
module load ffmpeg

# Activate virtual environment
# Load anaconda module, and other modules
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate /gpfs/home/ngillman/.conda/envs/chronos-doover

# Move to correct working directory
HOME_DIR=/oscar/data/superlab/users/nates_stuff/fourier-head/time-series-forecasting
cd ${HOME_DIR}

# put the script execution statement here

# training
# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
torchrun --nproc-per-node=8 scripts/train/train.py --config scripts/train/configs/linear.yaml
# torchrun --nproc-per-node=8 scripts/train/train.py --config scripts/train/configs/fourier-64.yaml
# torchrun --nproc-per-node=8 scripts/train/train.py --config scripts/train/configs/fourier-128.yaml