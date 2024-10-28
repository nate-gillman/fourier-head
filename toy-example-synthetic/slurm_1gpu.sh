#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --constraint=a6000|geforce3090|l40s
#SBATCH --exclude=gpu1506,gpu2108,gpu2114,gpu2115,gpu2116
#SBATCH -N 1 # gives one node, makes sure cpu cores are on same node
#SBATCH -c 1 # num CPU cores
#SBATCH --mem=31G
#SBATCH -t 24:00:00
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
HOME_DIR=/oscar/data/superlab/users/nates_stuff/fourier-head/toy-example-synthetic
cd ${HOME_DIR}

# linear head, running all datasets
# sh ./run_exps_linear.sh gaussian
# sh ./run_exps_linear.sh gmm
# sh ./run_exps_linear.sh gmm2

# fourier head with no regularization
# sh ./run_exps_fourier_no_reg.sh gaussian
# sh ./run_exps_fourier_no_reg.sh gmm
# sh ./run_exps_fourier_no_reg.sh gmm2

# experiments with regularization
# sh ./run_exps_fourier_reg.sh gaussian
# sh ./run_exps_fourier_reg.sh gmm
sh ./run_exps_fourier_reg.sh gmm2
