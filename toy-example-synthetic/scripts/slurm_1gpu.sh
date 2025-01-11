#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p gpu-he --gres=gpu:1
#SBATCH --constraint=a6000|geforce3090|l40s
#SBATCH --exclude=gpu1506,gpu2108,gpu2114,gpu2115,gpu2116
#SBATCH -N 1 # gives one node, makes sure cpu cores are on same node
#SBATCH -c 1 # num CPU cores
#SBATCH --mem=31G
#SBATCH -t 24:00:00
#SBATCH -e output/slurm_logs/%j.err
#SBATCH -o output/slurm_logs/%j.out
#SBATCH --mail-user=daksh_aggarwal@brown.edu
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
conda activate chronos-doover

# Move to correct working directory
HOME_DIR=/users/daggarw5/scratch/fourier-head/toy-example-synthetic/scripts
cd ${HOME_DIR}

# linear head, running all datasets
#sh ./run_exps_linear.sh gaussian
# sh ./run_exps_linear.sh gmm
# sh ./run_exps_linear.sh gmm2

# fourier head with no regularization
# sh ./run_exps_fourier_no_reg.sh gaussian
# sh ./run_exps_fourier_no_reg.sh gmm
# sh ./run_exps_fourier_no_reg.sh gmm2

# experiments with regularization
# sh ./run_exps_fourier_reg.sh gaussian
# sh ./run_exps_fourier_reg.sh gmm
# sh ./run_exps_fourier_reg.sh gmm2

# sh ./run_exps_fourier-mle_no_reg.sh gaussian
# sh ./run_exps_fourier-mle_no_reg.sh gmm2
# sh ./run_exps_fourier-mle_no_reg.sh beta

sh ./run_exps_fourier-mle_reg.sh gaussian
sh ./run_exps_fourier-mle_reg.sh gmm2
sh ./run_exps_fourier-mle_reg.sh beta