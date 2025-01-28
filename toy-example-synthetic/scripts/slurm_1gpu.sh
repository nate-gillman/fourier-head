#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --constraint=a6000|geforce3090|l40s
#SBATCH --exclude=gpu1506,gpu2108,gpu2114,gpu2115,gpu2116
#SBATCH -N 1 # gives one node, makes sure cpu cores are on same node
#SBATCH -c 1 # num CPU cores
#SBATCH --mem=31G
#SBATCH -t 6:00:00
#SBATCH -e output/slurm_logs/%j.err
#SBATCH -o output/slurm_logs/%j.out
#SBATCH --mail-user=nate_gillman@brown.edu
#SBATCH --mail-type=ALL

# SET UP COMPUTING ENV
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Load anaconda module, activate virtual environment
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate toy-example-synthetic

# Move to correct working directory
HOME_DIR=/oscar/data/superlab/users/nates_stuff/fourier-head/toy-example-synthetic
cd ${HOME_DIR}

# linear classification head
# sh scripts/run_exps_linear.sh gaussian
# sh scripts/run_exps_linear.sh gmm2
# sh scripts/run_exps_linear.sh beta

# # gaussian mixture model head
# sh scripts/run_exps_gmm.sh gaussian
# sh scripts/run_exps_gmm.sh gmm2
# sh scripts/run_exps_gmm.sh beta

# # fourier head (with no regularization)
# sh scripts/run_exps_fourier_no_reg.sh gaussian
# sh scripts/run_exps_fourier_no_reg.sh gmm2
# sh scripts/run_exps_fourier_no_reg.sh beta

# # fourier head (with regularization)
# sh scripts/run_exps_fourier_reg.sh gaussian
# sh scripts/run_exps_fourier_reg.sh gmm2
# sh scripts/run_exps_fourier_reg.sh beta

# # linear regression head (pointwise estimate)
# sh scripts/run_exps_linear_regression.sh gaussian
# sh scripts/run_exps_linear_regression.sh gmm2
# sh scripts/run_exps_linear_regression.sh beta

# # Fourier-MLE head (with no regularization)
# sh scripts/run_exps_fourier-mle_no_reg.sh gaussian
# sh scripts/run_exps_fourier-mle_no_reg.sh gmm2
# sh scripts/run_exps_fourier-mle_no_reg.sh beta

# # Fourier-MLE head (with regularization)
# sh scripts/run_exps_fourier-mle_reg.sh gaussian
# sh scripts/run_exps_fourier-mle_reg.sh gmm2
sh scripts/run_exps_fourier-mle_reg.sh beta

# # GMM-MLE head
# sh scripts/run_exps_gmm-mle.sh gaussian
# sh scripts/run_exps_gmm-mle.sh gmm2
# sh scripts/run_exps_gmm-mle.sh beta


