#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p batch
#SBATCH -N 1 # gives one node, makes sure cpu cores are on same node
#SBATCH -c 8 # num CPU cores
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -e output/slurm_logs/%j.err
#SBATCH -o output/slurm_logs/%j.out
#SBATCH --mail-user=nate_gillman@brown.edu
#SBATCH --mail-type=ALL

module load mesa
module load boost/1.80.0
module load patchelf
module load glew
module load ffmpeg

# Activate virtual environment
# Load anaconda module, and other modules
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate /gpfs/home/ngillman/.conda/envs/chronos-doover

# Move to correct working directory
HOME_DIR=/oscar/data/superlab/users/nates_stuff/fourier-head/imitation-learning
cd ${HOME_DIR}

# python scripts/eval/compute_smoothness_dict.py BankHeist
# python scripts/eval/compute_smoothness_dict.py DoubleDunk
# python scripts/eval/compute_smoothness_dict.py Seaquest
