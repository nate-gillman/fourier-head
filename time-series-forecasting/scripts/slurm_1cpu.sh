#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p batch
#SBATCH -N 1 # gives one node, makes sure cpu cores are on same node
#SBATCH -c 1 # num CPU cores
#SBATCH --mem=25G
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
HOME_DIR=/oscar/data/superlab/users/nates_stuff/fourier-head/time-series-forecasting
cd ${HOME_DIR}

seed=312 # [42, 123, 231, 312]

# put python script here
# python scripts/data_prep/convert_to_arrow_subset.py 20000 2000 $seed
# python scripts/data_prep/convert_to_arrow_subset.py 40000 4000 $seed
# python scripts/data_prep/convert_to_arrow_subset.py 60000 6000 $seed
# python scripts/data_prep/convert_to_arrow_subset.py 80000 8000 $seed
# python scripts/data_prep/convert_to_arrow_subset.py 100000 10000 $seed

python scripts/data_prep/convert_to_arrow_subset.py 120000 12000 $seed
python scripts/data_prep/convert_to_arrow_subset.py 140000 14000 $seed
python scripts/data_prep/convert_to_arrow_subset.py 160000 16000 $seed
python scripts/data_prep/convert_to_arrow_subset.py 180000 18000 $seed
python scripts/data_prep/convert_to_arrow_subset.py 200000 20000 $seed

