#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --constraint=a6000|geforce3090|l40s
#SBATCH --exclude=gpu1506,gpu2108,gpu2114,gpu2115,gpu2116
#SBATCH -N 1 # gives one node, makes sure cpu cores are on same node
#SBATCH -c 1 # num CPU cores
#SBATCH --mem=31G
#SBATCH -t 4:00:00
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

# EVAL SET 1: model size sweep

# put the script execution statement here
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-0/00-tiny-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-1/00-tiny-fourier-128-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-2/01-mini-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-6/01-mini-fourier-128-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-3/02-small-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-10/02-small-fourier-128-size-10000.yaml

# base models need 4ish hours? and an a6000
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-4/03-base-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-5/03-base-fourier-128-size-10000.yaml


# EVAL SET 2: dataset size sweep

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-20/fourier-256-size-20000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-30/fourier-256-size-40000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-40/fourier-256-size-60000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-21/fourier-256-size-80000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-32/fourier-256-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-26/fourier-256-size-120000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-38/fourier-256-size-140000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-44/fourier-256-size-160000.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config 
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config 

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-25/linear-size-20000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-35/linear-size-40000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-43/linear-size-60000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-27/linear-size-80000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-39/linear-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-33/linear-size-120000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-42/linear-size-140000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-49/linear-size-160000.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config 
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config 



# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-0/00-tiny-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-1/00-tiny-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-10/01-mini-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-11/01-mini-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-12/01-mini-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-13/01-mini-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-14/01-mini-fourier-128-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-15/01-mini-fourier-128-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-16/01-mini-fourier-128-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-17/01-mini-fourier-128-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-18/02-small-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-19/02-small-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-2/00-tiny-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-20/02-small-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-21/02-small-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-22/02-small-fourier-128-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-23/02-small-fourier-128-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-24/02-small-fourier-128-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-25/02-small-fourier-128-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-5/00-tiny-fourier-128-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-6/00-tiny-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-7/00-tiny-fourier-128-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-8/00-tiny-fourier-128-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-9/00-tiny-fourier-128-size-10000.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-26/03-base-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-27/03-base-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-28/03-base-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt3-ablation-model-size/run-29/03-base-linear-size-10000.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-0/fourier-256-size-20000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-1/fourier-256-size-140000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-10/linear-size-20000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-11/linear-size-20000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-12/fourier-256-size-40000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-13/fourier-256-size-40000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-14/fourier-256-size-40000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-15/fourier-256-size-40000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-16/linear-size-140000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-17/linear-size-140000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-18/linear-size-140000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-19/linear-size-140000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-2/fourier-256-size-20000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-20/linear-size-40000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-21/linear-size-40000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-22/linear-size-40000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-23/linear-size-40000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-24/fourier-256-size-160000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-25/fourier-256-size-60000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-26/fourier-256-size-160000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-27/fourier-256-size-60000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-28/fourier-256-size-160000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-29/fourier-256-size-60000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-3/fourier-256-size-140000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-30/fourier-256-size-160000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-31/fourier-256-size-60000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-32/linear-size-60000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-33/linear-size-60000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-34/linear-size-60000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-35/linear-size-60000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-36/fourier-256-size-80000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-37/fourier-256-size-80000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-38/linear-size-160000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-39/fourier-256-size-80000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-4/fourier-256-size-20000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-40/linear-size-160000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-41/fourier-256-size-80000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-42/linear-size-160000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-43/linear-size-160000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-44/linear-size-80000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-45/linear-size-80000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-46/linear-size-80000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-47/fourier-256-size-180000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-48/linear-size-80000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-49/fourier-256-size-180000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-5/fourier-256-size-140000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-50/fourier-256-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-51/fourier-256-size-180000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-52/fourier-256-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-53/fourier-256-size-180000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-54/fourier-256-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-55/fourier-256-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-56/linear-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-57/linear-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-58/linear-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-59/linear-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-6/fourier-256-size-20000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-60/fourier-256-size-120000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-61/linear-size-180000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-62/fourier-256-size-120000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-63/linear-size-180000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-64/fourier-256-size-120000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-65/linear-size-180000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-66/fourier-256-size-120000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-67/linear-size-180000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-68/fourier-256-size-200000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-69/fourier-256-size-200000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-7/fourier-256-size-140000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-70/fourier-256-size-200000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-71/fourier-256-size-200000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-72/linear-size-120000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-73/linear-size-120000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-74/linear-size-120000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-75/linear-size-120000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-76/linear-size-200000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-77/linear-size-200000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-78/linear-size-200000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-79/linear-size-200000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-8/linear-size-20000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-26-pt4-ablation-dataset-size-smaller/run-9/linear-size-20000.yaml


# OUTPUT_CONFIG=output/linear/linear.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/save_multinomials_from_checkpoint.py --config $OUTPUT_CONFIG

# OUTPUT_CONFIG=output/fourier-550/fourier-550.yaml
# python scripts/eval/compute_smoothness.py $OUTPUT_CONFIG


# model_size=03-base # [00-tiny, 01-mini, 02-small, 03-base]
# dataset_size=10000
# seed=42

# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-26-pt1-ablation-model-size/seed-$seed/$model_size-linear-size-$dataset_size.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-26-pt1-ablation-model-size/seed-$seed/$model_size-fourier-128-size-$dataset_size.yaml


# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt1-ablation-model-size/run-0/02-small-fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt1-ablation-model-size/run-1/02-small-fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt1-ablation-model-size/run-2/02-small-fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt1-ablation-model-size/run-3/02-small-fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt1-ablation-model-size/run-4/02-small-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt1-ablation-model-size/run-5/02-small-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt1-ablation-model-size/run-6/02-small-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt1-ablation-model-size/run-7/02-small-linear-size-10000.yaml


# # SWEEP #1. VERSION 1: MODEL SIZE

# model_sizes=("00-tiny" "01-mini")
# dataset_size=10000
# seed=312 # [42, 123, 231, 312]

# # # Loop over model sizes
# for model_size in "${model_sizes[@]}"; do
#     echo "Running training for model: $model_size"
#     echo "Running training for dataset_size: $dataset_size"
#     echo "Running training for seed: $seed"
    
#     # Run linear version
#     echo "Training linear architecture..."
#     linear_config="scripts/train/configs/11-27-pt1-ablation-model-size/seed-$seed/$model_size-linear-size-$dataset_size.yaml"
#     CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config "$linear_config"

#     # add a small delay between runs
#     sleep 10
    
#     # Run fourier version
#     echo "Training fourier architecture..."
#     fourier_config="scripts/train/configs/11-27-pt1-ablation-model-size/seed-$seed/$model_size-fourier-128-size-$dataset_size.yaml"
#     CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config "$fourier_config"
    
#     # add a small delay between runs
#     sleep 10
# done

# SWEEP #1. VERSION 2: MODEL SIZE

# model_sizes=("00" "01")
# model_sizes=("02" "03")
# model_sizes=("04" "05")
# model_sizes=("06" "07")
# model_sizes=("08" "09")
# model_sizes=("10" "11")
# model_sizes=("12" "13")

# dataset_size=10000
# seed=42 # [42, 123, 231, 312]

# # # Loop over model sizes
# for model_size in "${model_sizes[@]}"; do
#     echo "Running training for model: $model_size"
#     echo "Running training for dataset_size: $dataset_size"
#     echo "Running training for seed: $seed"
    
#     # Run fourier version
#     echo "Training fourier architecture..."
#     fourier_config="scripts/train/configs/11-27-pt3-ablation-model-size/seed-$seed/fourier-$model_size-256-size-$dataset_size.yaml"
#     CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config "$fourier_config"

#     # add a small delay between runs
#     sleep 10
    
#     # Run linear version
#     echo "Training linear architecture..."
#     linear_config="scripts/train/configs/11-27-pt3-ablation-model-size/seed-$seed/linear-$model_size-size-$dataset_size.yaml"
#     CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config "$linear_config"
    
#     # add a small delay between runs
#     sleep 10
# done

# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt1-ablation-model-size/seed-42/02-small-fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt1-ablation-model-size/seed-42/02-small-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt1-ablation-model-size/seed-123/02-small-fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt1-ablation-model-size/seed-123/02-small-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt1-ablation-model-size/seed-231/02-small-fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt1-ablation-model-size/seed-231/02-small-linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt1-ablation-model-size/seed-312/02-small-fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt1-ablation-model-size/seed-312/02-small-linear-size-10000.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt1-ablation-model-size/seed-42/00-tiny-fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt1-ablation-model-size/seed-123/00-tiny-fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt1-ablation-model-size/seed-231/00-tiny-fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt1-ablation-model-size/seed-312/00-tiny-fourier-256-size-10000.yaml

# # SWEEP #2: DATASET SIZE

# seed=312 # [42, 123, 231, 312]

# dataset_sizes=(10000 20000 40000)
# dataset_sizes=(60000 80000 100000)
# dataset_sizes=(120000 140000 160000)
# dataset_sizes=(180000 200000)
# model_types=("fourier-256" "linear")

# # Outer loop over dataset sizes
# for dataset_size in "${dataset_sizes[@]}"; do
#     # Inner loop over model types
#     for model_type in "${model_types[@]}"; do

#         echo "Running training for model: $model_type"
#         echo "Running training for dataset_size: $dataset_size"
#         echo "Running training for seed: $seed"

#         config="scripts/train/configs/11-27-pt2-ablation-dataset-size-smaller/seed-$seed/$model_type-size-$dataset_size.yaml"
#         CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config "$config"

#         # Add a small delay between runs
#         sleep 10
#     done
# done

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-0/fourier-00-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-1/fourier-02-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-10/fourier-01-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-11/fourier-03-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-12/linear-06-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-13/fourier-05-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-14/linear-08-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-15/linear-10-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-16/fourier-07-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-17/fourier-09-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-18/linear-12-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-19/linear-01-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-2/fourier-04-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-20/fourier-11-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-21/linear-03-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-22/linear-05-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-23/fourier-13-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-24/linear-07-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-25/linear-09-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-26/linear-11-size-10000.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-27/linear-13-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-3/fourier-06-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-4/fourier-08-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-5/fourier-10-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-6/fourier-12-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-7/linear-00-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-8/linear-02-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt3-ablation-model-size/run-9/linear-04-size-10000.yaml


# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt4-ablation-model-size/seed-42/fourier-0.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt4-ablation-model-size/seed-42/fourier-1.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt4-ablation-model-size/seed-42/fourier-2.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt4-ablation-model-size/seed-42/fourier-3.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt4-ablation-model-size/seed-42/linear-0.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt4-ablation-model-size/seed-42/linear-1.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt4-ablation-model-size/seed-42/linear-2.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-27-pt4-ablation-model-size/seed-42/linear-3.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-10/fourier-3.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-3/fourier-3.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-8/fourier-3.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-9/fourier-3.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-0/fourier-0.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-1/fourier-1.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-11/fourier-0.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-12/fourier-0.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-13/fourier-0.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-17/linear-0.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-18/linear-0.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-19/linear-0.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-2/fourier-2.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-20/linear-2.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-21/linear-2.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-22/linear-2.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-23/fourier-2.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-24/fourier-2.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-25/fourier-2.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-27/linear-1.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-28/linear-3.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-29/linear-3.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-30/linear-3.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-31/fourier-1.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-32/fourier-1.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-33/fourier-1.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-34/linear-1.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-35/linear-1.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-4/linear-0.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-5/linear-1.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-6/linear-2.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-27-pt4-ablation-model-size/run-7/linear-3.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-40/fourier-0.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-41/fourier-0.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-42/fourier-0.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-43/fourier-0.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-36/fourier-2.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-37/fourier-2.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-38/fourier-2.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-39/fourier-2.yaml

# sleep 0.01h
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-29-pt1-ablation-model-size/seed-42/fourier-0.yaml


# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config output/11-29-pt1-ablation-model-size/run-60/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config output/11-29-pt1-ablation-model-size/run-61/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config output/11-29-pt1-ablation-model-size/run-62/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config output/11-29-pt1-ablation-model-size/run-63/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config output/11-29-pt1-ablation-model-size/run-64/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config output/11-29-pt1-ablation-model-size/run-65/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config output/11-29-pt1-ablation-model-size/run-66/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config output/11-29-pt1-ablation-model-size/run-67/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config output/11-29-pt1-ablation-model-size/run-68/fourier-0.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-60/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-61/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-62/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-63/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-64/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-65/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-66/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-67/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-68/fourier-0.yaml

sleep 0.15h
CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-29-pt1-ablation-model-size/seed-42/linear-2.yaml

# sleep 2h
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-48/fourier-2-192.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-49/fourier-2-192.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-50/fourier-2-192.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config 
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config 
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config 
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config 


# sleep 1h

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-52/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-53/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-54/fourier-2-352.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-55/fourier-2-352.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-56/fourier-2-320.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-57/fourier-2-320.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-58/fourier-2-320.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/11-29-pt1-ablation-model-size/run-59/fourier-2-320.yaml

