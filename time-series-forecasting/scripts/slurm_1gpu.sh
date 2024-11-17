#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --constraint=a6000|geforce3090|l40s
#SBATCH --exclude=gpu1506,gpu2108,gpu2114,gpu2115,gpu2116
#SBATCH -N 1 # gives one node, makes sure cpu cores are on same node
#SBATCH -c 1 # num CPU cores
#SBATCH --mem=31G
#SBATCH -t 2:00:00
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
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-0/fourier-256-size-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-1/linear-size-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-2/fourier-256-size-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-3/linear-size-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-4/fourier-256-size-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-5/linear-size-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-6/fourier-256-size-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-7/linear-size-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-8/fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-9/linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-10/fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-11/linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-12/fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-13/linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-14/fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-15/linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-16/fourier-256-size-100000.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-17/linear-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-18/fourier-256-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-19/linear-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-20/fourier-256-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-21/linear-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-22/fourier-256-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-23/linear-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-24/fourier-256-size-1000000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-25/linear-size-1000000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-26/fourier-256-size-1000000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-27/linear-size-1000000.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-28/fourier-256-size-1000000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-29/linear-size-1000000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-30/fourier-256-size-1000000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-31/linear-size-1000000.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-34/fourier-256-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-35/linear-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-36/fourier-256-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-37/linear-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-38/fourier-256-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-39/linear-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-40/fourier-256-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-41/linear-size-100000.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-42/fourier-256-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-43/linear-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-44/fourier-256-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-45/linear-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-46/fourier-256-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-47/linear-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-48/fourier-256-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-49/linear-size-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-33/linear-size-10000000.yaml




# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-50/fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-51/linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-52/fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-53/linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-54/fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-55/linear-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-56/fourier-256-size-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-57/linear-size-10000.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-58/fourier-256-size-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-59/linear-size-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-60/fourier-256-size-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-61/linear-size-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-62/fourier-256-size-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-63/linear-size-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-64/fourier-256-size-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-65/linear-size-1000.yaml

CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config output/run-32/fourier-256-size-10000000.yaml




# OUTPUT_CONFIG=output/linear/linear.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/save_multinomials_from_checkpoint.py --config $OUTPUT_CONFIG

# OUTPUT_CONFIG=output/fourier-550/fourier-550.yaml
# python scripts/eval/compute_smoothness.py $OUTPUT_CONFIG

# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-15-ablations-dataset-size-proportional/fourier-128-tsmixup-1000-kernelsynth-100.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-15-ablations-dataset-size-proportional/fourier-128-tsmixup-10000-kernelsynth-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-15-ablations-dataset-size-proportional/fourier-128-tsmixup-100000-kernelsynth-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-15-ablations-dataset-size-proportional/fourier-128-tsmixup-1000000-kernelsynth-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-15-ablations-dataset-size-proportional/fourier-128-tsmixup-10000000-kernelsynth-1000000.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-15-ablations-dataset-size-proportional/linear-tsmixup-1000-kernelsynth-100.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-15-ablations-dataset-size-proportional/linear-tsmixup-10000-kernelsynth-1000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-15-ablations-dataset-size-proportional/linear-tsmixup-100000-kernelsynth-10000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-15-ablations-dataset-size-proportional/linear-tsmixup-1000000-kernelsynth-100000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-15-ablations-dataset-size-proportional/linear-tsmixup-10000000-kernelsynth-1000000.yaml

# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-16-ablation-dataset-size/fourier-256-size-10000000.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-16-ablation-dataset-size/linear-size-10000000.yaml

# size=1000
# seed=42

# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-17-pt2-ablation-dataset-size/seed-$seed/fourier-256-size-$size.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-17-pt2-ablation-dataset-size/seed-$seed/linear-size-$size.yaml