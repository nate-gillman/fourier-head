#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --constraint=a6000|geforce3090|l40s
#SBATCH --exclude=gpu1506,gpu2108,gpu2114,gpu2115,gpu2116
#SBATCH -N 1 # gives one node, makes sure cpu cores are on same node
#SBATCH -c 1 # num CPU cores
#SBATCH --mem=31G
#SBATCH -t 12:00:00
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






# OUTPUT_CONFIG=output/linear/linear.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/eval/save_multinomials_from_checkpoint.py --config $OUTPUT_CONFIG

# OUTPUT_CONFIG=output/fourier-550/fourier-550.yaml
# python scripts/eval/compute_smoothness.py $OUTPUT_CONFIG

# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-15-ablations-dataset-size-proportional/fourier-128-tsmixup-1000-kernelsynth-100.yaml

# model_size=03-base # [00-tiny, 01-mini, 02-small, 03-base]
# dataset_size=10000
# seed=42

# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-26-pt1-ablation-model-size/seed-$seed/$model_size-linear-size-$dataset_size.yaml
# CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/11-26-pt1-ablation-model-size/seed-$seed/$model_size-fourier-128-size-$dataset_size.yaml



# # SWEEP #1: MODEL SIZE

# model_sizes=("00-tiny" "01-mini" "02-small" "03-base")
# dataset_size=10000
# seed=312 # [42, 123, 231, 312]

# # Loop over model sizes
# for model_size in "${model_sizes[@]}"; do
#     echo "Running training for model: $model_size"
#     echo "Running training for dataset_size: $dataset_size"
#     echo "Running training for seed: $seed"
    
#     # Run linear version
#     echo "Training linear architecture..."
#     linear_config="scripts/train/configs/11-26-pt3-ablation-model-size/seed-$seed/$model_size-linear-size-$dataset_size.yaml"
#     CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config "$linear_config"

#     # add a small delay between runs
#     sleep 10
    
#     # Run fourier version
#     echo "Training fourier architecture..."
#     fourier_config="scripts/train/configs/11-26-pt3-ablation-model-size/seed-$seed/$model_size-fourier-128-size-$dataset_size.yaml"
#     CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config "$fourier_config"
    
#     # add a small delay between runs
#     sleep 10
# done


# # SWEEP #2: DATASET SIZE

seed=312 # [42, 123, 231, 312]

# dataset_sizes=(20000 40000 60000 80000 100000 120000)
dataset_sizes=(140000 160000 180000 200000)
model_types=("fourier-256" "linear")

# Outer loop over dataset sizes
for dataset_size in "${dataset_sizes[@]}"; do
    # Inner loop over model types
    for model_type in "${model_types[@]}"; do

        echo "Running training for model: $model_type"
        echo "Running training for dataset_size: $dataset_size"
        echo "Running training for seed: $seed"

        config="scripts/train/configs/11-26-pt4-ablation-dataset-size-smaller/seed-$seed/$model_type-size-$dataset_size.yaml"
        CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config "$config"

        # Add a small delay between runs
        sleep 10
    done
done