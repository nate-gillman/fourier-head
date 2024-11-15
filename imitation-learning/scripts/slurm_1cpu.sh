#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p batch
#SBATCH -N 1 # gives one node, makes sure cpu cores are on same node
#SBATCH -c 8 # num CPU cores
#SBATCH --mem=64G
#SBATCH -t 48:00:00
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
conda activate /gpfs/home/ngillman/.conda/envs/dt-atari

# Move to correct working directory
HOME_DIR=/oscar/data/superlab/users/nates_stuff/fourier-head/imitation-learning
cd ${HOME_DIR}

# put python script here
games=(
    "Jamesbond" 
    "Kangaroo" 
    "Krull" 
    "MontezumaRevenge"
    "PrivateEye" 
    "Riverraid" 
    "RoadRunner"
    "Robotank"
    "StarGunner" 
    "Tennis" 
    "Venture" 
    "Zaxxon"
)

calculate_progress() {
    local source=$1
    local dest=$2
    local game=$3
    
    if [ "$game" = "Seaquest" ]; then
        local total_size=$(gsutil du -s "$source/{1,2}" | awk '{sum+=$1} END {print sum}')
    else
        local total_size=$(gsutil du -s "$source/1" | awk '{print $1}')
    fi
    
    local current_size=0
    while [ -d "$dest" ]; do
        current_size=$(du -sb "$dest" | awk '{print $1}')
        local percent=$((current_size * 100 / total_size))
        echo -ne "\rProgress: $percent%"
        sleep 1
    done
    echo
}

for game in "${games[@]}"; do
    mkdir -p "dataset/$game"
    if [ "$game" = "Seaquest" ]; then
        echo "Downloading $game subdirs 1 and 2..."
        calculate_progress "gs://atari-replay-datasets/dqn/$game" "dataset/$game" "$game" &
        gsutil -m cp -R "gs://atari-replay-datasets/dqn/$game/{1,2}" "dataset/$game/"
    else
        echo "Downloading $game subdir 1..."
        calculate_progress "gs://atari-replay-datasets/dqn/$game" "dataset/$game" "$game" &
        gsutil -m cp -R "gs://atari-replay-datasets/dqn/$game/1" "dataset/$game/"
    fi
    kill $! 2>/dev/null
done
