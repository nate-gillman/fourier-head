
# Decision transformer experiments

## Setup

### Step 1: set up conda environment

```bash
conda create -n dt-atari python=3.7.9
conda activate dt-atari

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

# confirm that torch has been installed correctly
python -c 'import torch; print(torch.cuda.is_available()); a = torch.zeros(5); a.to("cuda:0"); print(a)'
python -c 'import torch; '

pip install tqdm
pip install atari-py
pip install opencv-python
pip install blosc

git clone https://github.com/google/dopamine
pip install absl-py
pip install gin-config
pip install tensorflow==1.15
pip install protobuf==3.20.3

python -m atari_py.import_roms Atari-2600-VCS-ROM-Collection/ROMS/
```

### Step 2: download dataset

```bash
mkdir dataset

pip install gsutil

# games with the same 18-dimensional action space
games=(
    "BankHeist"       
    "BattleZone"        # doesn't work
    "Boxing"            
    "Centipede" 
    "DoubleDunk"    
    "FishingDerby" 
    "Frostbite" 
    "Gravitar" 
    "Hero" 
    "IceHockey"
    "Jamesbond" 
    "Kangaroo" 
    "Krull"             # doesn't work
    "MontezumaRevenge"
    "PrivateEye" 
    "Riverraid" 
    "RoadRunner"
    "Robotank"
    "Seaquest"          # works
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
```

## Running the experiments

### Training

Linear:

```bash
GAME_NAME=BattleZone
sh scripts/train/exps_linear.sh $GAME_NAME
```


Fourier:

```bash
for fourier_frequencies in 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32
do
    sh scripts/train/exps_fourier.sh $GAME_NAME $fourier_frequencies
done
```

### Evaluation

Once the linear and Fourier models finish training, we can run this script which will compute the smoothness values for the saved multinomials.

```bash
python scripts/eval/compute_smoothness_dict.py $GAME_NAME
```

Then, we can graph the metrics (normalized returns, smoothness) for all of our frequencies.

```bash
python scripts/eval/atari_graph_varying_freqs.py $GAME_NAME
```

Next, we can graph the learned PMFs side by side.
Right now, it'll generate a graph for every saved multinomial in the test split for Fourier-8 and Linear for Seaquest.

```bash
python scripts/eval/atari_graph_PMFs.py
```
