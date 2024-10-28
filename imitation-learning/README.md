
# Fourier head experiments

## Setup

### Step 1: set up conda evn

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
gsutil -m cp -R gs://atari-replay-datasets/dqn/Seaquest dataset

```

## Running the experiments

### Training

Linear:

```bash
sh scripts/train/exps_linear.sh
```


Fourier:

```bash
for fourier_frequencies in 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32
do
    sh scripts/train/exps_fourier.sh $fourier_frequencies
done
```

### Evaluation

Once the linear and Fourier models finish training, we can run this script which will compute the smoothness values for the saved multinomials.

```bash
python scripts/eval/compute_smoothness_dict.py
```

Then, we can graph the metrics (normalized returns, smoothness) for all of our frequencies.

```bash
python scripts/eval/atari_graph_varying_freqs.py
```

Next, we can graph the learned PMFs side by side.
Right now, the paths are hard-coded so that you can generate the same graphs in our paper, but you can redirect them to the multinomials from the test split that were generated.

```bash
python scripts/eval/atari_graph_PMFs.py
```
