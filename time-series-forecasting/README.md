
# Chronos experiments

## Environment setup

```bash
conda create -n chronos python=3.11
conda activate chronos

pip install "chronos[training] @ git+https://github.com/amazon-science/chronos-forecasting.git"
pip install datasets
pip install matplotlib
pip install opencv-python
pip install gputil
pip install wandb
```

NOTE: sometimes, changes to code need to be propagated by running the following:

```bash
pip uninstall chronos; pip install -e .
```

## Dataset preprocessing

Need to run the following script one time, on a machine that has at least 128 GB of CPU memory.
It takes a while to run; it has to download all the data, then convert it to the arrow format.

```bash
python scripts/data_prep/convert_to_arrow.py
```

## Training

We ran all of our training experiments on an 8xGPU cluster.

```bash
# Linear baseline experiment
torchrun --nproc-per-node=8 scripts/train/train.py --config scripts/train/configs/linear.yaml

# Fourier head experiments
torchrun --nproc-per-node=8 scripts/train/train.py --config scripts/train/configs/fourier-64.yaml
torchrun --nproc-per-node=8 scripts/train/train.py --config scripts/train/configs/fourier-128.yaml
torchrun --nproc-per-node=8 scripts/train/train.py --config scripts/train/configs/fourier-256.yaml
torchrun --nproc-per-node=8 scripts/train/train.py --config scripts/train/configs/fourier-550.yaml

# Fourier head ablations
torchrun --nproc-per-node=8 scripts/train/train.py --config scripts/train/configs/fourier-550_no-regularization.yaml
torchrun --nproc-per-node=8 scripts/train/train.py --config scripts/train/configs/fourier-550_uniform_binning.yaml
```

Each of these will output the checkpoints, logs, and other training artifacts into `output/run-i`.
Once the run finishes, I like to rename the last directory from `run-i` to `fourier-64` or whatever run it corresponds to, to stay organized :)


Note: if you'd rather so a scaled-down run on a single GPU, you can run it as follows:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py --config scripts/train/configs/fourier-64.yaml
```

## Evaluation

Compute MASE and WQL, save json to disk:

```bash
OUTPUT_CONFIG=output/fourier-64/fourier-64.yaml

CUDA_VISIBLE_DEVICES=0 python scripts/eval/compute_MASE_and_WQL.py --config $OUTPUT_CONFIG
```

Compute smoothness, save json to disk:

```bash
# part 1: compute+save a bunch of npy files corresponding to the logits
CUDA_VISIBLE_DEVICES=0 python scripts/eval/save_multinomials_from_checkpoint.py --config $OUTPUT_CONFIG

# part 2: compute all the smoothnesses and save them to disk in a json
python scripts/eval/compute_smoothness.py $OUTPUT_CONFIG
```