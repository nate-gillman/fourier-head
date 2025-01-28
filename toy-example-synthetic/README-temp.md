# 🎯 Toy Experiment: Learning Known Conditional Distributions

This repository contains code for experimenting with different approaches to learning known conditional distributions. The experiments compare various model architectures including linear heads, Fourier features, and Gaussian mixture models across multiple synthetic datasets.

## 🚀 Quick Start

### Environment Setup

Create and activate a fresh conda environment:

```bash
conda create -n toy-example-synthetic python=3.11
conda activate toy-example-synthetic
```

Install PyTorch with CUDA support:
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```

Verify PyTorch installation:
```python
python -c 'import torch; print(torch.cuda.is_available()); a = torch.zeros(5); a = a.to("cuda:0"); print(a)'
```

Install additional dependencies:
```bash
conda install scikit-learn tqdm pandas matplotlib
pip install wandb
```

## 🧪 Running Experiments

The experiments can be run on three synthetic datasets:
- `gaussian`
- `gmm2`
- `beta`

### Cross Entropy Training Examples

#### Linear Head
```bash
python scripts/toy_synthetic.py --head "linear" --n_freqs 0 --dataset "gmm2"
```

#### Fourier Head
Without regularization:
```bash
python scripts/toy_synthetic.py --head "fourier" --n_freqs 12 --gamma 0.0 --dataset "gmm2" 
```

With regularization:
```bash
python scripts/toy_synthetic.py --head "fourier" --n_freqs 12 --gamma 1e-6 --dataset "gmm2" 
```

> 💡 Add the `--wandb` flag to log experiments to Weights & Biases

### Batch Experiment Scripts

#### Cross Entropy Training
```bash
# Linear Classification Head
sh scripts/run_exps_linear.sh [dataset]

# Gaussian Mixture Model Head
sh scripts/run_exps_gmm.sh [dataset]

# Fourier Head (No Regularization)
sh scripts/run_exps_fourier_no_reg.sh [dataset]

# Fourier Head (With Regularization)
sh scripts/run_exps_fourier_reg.sh [dataset]

# Linear Regression Head
sh scripts/run_exps_linear_regression.sh [dataset]
```

#### MLE Training
```bash
# Fourier-MLE Head (No Regularization)
sh scripts/run_exps_fourier-mle_no_reg.sh [dataset]

# Fourier-MLE Head (With Regularization)
sh scripts/run_exps_fourier-mle_reg.sh [dataset]

# GMM-MLE Head
sh scripts/run_exps_gmm-mle.sh [dataset]
```

> Note: Each script typically takes less than 24 hours to complete on a GeForce RTX 3090 GPU.

## 📊 Results Analysis

### Aggregating Results

For Cross Entropy experiments:
```bash
python eval/aggregate.py --dir output --datasets 'gaussian' 'gmm2' 'beta'
```
This will:
- Compute L2-smoothness metrics for saved PMFs
- Save metrics to `smoothness_dict.json`
- Print performance tables for each dataset
- Show the best model for KL divergence, MSE, and smoothness

For MLE experiments:
```bash
python eval/aggregate_mle.py --dir output --datasets 'gaussian' 'gmm2' 'beta'
```

### Visualization

Generate KL divergence and smoothness plots (Cross Entropy):
```bash
python eval/graphing/graph_varying_freqs.py --dir output
```

Generate KL divergence and perplexity plots (MLE):
```bash
python eval/graphing/graph_varying_freqs_mle.py --dir output
```

### Comparing Learned Distributions

Visualize PMF vs true PDF (Cross Entropy models):
```bash
python eval/graphing/visualize_pmfs.py
```

Visualize learned PDF vs true PDF (MLE models):
```bash
python eval/graphing/visualize_pdfs_mle.py
```

> 📝 The visualization scripts can be customized by modifying the indices and output paths at the bottom of each file.

## 📦 Output Structure

- Model checkpoints and predictions are saved in the `output` directory
- Each run creates:
  - Final predicted PMF/PDF (`.npy` files)
  - True PMF/PDF (`.npy` files)
  - Model metrics (`model_metrics.json`)