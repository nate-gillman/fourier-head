#!/bin/bash

# Check if dataset argument is provided
if [ -z "$1" ]; then
  echo "Please provide a dataset as an argument. Options are: 'gaussian', 'gmm2', 'beta'."
  exit 1
fi

dataset=$1

# with regularization
for seed in 1 2 3 42; do
  for n_freqs in {2..20..2}; do
    echo "regularization_gamma = 1e-6"
    echo "dataset = $dataset"
    echo "fourier_frequencies = $n_freqs"
    echo "seed = $seed"
    python toy_synthetic_mle.py --head "fourier-mle" --n_freqs $n_freqs --dataset $dataset --gamma 1e-6 --seed $seed
  done
done
