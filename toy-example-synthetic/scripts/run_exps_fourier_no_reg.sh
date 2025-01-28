#!/bin/bash

# Check if dataset argument is provided
if [ -z "$1" ]; then
  echo "Please provide a dataset as an argument. Options are: 'gaussian', 'gmm2', 'beta'"
  exit 1
fi

dataset=$1

# no regularization
for seed in 1 2 3 42; do
  for n_freqs in {2..20..2}; do
    echo "no regularization"
    echo "dataset = $dataset"
    echo "fourier_frequencies = $n_freqs"
    echo "seed = $seed"
    python scripts/toy_synthetic.py --head "fourier" --n_freqs $n_freqs --dataset $dataset --gamma 0.0 --seed $seed
  done
done