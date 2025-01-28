#!/bin/bash

# Check if dataset argument is provided
if [ -z "$1" ]; then
  echo "Please provide a dataset as an argument. Options are: 'gaussian', 'gmm2', 'beta'."
  exit 1
fi

dataset=$1

# linear experiments
for seed in 1 2 3 42; do
  echo "dataset = $dataset"
  echo "linear head"
  echo "seed = $seed"
  python toy_synthetic.py --head "linear" --n_freqs 0 --dataset $dataset --seed $seed
done
