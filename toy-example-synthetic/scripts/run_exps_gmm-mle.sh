#!/bin/bash

# Check if dataset argument is provided
if [ -z "$1" ]; then
  echo "Please provide a dataset as an argument. Options are: 'gaussian', 'gmm2', 'beta'"
  exit 1
fi

dataset=$1

if [[ "$dataset" == "beta" || "$dataset" == "gmm2" ]]; then
  for seed in 1 2 3 42; do
    echo "dataset = $dataset"
    echo "gmm head"
    echo "seed = $seed"
    python scripts/toy_synthetic_mle.py --head "gmm-mle" --n_gaussians 2 --dataset $dataset --seed $seed
  done
elif [[ "$dataset" == "gaussian" ]]; then
  for seed in 1 2 3 42; do
    echo "dataset = $dataset"
    echo "gmm head"
    echo "seed = $seed"
    python scripts/toy_synthetic_mle.py --head "gmm-mle" --n_gaussians 1 --dataset $dataset --seed $seed
  done
else
  echo "Invalid dataset: $dataset. Options are: 'gaussian', 'gmm2', 'beta'"
  exit 1
fi
