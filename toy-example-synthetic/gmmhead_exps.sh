#!/bin/bash

#dataset=$1

for dataset in "beta" "gmm" "gmm2"; do
  for seed in 1 2 3 42; do
    echo "dataset = $dataset"
    echo "gmm head"
    echo "seed = $seed"
    python toy_synthetic.py --head "gmm" --n_gaussians 2 --dataset $dataset --seed $seed
  done
done


dataset="gaussian"

for seed in 1 2 3 42; do
    echo "dataset = $dataset"
    echo "gmm head"
    echo "seed = $seed"
    python toy_synthetic.py --head "gmm" --n_gaussians 1 --dataset $dataset --seed $seed
done