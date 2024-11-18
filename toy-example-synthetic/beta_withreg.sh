#!/bin/bash

dataset='beta'

for seed in 1 2 3 42; do
  for n_freqs in {2..20..2}; do
    echo "no regularization"
    echo "dataset = $dataset"
    echo "fourier_frequencies = $n_freqs"
    echo "seed = $seed"
    python toy_synthetic.py --head "fourier" --n_freqs $n_freqs --dataset $dataset --gamma 1e-6 --seed $seed
  done
done