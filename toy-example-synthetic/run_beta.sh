#!/bin/bash

dataset='beta'

for seed in 42; do
  for n_freqs in {2..10..2}; do
    echo "no regularization"
    echo "dataset = $dataset"
    echo "fourier_frequencies = $n_freqs"
    echo "seed = $seed"
    python toy_synthetic.py --head "fourier" --n_freqs $n_freqs --dataset $dataset --gamma 0.0 --seed $seed
  done
done