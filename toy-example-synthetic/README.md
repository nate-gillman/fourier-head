# Toy Experiment: Learning a Known Conditional Distribution

## Environment

These experiments work in the `chronos` conda env built in [time-series-forecasting](../time-series-forecasting/README.md).

## Running the experiments

The experiment can be run on three datasets: `gaussian`, `gmm2`, `beta`. 

### Example usage:

Linear head:
```bash
python toy_synthetic.py --head "linear" --n_freqs 0 --dataset "gmm2"
```

Fourier head with no quadratic regularization:
```bash
python toy_synthetic.py --head "fourier" --n_freqs 12 --gamma 0.0 --dataset "gmm2" 
```

Fourier head with quadratic regularization:
```bash
python toy_synthetic.py --head "fourier" --n_freqs 12 --gamma 1e-6 --dataset "gmm2" 
```

* To log the experiments to wandb, use the `--wandb` flag.

KL divergence and MSE are evaluated and printed every 10 epochs. Each run saves the final predicted pmf and true pmf to the appropriate model directory as `npy` files under the `output` directory. The metrics are saved in `model_metrics.json` in the model directory.

To reproduce all the synthetic toy experiments, you can run the following scripts.
Each script took less than 24h on a geforce3090 GPU.

```bash
# linear head, running all datasets
cd scripts
sh ./run_exps_linear.sh gaussian
sh ./run_exps_linear.sh gmm2
sh ./run_exps_linear.sh beta

# fourier head with no regularization
sh ./run_exps_fourier_no_reg.sh gaussian
sh ./run_exps_fourier_no_reg.sh gmm2
sh ./run_exps_fourier_no_reg.sh beta

# experiments with regularization
sh ./run_exps_fourier_reg.sh gaussian
sh ./run_exps_fourier_reg.sh gmm2
sh ./run_exps_fourier_no_reg.sh beta
```

If you would like to also run experiments with the GMM-head (which learns parameters for an optimal Gaussian mixture model), you can run
```bash
sh ./run_exps_gmm.sh gaussian
sh ./run_exps_gmm.sh gmm2
sh ./run_exps_gmm.sh beta
```

You can also consider a pointwise estimate for the value of z given (x,y) via an MLP trained using an MSE objective. Linear-MSE experiments can be run using
```bash
sh ./run_exps_linear_regression.sh gaussian
sh ./run_exps_linear_regression.sh gmm2
sh ./run_exps_linear_regression.sh beta
```

Once all the experiments have finished, to aggregate all the results from the experiments, run:
```bash
cd ..
python eval/aggregate.py --dir output --datasets 'gaussian' 'gmm2' 'beta'
```

This will also compute the L2-smoothness metrics for the saved pmfs and save them to `smoothness_dict.json` in the appropriate model directory. It will print a table for each dataset showing the aggregated metrics as well as the best model for each of the three metrics (KL divergence, MSE, smoothness). 

Finally, we can graph the KL divergence and smoothness as number of Fourier frequencies vary via:

```bash
python eval/graphing/graph_varying_freqs.py --dir output
```

## Visualizing learned pmfs

We can also visualize the learned pmf vs true pdf for the Linear and Fourier heads on the 3 datasets using: 
```bash
python eval/graphing/visualize_pmfs.py 
```
You can alter the lines at the bottom of that file to choose from your output pmfs as well as set different indices for the pmfs to be visualized (currently it runs with the pmfs in our paper).
