# Toy Experiment: Learning a Known Conditional Distribution

![teaser](misc/assets/toy_predicted_vs_true.png)

## Recreating results from paper

<details>
  <summary><b> Environment setup </b></summary>

<br>

## Environment

```bash
# create and activate conda environment
conda create -n toy-example-synthetic python=3.11
conda activate toy-example-synthetic

# install any version of torch, verify it was installed correctly
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
python -c 'import torch; print(torch.cuda.is_available()); a = torch.zeros(5); a = a.to("cuda:0"); print(a)'

# install remaining things
conda install scikit-learn tqdm pandas matplotlib
pip install wandb
```

</details>

## Running the experiments

<details>
  <summary><b> Example usage </b></summary>

<br>

The experiment can be run on three datasets: `gaussian`, `gmm2`, `beta`. 
This is how you can run the linear classification head on the `gmm2` dataset:

```bash
python scripts/toy_synthetic.py --head "linear" --n_freqs 0 --dataset "gmm2"
```

This is how you can run the Fourier head with no regularization and 12 frequencies:
```bash
python scripts/toy_synthetic.py --head "fourier" --n_freqs 12 --gamma 0.0 --dataset "gmm2" 
```

And this is how you can run the Fourier head with `1e-6` regularization and 12 frequencies:
```bash
python scripts/toy_synthetic.py --head "fourier" --n_freqs 12 --gamma 1e-6 --dataset "gmm2" 
```

To log the experiments to wandb, you can add a `--wandb` flag.
KL divergence and MSE are evaluated and printed every 10 epochs. Each run saves the final predicted pmf and true pmf to the appropriate model directory as `npy` files under the `output` directory. The metrics are saved in `model_metrics.json` in the model directory.

</details>

<details>
  <summary><b> Reproducing all experiments from the paper </b></summary>

<br>

To reproduce all the synthetic toy experiments in the paper, you can run the following scripts.
Each script takes less than 6 hours on a geforce3090 GPU.

```bash
# linear classification head
sh scripts/run_exps_linear.sh gaussian
sh scripts/run_exps_linear.sh gmm2
sh scripts/run_exps_linear.sh beta

# gaussian mixture model head
sh scripts/run_exps_gmm.sh gaussian
sh scripts/run_exps_gmm.sh gmm2
sh scripts/run_exps_gmm.sh beta

# fourier head (with no regularization)
sh scripts/run_exps_fourier_no_reg.sh gaussian
sh scripts/run_exps_fourier_no_reg.sh gmm2
sh scripts/run_exps_fourier_no_reg.sh beta

# fourier head (with regularization)
sh scripts/run_exps_fourier_reg.sh gaussian
sh scripts/run_exps_fourier_reg.sh gmm2
sh scripts/run_exps_fourier_reg.sh beta

# linear regression head (pointwise estimate)
sh scripts/run_exps_linear_regression.sh gaussian
sh scripts/run_exps_linear_regression.sh gmm2
sh scripts/run_exps_linear_regression.sh beta
```

Once all the experiments have finished, to aggregate all the results from the experiments, run:
```bash
python eval/aggregate.py --dir output --datasets 'gaussian' 'gmm2' 'beta'
```

This will also compute the L2-smoothness metrics for the saved pmfs and save them to `smoothness_dict.json` in the appropriate model directory. It will print a table for each dataset showing the aggregated metrics as well as the best model for each of the three metrics (KL divergence, MSE, smoothness). 

Finally, we can graph the KL divergence and smoothness as number of Fourier frequencies vary via:

```bash
python eval/graphing/graph_varying_freqs.py --dir output
```

### MLE training:

To reproduce all the synthetic toy experiments using MLE training, you can run the following scripts.
Each script took less than 24h on a geforce3090 GPU.

```bash
# Fourier-MLE head (with no regularization)
sh scripts/run_exps_fourier-mle_no_reg.sh gaussian
sh scripts/run_exps_fourier-mle_no_reg.sh gmm2
sh scripts/run_exps_fourier-mle_no_reg.sh beta

# Fourier-MLE head (with regularization)
sh scripts/run_exps_fourier-mle_reg.sh gaussian
sh scripts/run_exps_fourier-mle_reg.sh gmm2
sh scripts/run_exps_fourier-mle_reg.sh beta

# GMM-MLE head
sh scripts/run_exps_gmm-mle.sh gaussian
sh scripts/run_exps_gmm-mle.sh gmm2
sh scripts/run_exps_gmm-mle.sh beta
```

Once all the experiments have finished, to aggregate all the results from the MLE experiments, run:
```bash
python eval/aggregate_mle.py --dir output --datasets 'gaussian' 'gmm2' 'beta'
```
Finally, we can graph the KL divergence and Perplexity as number of Fourier frequencies vary via:

```bash
python eval/graphing/graph_varying_freqs_mle.py --dir output
```


## Visualizing learned pmfs/pdfs 

We can also visualize the learned pmf vs true pdf for the Linear and Fourier heads on the 3 datasets using: 
```bash
python eval/graphing/visualize_pmfs.py 
```
You can alter the lines at the bottom of that file to choose from your output pmfs as well as set different indices for the pmfs to be visualized (currently it runs with the pmfs in our paper).

Similarly, we can visualize the learned pdf vs true pdf for the GMM-MLE and Fourier-MLE heads using:
```bash
python eval/graphing/visualize_pdfs_mle.py 
```

</details>


