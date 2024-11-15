import numpy as np
from toy_synthetic import *
from sklearn.model_selection import train_test_split
import torch

def compute_expected_value(pmfs, bins):
    return np.sum(np.arange(bins) * pmfs, axis=1)


var = 0.01
num_samples = 5000
bins = 50
bin_edges = np.linspace(-1, 1, bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

dataset_dict = {"gaussian": generate_gaussian_dataset, 'gmm': generate_gmm_dataset, 'gmm2': generate_gmm_dataset2}
exper = 'gmm'

best_mse = 10000
gamma = 0.0
freq = 12

for exper in ['gaussian', 'gmm', 'gmm2']:
#for gamma in ['0.0', '1e-06']:
    mses = []
    for seed in [1,2,3,42]:
        dataset = dataset_dict[exper](num_samples, var, seed=seed)
        X = dataset[:, :2]  # Features: (u, v)
        y = dataset[:, 2]   # Target: w
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test = torch.tensor(quantize_dataset(X_test, bins), dtype=torch.float32)
        y_test = torch.tensor(quantize_dataset(y_test, bins), dtype=torch.long)

        predicted_pmfs = np.load(f'eval/graphing/saved_pmfs/{exper}/linear/{gamma}/{0}/pmfs_{seed}.npy')
        expected_vals = compute_expected_value(predicted_pmfs, bins)
        expected_vals = bin_centers[np.round(expected_vals).astype(int)]
        #expected_vals = np.sum(bin_centers * predicted_pmfs, axis=1)
        mses.append(np.mean((expected_vals - bin_centers[y_test])**2))
    mse = np.mean(mses)
    std_dev = np.std(mses)
    print(f'{exper} mse: {mse}, std_dev: {std_dev}')
    # if mse < best_mse:
    #     best_mse = mse
    #     best_gamma = gamma
    #     best_freq = freq


#print(f'{exper} best mse: {best_mse}, best gamma: {best_gamma}, best freqL: {best_freq}')


