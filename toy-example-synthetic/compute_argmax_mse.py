from generate_datasets import *
import os, json
from sklearn.model_selection import train_test_split
from toy_synthetic import quantize_dataset

var = 0.01
num_samples = 5000
bins = 50
bin_edges = np.linspace(-1, 1, bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2


dataset_dict = {"gaussian": generate_gaussian_dataset, 'gmm': generate_gmm_dataset, 'gmm2': generate_gmm_dataset2, "beta": generate_beta_dataset}

def compute_expected_value(pmfs, bins):
    return np.sum(np.arange(bins) * pmfs, axis=1)

def compute_mse(base_dir, exper):
    all_data = []  # List to store all rows of data
    l2_metrics = []
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(os.path.join(base_dir, exper)):
        for file in files:
             if file.endswith('.npy'):
                if file[0] == 't':
                    continue
                seed = file.split('_')[-1].split('.')[0]
                pred_pmfs = np.load(os.path.join(root, file))
                predicted = np.argmax(pred_pmfs, axis=-1)
                print(file, seed)
                dataset = dataset_dict[exper](num_samples, var, seed=int(seed))
                X = dataset[:, :2]  # Features: (u, v)
                y = dataset[:, 2]   # Target: w
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                y_test = quantize_dataset(y_test, bins)
                mse_max = np.mean((bin_centers[predicted] - bin_centers[y_test])**2)

                expected_vals = compute_expected_value(pred_pmfs, bins)
                expected_vals = bin_centers[np.round(expected_vals).astype(int)]
                mse_expected = np.mean((expected_vals - bin_centers[y_test])**2)

                metrics_path =  os.path.join(root, "model_metrics.json")
                with open(metrics_path, "r") as json_file:
                    metrics_all = json.load(json_file)
                    metrics_all[str(seed)]["MSE"] = mse_expected
                    metrics_all[str(seed)]["MSE_argmax"] = mse_max
                with open(metrics_path, "w") as json_file:
                    json.dump(metrics_all, json_file, indent=4)


compute_mse("eval/graphing/saved_pmfs", "gmm2")