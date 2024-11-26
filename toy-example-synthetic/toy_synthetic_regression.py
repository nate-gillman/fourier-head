"""
The MIT License (MIT) Copyright (c) 2024 Nate Gillman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
and associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial 
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import norm
from tqdm import tqdm
import wandb, json
import os
import argparse
import sys

for path in sys.path:
    if path.endswith("/toy-example-synthetic"):
        sys.path.append(path.replace("/toy-example-synthetic", "/"))

from fourier_head import Fourier_Head

def generate_gaussian_dataset(n_samples, var=0.1, seed=42):
    """
    Generates a 3D dataset with n_samples samples.

    The dataset is generated as follows:
    1. x is sampled uniformly from (-0.8, 0.8)
    2. y is sampled from a Gaussian centered at x with variance var
    3. z is sampled from a Gaussian centered at y with variance var

    Parameters:
    - n_samples (int): Number of samples to generate.

    Returns:
    - dataset (ndarray): An array of shape (n_samples, 3) containing the 3D dataset.
    """
    rng = np.random.default_rng(seed=seed)
    # Step 1: Sample x uniformly from (-0.8, 0.8)
    x = rng.uniform(-0.8, 0.8, n_samples)

    # Step 2: Sample y from a Gaussian centered at x with variance var
    y = rng.normal(loc=x, scale=np.sqrt(var))

    # Step 3: Sample z from a Gaussian centered at y with variance var
    z = rng.normal(loc=y, scale=np.sqrt(var))

    # Combine x, y, z into a single dataset
    dataset = np.vstack((x, y, z)).T
    return dataset

def gaussian_pdf(bin_centers, loc, var=0.01):
    pmf =  norm.pdf(bin_centers, loc, np.sqrt(var))*2 / bin_centers.shape[0]
    return pmf / np.sum(pmf)

def generate_gmm_dataset(n_samples, var=0.01, seed=42):
    """
    Generates a 3D dataset with n_samples samples.

    The dataset is generated as follows:
    1. x is sampled uniformly from (-0.8, 0.8)
    2. y is sampled from a Gaussian centered at x with variance 0.01
    3. z is sampled from a GMM with means min{x,y}-0.1 and max{x,y}+0.1, each with variance 0.01

    Parameters:
    - n_samples (int): Number of samples to generate.

    Returns:
    - dataset (ndarray): An array of shape (n_samples, 3) containing the 3D dataset.
    """

    rng = np.random.default_rng(seed=seed)
    # Step 1: Sample x uniformly from (-0.8, 0.8)
    x = rng.uniform(-0.8, 0.8, n_samples)

    # Step 2: Sample y from a Gaussian centered at x with variance 0.01
    y = rng.normal(loc=x, scale=np.sqrt(var), size=n_samples)

    # Step 3: Sample z from a GMM with means x and y, each with variance 0.01
    z = np.zeros(n_samples)

    a = np.minimum(x,y) - 0.1
    b = np.maximum(x,y) + 0.1
    for i in range(n_samples):
        # Randomly choose either x[i] or y[i] as the mean for z
        if rng.uniform(0, 1) < 0.5:
            z[i] = rng.normal(loc=a[i], scale=np.sqrt(var))
        else:
            z[i] = rng.normal(loc=b[i], scale=np.sqrt(var))

    # Combine x, y, z into a single dataset
    dataset = np.vstack((x, y, z)).T
    return dataset

def generate_gmm_dataset2(n_samples, var=0.01, seed=42):
    """
    Generates a 3D dataset with n_samples samples.

    The dataset is generated as follows:
    1. x and y are sampled uniformly from (-0.8, 0.8)
    3. z is sampled from a GMM with means x and y, each with variance var

    Parameters:
    - n_samples (int): Number of samples to generate.

    Returns:
    - dataset (ndarray): An array of shape (n_samples, 3) containing the 3D dataset.
    """
    rng = np.random.default_rng(seed=seed)
    # Step 1: Sample x uniformly from (-0.8, 0.8)
    x = rng.uniform(-0.8, 0.8, n_samples)
    y = rng.uniform(-0.8, 0.8, n_samples)

    # Step 3: Sample z from a GMM with means x and y, each with variance 0.01
    z = np.zeros(n_samples)
    for i in range(n_samples):
        # Randomly choose either x[i] or y[i] as the mean for z
        if rng.uniform(0, 1) < 0.5:
            z[i] = rng.normal(loc=x[i], scale=np.sqrt(var))
        else:
            z[i] = rng.normal(loc=y[i], scale=np.sqrt(var))

    # Combine x, y, z into a single dataset
    dataset = np.vstack((x, y, z)).T
    return dataset


def gmm1_pdf(bin_centers, locs, var=0.01):
    return (gaussian_pdf(bin_centers, np.min(locs)-0.1, var) + gaussian_pdf(bin_centers, np.max(locs)+0.1, var))/2

def gmm2_pdf(bin_centers, locs, var=0.01):
    return (gaussian_pdf(bin_centers, locs[0], var) + gaussian_pdf(bin_centers, locs[1], var))/2

# Quantization function, assuming dataset in the range (-1, 1)
def quantize_dataset(dataset, b):
    data_range = (-1, 1)
    bin_edges = np.linspace(data_range[0], data_range[1], b + 1)
    digitized_data = np.digitize(dataset, bin_edges) - 1
    digitized_data = np.clip(digitized_data, 0, b - 1)
    return digitized_data


# Define the MLP model with a hidden layer and a linear/fourier head
class MLP(nn.Module):
    def __init__(self, input_size, num_classes, head='linear_mse', num_frequencies=9, regularizion_gamma=0):
        super(MLP, self).__init__()
        self.mlp_head = nn.Linear(32, 1)

        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            self.mlp_head
        )
        
    def forward(self, x):
        return self.layers(x)
    

# Function to run the experiment
def run_experiment(
    exper, 
    dataset, 
    epochs, 
    freqs=0, 
    num_samples=1000, 
    var=0.01, 
    head='linear', 
    batch_size=32, 
    seed=42, 
    gamma=0, 
    bins=50, 
    logging=False):

    # Start a new wandb run to track this script
    if logging:
        wandb.init(
            project="fourier_toy_synthetic_mse",
            name=head+"_"+exper+"_"+str(freqs)+"_"+str(gamma)+"_"+str(seed),
            config={
                "architecture": "MLP_" + head,
                "dataset": exper,
                "freqs": freqs,
                "seed": seed,
                "regularization_gamma": gamma
            }
        )

    # Split the data into inputs (u, v) and output (w)
    X = dataset[:, :2]  # Features: (u, v)
    y = dataset[:, 2]   # Target: w

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    undig_test = X_test # unquantized version of test data

    # Convert to PyTorch tensors
    X_train = torch.tensor(quantize_dataset(X_train, bins), dtype=torch.float32)
    X_test = torch.tensor(quantize_dataset(X_test, bins), dtype=torch.float32)
    y_train = torch.tensor(quantize_dataset(y_train, bins), dtype=torch.long)
    y_test = torch.tensor(quantize_dataset(y_test, bins), dtype=torch.long)

    # Create PyTorch DataLoader for batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model, loss function, and optimizer
    model = MLP(input_size=2, num_classes=bins, head=head, num_frequencies=freqs, regularizion_gamma=gamma).cuda()
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    bin_edges = np.linspace(-1, 1, bins + 1)
    bin_centers = torch.tensor((bin_edges[:-1] + bin_edges[1:])/2, dtype=torch.float32).cuda()

    if exper == 'gaussian':
        target_pdfs = torch.tensor(np.array([gaussian_pdf(bin_centers.cpu(), x[1], var) for x in undig_test])).cuda()

    elif exper == 'gmm':
        target_pdfs = torch.tensor(np.array([gmm1_pdf(bin_centers.cpu(), x, var) for x in undig_test])).cuda()
    
    else:
        target_pdfs = torch.tensor(np.array([gmm2_pdf(bin_centers.cpu(), x, var) for x in undig_test])).cuda()

    saved_pdfs = None
    kl = None
    mse = None
    # Training loop with Wandb logging
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, bin_centers[labels].unsqueeze(-1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            # Evaluate the model
            with torch.no_grad():
                predicted = model(X_test.cuda())
                y_test = y_test.cpu()
              
                # MSE
                # print((predicted - bin_centers[y_test].unsqueeze(-1)).reshape((-1)))
                mse = torch.mean((predicted - bin_centers[y_test].unsqueeze(-1))**2)
                
                tqdm.write(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, MSE: {mse:.4f}')

                if logging:
                    wandb.log({"loss": avg_loss, "MSE": mse})
            model.train()

    if logging:
        wandb.finish()
    return saved_pdfs, {"MSE": mse}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse input arguments.")
    
    # Adding arguments
    parser.add_argument('--head', type=str, required=True, 
                        help='Specify head option (string)')
    parser.add_argument('--n_freqs', type=int, required=True, 
                        help='Number of frequencies (int)')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Path to the dataset (string)')
    parser.add_argument('--gamma', type=float, default=0.0, 
                        help='Gamma value (float)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Seed value (int)')
    parser.add_argument('--wandb', action='store_true', help='Flag to enable wandb logging')
    
    # Parsing arguments
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    epochs = 500
    num_samples = 5000
    var = 0.01
    bins = 50
    dataset_dict = {"gaussian": generate_gaussian_dataset, 'gmm': generate_gmm_dataset, 'gmm2': generate_gmm_dataset2}
    dataset = dataset_dict[args.dataset](num_samples, var, seed=args.seed)
    pdfs, metrics = run_experiment(
        args.dataset, 
        dataset, 
        epochs=epochs,
        freqs=args.n_freqs, 
        num_samples=num_samples, 
        var=var, 
        head=args.head, 
        seed=args.seed,
        gamma=args.gamma,
        logging=args.wandb,
        bins=bins
    )

    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'toy-example-synthetic':
        prefix = f'output/{args.dataset}/'
    else:
        prefix = f'toy-example-synthetic/output/{args.dataset}/'
    
    model_path = f'{args.head}/{args.gamma}/{args.n_freqs}/'
    os.makedirs(prefix+model_path, exist_ok=True)
    np.save(prefix+model_path+f'pmfs_{args.seed}.npy', pdfs[0].cpu())
    np.save(prefix+f'true_{args.seed}.npy', pdfs[1].cpu())

    metrics_path = prefix+model_path+"model_metrics.json"
    if os.path.exists(metrics_path):
        # Load existing data
        with open(metrics_path, "r") as json_file:
            metrics_all = json.load(json_file)
    else:
        # If file doesn't exist, create an empty dictionary
        metrics_all = {}

    # Evaluate model and add results for the current seed
    metrics_all[str(args.seed)] = metrics

    with open(metrics_path, "w") as json_file:
        json.dump(metrics_all, json_file, indent=4)

    ## Run script via 
    ## python toy_synthetic.py --head "fourier" --n_freqs 12 --dataset "gmm2" --gamma 0.0