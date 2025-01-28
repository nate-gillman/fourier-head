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
    if path.endswith("/toy-example-synthetic/scripts"):
        sys.path.append(path.replace("/toy-example-synthetic/scripts", "/"))

from fourier_head import Fourier_Head
from generate_datasets import *

# Define the MLP model with a hidden layer and a linear/fourier head
class MLP(nn.Module):
    def __init__(self, input_size):

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
    num_samples=1000, 
    var=0.01, 
    batch_size=32, 
    seed=42, 
    bins=50, 
    logging=False):

    # Start a new wandb run to track this script
    if logging:
        head = 'linear_mse'
        wandb.init(
            project="fourier_toy_synthetic_mse",
            name=head+"_"+exper+"_"+str(freqs)+"_"+str(gamma)+"_"+str(seed),
            config={
                "architecture": "MLP_" + head,
                "dataset": exper,
                "freqs": 0,
                "seed": seed,
                "regularization_gamma": 0.0
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
    model = MLP(input_size=2).cuda()
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    bin_edges = np.linspace(-1, 1, bins + 1)
    bin_centers = torch.tensor((bin_edges[:-1] + bin_edges[1:])/2, dtype=torch.float32).cuda()
    print(torch.mean(bin_centers[y_test]**2))
    if exper == 'gaussian':
        target_pdfs = torch.tensor(np.array([gaussian_pdf(bin_centers.cpu(), x[1], var) for x in undig_test])).cuda()
    
    elif exper == 'gmm2':
        target_pdfs = torch.tensor(np.array([gmm2_pdf(bin_centers.cpu(), x, var) for x in undig_test])).cuda()

    elif exper == 'beta':
        target_pdfs = torch.tensor(np.array([beta_pdf(bin_centers.cpu(), x, var) for x in undig_test])).cuda()

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
                quantized_predicted = quantize_dataset(predicted.cpu(), bins)
                mse = torch.mean((bin_centers[quantized_predicted] - bin_centers[y_test].unsqueeze(-1))**2)
                tqdm.write(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, MSE: {mse:.4f}')

                if logging:
                    wandb.log({"loss": avg_loss, "MSE": mse})
            model.train()

    if logging:
        wandb.finish()
    return {"MSE": mse.item()}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse input arguments.")
    
    # Adding arguments
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Path to the dataset (string)')
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
    dataset_dict = {"gaussian": generate_gaussian_dataset, 'gmm2': generate_gmm_dataset2, 'beta': generate_beta_dataset}
    dataset = dataset_dict[args.dataset](num_samples, var, seed=args.seed)
    metrics = run_experiment(
        args.dataset, 
        dataset, 
        epochs=epochs,
        num_samples=num_samples, 
        var=var, 
        seed=args.seed,
        logging=args.wandb,
        bins=bins
    )

    prefix = f'output/{args.dataset}/'
    model_path = f'linear_mse/0.0/0/'
    os.makedirs(prefix+model_path, exist_ok=True)

    metrics_path = prefix+model_path+"mse.json"
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
    ## python toy_synthetic_regression.py --dataset "gmm2" --seed 42