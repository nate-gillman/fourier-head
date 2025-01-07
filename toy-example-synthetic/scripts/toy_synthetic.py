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
from tqdm import tqdm
import wandb, json
import os
import argparse
import sys

for path in sys.path:
    if path.endswith("/toy-example-synthetic/scripts"):
        sys.path.append(path.replace("/toy-example-synthetic/scripts", "/"))

from fourier_head import Fourier_Head
from gmm_head import GMM_Head
from generate_datasets import *


# Define the MLP model with a hidden layer and a linear/fourier head
class MLP(nn.Module):
    def __init__(self, input_size, num_classes, head='linear', num_frequencies=9, num_gaussians=0, regularizion_gamma=0):
        super(MLP, self).__init__()
        self.mlp_head = nn.Linear(32, num_classes)
        if head == 'fourier':
            self.mlp_head = Fourier_Head(32, num_classes, num_frequencies, regularizion_gamma)
        elif head == 'gmm':
            self.mlp_head = GMM_Head(32, num_classes, num_gaussians)

        else:
            return NotImplementedError


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
    gaussians=0,
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
            project="fourier_toy_synthetic",
            name=head+"_"+exper+"_"+str(freqs)+"_"+str(gamma)+"_"+str(seed),
            config={
                "architecture": "MLP_" + head,
                "dataset": exper,
                "freqs": freqs,
                "gaussians": gaussians,
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
    model = MLP(input_size=2, 
                num_classes=bins, 
                head=head, 
                num_frequencies=freqs, 
                num_gaussians=args.n_gaussians, 
                regularizion_gamma=gamma).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    bin_edges = np.linspace(-1, 1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    pdf_dict = {'gaussian': gaussian_pdf, 'gmm2': gmm2_pdf, 'beta': beta_pdf}
    target_pdfs = torch.tensor(np.array([pdf_dict[exper](bin_centers, x, var) for x in undig_test])).cuda()

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
            if head in ['linear', 'gmm']:
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels) + model.mlp_head.loss_regularization

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
                outputs = model(X_test.cuda())
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.cpu()
                y_test = y_test.cpu()

                # Accuracy
                accuracy = accuracy_score(y_test, predicted)

                # KL divergence
                pdfs = torch.softmax(outputs, 1)
                if epoch == epochs-1:
                  saved_pdfs = (pdfs, target_pdfs)
                kl_loss = nn.KLDivLoss(reduction='batchmean')
                kl = kl_loss((pdfs+1e-10).log(), target_pdfs.cuda())

                # MSE
                mse_max = np.mean((bin_centers[predicted]-bin_centers[y_test])**2)
                expected_bins = torch.sum(torch.arange(bins) * pdfs.cpu(), dim=1)
                expected_vals = bin_centers[torch.round(expected_bins).to(torch.int)]
                mse_expected = np.mean((expected_vals - bin_centers[y_test])**2)

                tqdm.write(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, KL divergence: {kl:.4f}, MSE (mean): {mse_expected:.4f}, MSE (argmax): {mse_max:.4f}')

                if logging:
                    wandb.log({"loss": avg_loss, "accuracy": accuracy, "KL divergence": kl, "MSE (mean)": mse_expected, "MSE (argmax)": mse_max})
            model.train()

    if logging:
        wandb.finish()
    return saved_pdfs, {"KL divergence":kl.item(), "MSE": mse_expected, "MSE_argmax": mse_max}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse input arguments.")
    
    # Adding arguments
    parser.add_argument('--head', type=str, required=True, 
                        help='Specify head option (string)')
    parser.add_argument('--n_freqs', type=int, default=0, 
                        help='Number of frequencies (int)')
    parser.add_argument('--n_gaussians', type=int, default=0,
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
    epochs = 20
    num_samples = 5000
    var = 0.01
    bins = 50
    dataset_dict = {'gaussian': generate_gaussian_dataset, 'gmm2': generate_gmm_dataset2, 'beta': generate_beta_dataset}
    dataset = dataset_dict[args.dataset](num_samples, var, seed=args.seed)
    pdfs, metrics = run_experiment(
        args.dataset, 
        dataset, 
        epochs=epochs,
        freqs=args.n_freqs,
        gaussians=args.n_gaussians,
        num_samples=num_samples, 
        var=var, 
        head=args.head, 
        seed=args.seed,
        gamma=args.gamma,
        logging=args.wandb,
        bins=bins
    )

    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'scripts':
        prefix = f'../output/{args.dataset}/'
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