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
from scipy.stats import norm
from tqdm import tqdm
import wandb, json
import os
import argparse
import sys

from generate_datasets import *

for path in sys.path:
    if path.endswith("/toy-example-synthetic/scripts"):
        sys.path.append(path.replace("/toy-example-synthetic/scripts", "/"))

from fourier_head_mle import Fourier_Head_MLE
from gmm_head_mle import GMM_Head_MLE

def compute_nearest_bin_pmf(bin_centers, samples):
    """
    Compute a PMF by counting samples closest to each bin center.
    
    Parameters:
    -----------
    bin_centers : ndarray
        1D array of bin centers
    samples : ndarray
        1D array of samples to be binned
    
    Returns:
    --------
    ndarray
        PMF array of same length as bin_centers, where each element is the
        count of samples closest to that bin center, normalized to sum to 1
    """

    # Input validation
    if bin_centers.ndim != 1 or samples.ndim != 1:
        raise ValueError("Both inputs must be 1-dimensional arrays")
    
    # For each sample, compute distance to all bin centers
    # Using broadcasting to compute distances efficiently
    distances = np.abs(samples[:, np.newaxis] - bin_centers)
    
    # Find index of closest bin center for each sample
    nearest_bin_indices = np.argmin(distances, axis=1)
    
    # Count occurrences of each bin index
    pmf = np.bincount(nearest_bin_indices, minlength=len(bin_centers))
    
    # Normalize to create proper probability distribution
    pmf = pmf / len(samples)
    
    return pmf


# Define the MLP model with a hidden layer and a linear/fourier head
class MLP(nn.Module):
    def __init__(self, input_size, num_classes, head='linear', num_frequencies=9, regularizion_gamma=0, num_gaussians=0):
        super(MLP, self).__init__()
        self.mlp_head = nn.Linear(32, num_classes)
        if head == 'fourier-mle':
            self.mlp_head = Fourier_Head_MLE(32, num_frequencies, regularizion_gamma)
        elif head == 'gmm-mle':
            self.mlp_head = GMM_Head_MLE(32, num_gaussians)

        else:
            return ValueError("Invalid head type")

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
    head='fourier-mle', 
    batch_size=32, 
    seed=42, 
    gamma=0, 
    bins=1000,
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
                "seed": seed,
                "regularization_gamma": gamma,
                "num_samples" : num_samples
            }
        )

    # Split the data into inputs (u, v) and output (w)
    X = dataset[:, :2]  # Features: (u, v)
    y = dataset[:, 2]   # Target: w

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    undig_test = X_test # unquantized version of test data

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create PyTorch DataLoader for batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model, loss function, and optimizer
    model = MLP(input_size=2, num_classes=bins, head=head, num_frequencies=freqs, regularizion_gamma=gamma, num_gaussians=args.n_gaussians).cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    bin_edges = np.linspace(-1, 1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    pdf_dict = {'gaussian': gaussian_pdf, 'gmm2': gmm2_pdf, 'beta': beta_pdf}
    target_pdfs = torch.tensor(np.array([pdf_dict[exper](bin_centers, x, var) for x in undig_test])).cuda()

    #bin_centers = bin_centers.expand(X_test.shape[0], bin_centers.shape[0]).cuda()

    saved_pdfs = None
    kl = None
    # Training loop with Wandb logging
    pbar = tqdm(range(epochs), desc="Training Progress")
    for epoch in pbar:
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs) # (bs=32, 2) --> (fourier_pdf, which maps (bs=32,) --> (bs=32,))
            if head == "fourier-mle":
                # loss is negative log likelihood plus regularization
                pdf_values = outputs.evaluate_at(labels)
                loss = torch.mean(-torch.log(pdf_values + 1e-10)) + model.mlp_head.loss_regularization

            elif head == 'gmm-mle':
                # loss is negative log likelihood
                pdf_values = outputs.evaluate_at(labels)
                loss = torch.mean(-torch.log(pdf_values + 1e-10))

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            running_loss += loss.item()

        # Average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        pbar.set_description(f"avg_loss for past epoch = {avg_loss:.4f}")
        scheduler.step()

        if logging:
            wandb.log({"loss": avg_loss})
        
        if (epoch+1) % 10 == 0:
            model.eval()
            # Evaluate the model
            with torch.no_grad():
               
                model_pdf = model(X_test.cuda()) # (1000, 2) --> (a fourier pdf which maps (bs=1000,) --> (bs=1000)

                model_pdf = model(X_test.cuda()) # (1000, 2) --> (a fourier pdf which maps (bs=1000,) --> (bs=1000)
                model_pdf_x_vals = torch.from_numpy(bin_centers).expand(X_test.shape[0], bin_centers.shape[0]).cuda() # (1000, 200)
                model_pdf_vals = []
                for batch_idx in range(model_pdf_x_vals.shape[1]):
                    model_pdf_vals_ = model_pdf.evaluate_at(model_pdf_x_vals[:, batch_idx])
                    model_pdf_vals.append(model_pdf_vals_)
                model_pdf_vals = torch.stack(model_pdf_vals).T # (1000, 200)
                model_pdf_vals = model_pdf_vals * (2.0/bins)
                
                if epoch == epochs-1:
                    saved_pdfs = (model_pdf_vals, target_pdfs)
  
                # KL divergence
                kl = kl_loss(torch.clamp(model_pdf_vals, min=1e-10).log(), torch.clamp(target_pdfs, min=1e-10))

                # Perplexity
                model_pdf_test = torch.clamp(model_pdf.evaluate_at(y_test.cuda()), min=1e-10)
                nll = torch.mean(-torch.log(model_pdf_test))
                perplexity = torch.exp(nll)

                tqdm.write(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, KL divergence: {kl:.4f}, Perplexity: {perplexity:.4f}')
                if logging:
                    wandb.log({"loss": avg_loss, "KL divergence": kl, "Perplexity": perplexity})

            model.train()

    if logging:
        wandb.finish()
    return saved_pdfs, {"KL divergence":kl.item(), "Perplexity": perplexity.item()}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse input arguments.")
    
    # Adding arguments
    parser.add_argument('--head', type=str, required=True, 
                        help='Specify head option (string)')
    parser.add_argument('--n_freqs', type=int, required=False, default=0,
                        help='Number of frequencies (int)')
    parser.add_argument('--n_gaussians', type=int, required=False,
                        help='Number of frequencies (int)')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Name of the dataset (string)')
    parser.add_argument('--dataset_size', type=int, default=5000, 
                        help='Size of the dataset (int)')
    parser.add_argument('--gamma', type=float, default=0.0, 
                        help='Gamma value (float)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Seed value (int)')
    parser.add_argument('--graph', action='store_true', help='Flag to enable test graphing')
    parser.add_argument('--wandb', action='store_true', help='Flag to enable wandb logging')
    
    # Parsing arguments
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    epochs = 500
    num_samples = args.dataset_size
    var = 0.01
    dataset_dict = {'gaussian': generate_gaussian_dataset, 'gmm2': generate_gmm_dataset2, 'beta': generate_beta_dataset}
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
    )

    prefix = f'output/{args.dataset}/'
    model_path = f'{args.head}/{args.gamma}/{args.n_freqs}/'
    os.makedirs(prefix+model_path, exist_ok=True)
    np.save(prefix+model_path+f'pmfs_{args.seed}.npy', pdfs[0].cpu())
    np.save(prefix+f'true_mle_{args.seed}.npy', pdfs[1].cpu())

    metrics_path = prefix+model_path+"mle_model_metrics.json"
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
    ## python toy_synthetic_mle.py --head "fourier-mle" --n_freqs 12 --dataset "gmm2" --gamma 0.0