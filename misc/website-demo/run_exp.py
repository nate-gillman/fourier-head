import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import norm
from tqdm import tqdm
import wandb
import os
import argparse
import sys
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

from _fourier_head import _Fourier_Head

sys.path.append("../../")
from smoothness_metric import get_smoothness_metric

class DistributionDataset(ABC):
    """Abstract base class for different probability distributions."""
    
    def __init__(self, dim_output: int):
        self.dim_output = dim_output
        
    @abstractmethod
    def generate_samples(self, batch_size: int) -> torch.Tensor:
        """Generate samples from the distribution."""
        pass
    
    @abstractmethod
    def get_kl_metric(self, softmax_logits: torch.Tensor) -> float:
        """Calculate KL divergence between softmax logits and true distribution."""
        pass
    
    @abstractmethod
    def get_ground_truth_pmf(self, shape: torch.Size) -> torch.Tensor:
        """Get the ground truth probability mass function."""
        pass
    
    def quantize_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """Quantize continuous samples into discrete bins."""
        bin_edges = np.linspace(-1, 1, self.dim_output + 1)
        digitized_data = np.digitize(samples.numpy(), bin_edges) - 1
        return torch.tensor(np.clip(digitized_data, 0, self.dim_output - 1))

class SquareWaveDataset(DistributionDataset):
    def generate_samples(self, batch_size: int) -> torch.Tensor:
        samples = torch.rand(batch_size) - 0.5
        return self.quantize_samples(samples)
    
    def get_ground_truth_pmf(self, shape: torch.Size) -> torch.Tensor:
        pmf = torch.zeros(shape).cuda()
        pmf[:, int(self.dim_output/4) : int(3*self.dim_output/4)] = 2 / self.dim_output
        return pmf
    
    def get_kl_metric(self, softmax_logits: torch.Tensor) -> float:
        ground_truth = self.get_ground_truth_pmf(softmax_logits.shape)
        return nn.KLDivLoss(reduction='batchmean')(torch.log(softmax_logits), ground_truth).detach().cpu().item()

    def get_discretized_pdf_curve(self, num_dots) -> np.array:
        
        _num_dots = int(num_dots)/5

        points_all = []

        # [-1, 0] to [-0.5, 0]
        x_vals = np.arange(-1, -0.5, 0.5/_num_dots)
        y_vals = np.zeros_like(x_vals)
        points = np.stack((x_vals, y_vals), axis=1)
        points_all.append(points)

        # [-0.5, 0] to [-0.5, 1/64]
        y_vals = np.arange(0, 2/self.dim_output, (2/self.dim_output)/(_num_dots))
        x_vals = np.zeros_like(y_vals) - 0.5
        points = np.stack((x_vals, y_vals), axis=1)
        points_all.append(points)

        # [-0.5, 1/64] to [0.5, 1/64]
        x_vals = np.arange(-0.5, 0.5, 1/(_num_dots))
        y_vals = np.zeros_like(x_vals) + 2/self.dim_output
        points = np.stack((x_vals, y_vals), axis=1)
        points_all.append(points)

        # [0.5, 1/64] to [0.5, 0]
        y_vals = np.arange(2/self.dim_output, 0, -(2/self.dim_output)/(_num_dots))
        x_vals = np.zeros_like(y_vals) + 0.5
        points = np.stack((x_vals, y_vals), axis=1)
        points_all.append(points)

        # [0.5, 0] to [1, 0]
        x_vals = np.arange(0.5, 1, 0.5/_num_dots)
        y_vals = np.zeros_like(x_vals)
        points = np.stack((x_vals, y_vals), axis=1)
        points_all.append(points)

        points_all = np.concatenate(points_all, axis=0)
        return points_all


class MixtureOfGaussiansDataset(DistributionDataset):
    def __init__(self, dim_output: int, means_and_stds):
        super().__init__(dim_output)
        self.means_and_stds = means_and_stds
        
    def generate_samples(self, batch_size: int) -> torch.Tensor:
        weights = [1.0 / len(self.means_and_stds)] * len(self.means_and_stds)
        samples = []
        
        for _ in range(batch_size):
            component_idx = np.random.choice(len(self.means_and_stds), p=weights)
            mean, std = self.means_and_stds[component_idx]
            
            sample = np.random.normal(mean, std)
            while sample < -1 or sample > 1:
                sample = np.random.normal(mean, std)
            samples.append(sample)
            
        return self.quantize_samples(torch.tensor(samples, dtype=torch.float32))
    
    def get_ground_truth_pmf(self, shape: torch.Size) -> torch.Tensor:
        bin_edges = torch.linspace(-1, 1, self.dim_output + 1)
        bin_centerpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        weights = 1.0 / len(self.means_and_stds)
        pdf_values = np.zeros_like(bin_centerpoints, dtype=np.float64)
        
        for mean, std in self.means_and_stds:
            pdf_values += weights * norm.pdf(bin_centerpoints, mean, std)
            
        pmf = torch.tensor(pdf_values / pdf_values.sum()).cuda()
        return pmf.unsqueeze(0).expand(shape)
    
    def get_kl_metric(self, softmax_logits: torch.Tensor) -> float:
        ground_truth = self.get_ground_truth_pmf(softmax_logits.shape)
        return nn.KLDivLoss(reduction='batchmean')(torch.log(softmax_logits), ground_truth).detach().cpu().item()

    def get_discretized_pdf_curve(self, num_dots) -> np.array:

        x_vals = np.linspace(-1, 1, num_dots)

        weights = 1.0 / len(self.means_and_stds)
        pdf_vals = np.zeros_like(x_vals, dtype=np.float64)
        for mean, std in self.means_and_stds:
            pdf_vals += weights * norm.pdf(x_vals, mean, std)

        # import pdb; pdb.set_trace()

        # normalize
        riemann_integral = ((2/num_dots) * pdf_vals[:-1]).sum()
        # pdf_vals_rescaled = pdf_vals / (riemann_integral*self.dim_output)
        pdf_vals_rescaled = pdf_vals / (riemann_integral*(self.dim_output/2))
        
        pdf_curve = np.stack((x_vals, pdf_vals_rescaled), axis=1)

        return pdf_curve

class FourierTrainer:
    """Handles the training of Fourier models for different distributions."""
    
    def __init__(
        self,
        dataset: DistributionDataset,
        dim_input: int,
        regularization_gamma: float,
        device: str = "cuda:0"
    ):
        self.dataset = dataset
        self.dim_input = dim_input
        self.regularization_gamma = regularization_gamma
        self.device = device
        
    def train_model(
        self,
        num_frequencies: int,
        num_training_steps: int = 2000,
        batch_size: int = 1024,
        initial_lr: float = 0.001
    ) -> Dict[str, float]:
        """Train a model and return metrics."""
        
        model = _Fourier_Head(
            self.dim_input,
            self.dataset.dim_output,
            num_frequencies,
            self.regularization_gamma,
            device=self.device
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        running_loss = 0.0
        pbar = tqdm(range(num_training_steps), desc="Training Progress")
        
        for step in pbar:
            # Linear learning rate decay
            current_lr = initial_lr * (1 - step / num_training_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
                
            gt_samples_labels = self.dataset.generate_samples(batch_size).to(self.device)
            
            optimizer.zero_grad()
            inverse_softmax_logits = model(torch.zeros(batch_size, self.dim_input).to(self.device))
            loss = criterion(inverse_softmax_logits, gt_samples_labels) + model.loss_regularization
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            avg_loss = running_loss / (step + 1)
            
            softmax_logits = torch.softmax(inverse_softmax_logits, dim=-1)[0:1]
            kl_divergence = self.dataset.get_kl_metric(softmax_logits)
            
            pbar.set_description(f"Train loss = {avg_loss:.4f}, kl divergence = {kl_divergence:.4f}")
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            final_logits = model(torch.zeros(1, self.dim_input).to(self.device))
            softmax_logits = torch.softmax(final_logits, dim=-1)
            
            return {
                "smoothness": get_smoothness_metric(softmax_logits.cpu().numpy())["L2"]["mean"],
                "kl": self.dataset.get_kl_metric(softmax_logits),
                "multinomial": softmax_logits.cpu().numpy()[0].tolist()
            }

class ExperimentRunner:
    """Handles running experiments with different configurations."""
    
    def __init__(
        self,
        dataset: DistributionDataset,
        dim_input: int = 10,
        regularization_gamma: float = 1e-6,
        device: str = "cuda:0"
    ):
        self.trainer = FourierTrainer(dataset, dim_input, regularization_gamma, device)
        
    def run_frequency_sweep(
        self,
        max_frequencies: int,
        num_training_steps: int = 10000,
        output_path: str = "categorical_distributions.json"
    ) -> None:
        """Run experiments with increasing number of frequencies."""

        pdf = self.trainer.dataset.get_discretized_pdf_curve(1000).tolist()
        
        results = {"pdf" : pdf}
        for num_freqs in range(1, max_frequencies + 1):
            print(f"TRAINING num_freqs = {num_freqs}")
            metrics = self.trainer.train_model(
                num_frequencies=num_freqs,
                num_training_steps=num_training_steps
            )
            results[num_freqs] = metrics
        
            with open(output_path, "w") as fp:
                json.dump(results, fp, indent=4)

def run_exps_square_wave():

    square_wave_dataset = SquareWaveDataset(dim_output=128)
    square_wave_runner = ExperimentRunner(square_wave_dataset)
    square_wave_runner.run_frequency_sweep(
        num_training_steps=10000,
        max_frequencies=64, 
        output_path="output/data-square-wave.json"
    )

def run_exps_square_mixture_of_gaussians_v1():

    means_and_stds = [
        (-0.7, 0.1), 
        (-0.55, 0.07), 
        (-0.45, 0.2), 
        (-0.4, 0.15), 
        (-0.37, 0.05), 
        (-0.4, 0.2), 
        (0.1, 0.05),
        (0.45, 0.04), 
        (0.5, 0.1), 
        (0.6, 0.06)
    ]
    mog_dataset = MixtureOfGaussiansDataset(dim_output=128, means_and_stds=means_and_stds)
    mog_runner = ExperimentRunner(mog_dataset)
    mog_runner.run_frequency_sweep(
        num_training_steps=10000,
        max_frequencies=64, 
        output_path="output/data-mixture-of-gaussians-v1.json"
    )

def run_exps_square_mixture_of_gaussians_v2():

    means_and_stds = [
        (-0.98, 0.04), (-0.94, 0.07), (-0.92, 0.02),
        (-0.63, 0.05), (-0.6, 0.03), (-0.55, 0.06), (-0.53, 0.015), (-0.5, 0.08),
        (-0.45, 0.03), (-0.4, 0.06), (-0.38, 0.02), (-0.36, 0.08), (-0.34, 0.03),
        (-0.3, 0.06), (-0.25, 0.015), (-0.2, 0.05), (-0.15, 0.03), (-0.1, 0.08),
        (-0.05, 0.02), (0.0, 0.07), (0.05, 0.015), (0.1, 0.05), (0.15, 0.04),
        (0.2, 0.06), (0.25, 0.015), (0.3, 0.08), (0.35, 0.03), (0.4, 0.07),
        (0.82, 0.03), (0.85, 0.06), (0.9, 0.02), (0.92, 0.07), (0.95, 0.015)
    ]
    mog_dataset = MixtureOfGaussiansDataset(dim_output=128, means_and_stds=means_and_stds)
    mog_runner = ExperimentRunner(mog_dataset)
    mog_runner.run_frequency_sweep(
        num_training_steps=10000,
        max_frequencies=64, 
        output_path="output/data-mixture-of-gaussians-v2.json"
    )



def main():

    # EXAMPLE 1: square wave
    run_exps_square_wave()

    # EXAMPLE 2: mixture of gaussians, kind of simple
    run_exps_square_mixture_of_gaussians_v1()

    # EXAMPLE 3: mixture of gaussians, very complicated
    run_exps_square_mixture_of_gaussians_v2()


if __name__ == "__main__":
    main()