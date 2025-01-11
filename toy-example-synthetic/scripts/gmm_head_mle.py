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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

class GMM_PDF(nn.Module):
    """
    A 1D Gaussian Mixture Model (GMM) PDF over [-1,1] with num_gaussians components.
    """

    def __init__(self, means, stds, mixture_weights):
        """
        Args:
            means (torch.Tensor): shape (batch_size, num_gaussians)
            stds (torch.Tensor): shape (batch_size, num_gaussians)
            mixture_weights (torch.Tensor): shape (batch_size, num_gaussians)
                Each row sums to 1.
        """
        super().__init__()
        self.means = means               # (batch_size, num_gaussians)
        self.stds = stds                 # (batch_size, num_gaussians)
        self.mixture_weights = mixture_weights  # (batch_size, num_gaussians)

        # Dimensions
        self.batch_size, self.num_gaussians = means.shape

    def evaluate_at(self, x):
        """
        Evaluate the mixture PDF at given points x in [-1, 1].

        Args:
            x (torch.Tensor): shape (batch_size,) or (batch_size, M).
                Each row corresponds to one batch item.

        Returns:
            pdf_values (torch.Tensor): same shape as x
                The GMM's PDF value at each x.
        """
        # Ensure x is on the same device
        device = self.means.device
        x = x.to(device)

        if x.ndim == 1:
            # shape: (batch_size,) => (batch_size, 1)
            x = x.unsqueeze(1)
        else:
            # shape: (batch_size, M) => (batch_size, 1, M)
            x = x.unsqueeze(1)

        # Evaluate each Gaussian’s pdf via torch.distributions.Normal
        # means, stds: (batch_size, num_gaussians)
        # broadcast so that final shape => (batch_size, num_gaussians, M)
        dist = Normal(self.means.unsqueeze(-1), self.stds.unsqueeze(-1))  # shape broadcast
        log_probs = dist.log_prob(x) # shape: (batch_size, num_gaussians, M)
        pdf_values_per_gaussian = torch.exp(log_probs)

        # Weight each Gaussian by mixture_weights
        weighted_pdfs = pdf_values_per_gaussian * self.mixture_weights.unsqueeze(-1)

        # Sum across gaussians
        # result => (batch_size, M)
        pdf_values = weighted_pdfs.sum(dim=1)

        # Squeeze out the trailing dimension if M=1
        return pdf_values.squeeze(-1)

    def sample_from(self, num_samples=1):
        """
        Draw samples from each batch element's GMM, using:
          1. Categorical to choose a Gaussian index
          2. Reparameterization for each chosen Gaussian
        Returns:
            samples (torch.Tensor): shape (batch_size, num_samples)
        """
        device = self.means.device
        batch_size = self.batch_size

        # 1) Build a categorical distribution over the mixture weights
        cat_dist = Categorical(self.mixture_weights)

        # 2) Sample which component each of the (batch_size x num_samples) points comes from
        comp_indices = cat_dist.sample((num_samples,)).T  # => (batch_size, num_samples)

        # 3) For each (batch, sample), gather the appropriate mean/std
        selected_means = self.means.gather(dim=1, index=comp_indices)
        selected_stds = self.stds.gather(dim=1, index=comp_indices)

        # 4) Reparameterization trick: z ~ Normal(0,1) => x = mean + std*z
        eps = torch.randn(batch_size, num_samples, device=device)
        samples = selected_means + selected_stds * eps
        
        return samples


class GMM_Head_MLE(nn.Module):

    def __init__(
        self,
        input_dim,
        num_gaussians,
        learn_stds=True,
        mixture_weights_learned=True,
        init_std=2.0,
        device="cpu"
    ):
        """
         A PyTorch implementation of a Continuous Fourier Head, which inputs a vector, and uses a linear layer 
         to output the parameters of a 1D Gaussian Mixture Model (means, stds, mixture weights).

        Args:
            input_dim (int): Dimension of the input vectors
            num_gaussians (int): Number of Gaussians in the mixture
            learn_stds (bool): Whether standard deviations are learnable
            mixture_weights_learned (bool): Whether mixture weights are learned (instead of uniform)
            init_std (float): A base scale factor for std initialization
            device (str): "cpu" or "cuda"
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_gaussians = num_gaussians
        self.learn_stds = learn_stds
        self.mixture_weights_learned = mixture_weights_learned
        self.device = device

        # 1) Means layer
        self.means_layer = nn.Linear(input_dim, num_gaussians)

        # 2) Standard deviations
        if learn_stds:
            # One std per Gaussian, so size = (num_gaussians,)
            self.log_stds = nn.Parameter(torch.full((num_gaussians,), math.log(init_std)))
        else:
            # Fixed std
            self.register_buffer(
                "fixed_std",
                torch.tensor(init_std, device=device, dtype=torch.float),
            )

        # 3) Mixture weights
        if mixture_weights_learned:
            # Linear layer to learn weights
            self.weights_layer = nn.Linear(input_dim, num_gaussians)
        else:
            self.weights_layer = None  # just do uniform in forward()

        # Move to device if needed
        self.to(device)

    def forward(self, x):
        """
        Forward pass:
         1) Compute mixture means, stds, and mixture weights from x
         2) Return a GMM_PDF object with those parameters

        Args:
            x (torch.Tensor): shape (batch_size, input_dim)
        
        Returns:
            GMM_PDF
        """
        batch_size = x.shape[0]
        device = x.device

        # Means in [-1,1] with tanh (if desired):
        raw_means = self.means_layer(x)  # (batch_size, num_gaussians)
        means = torch.tanh(raw_means)

        # Stds
        if self.learn_stds:
            # apply softplus or exp to ensure positivity
            # shape => (num_gaussians,) => broadcast to (batch_size, num_gaussians)
            std_values = F.softplus(self.log_stds)
            stds = std_values.unsqueeze(0).expand(batch_size, -1)
        else:
            # shape => (batch_size, num_gaussians)
            stds = self.fixed_std.unsqueeze(0).expand(batch_size, self.num_gaussians)

        # Mixture weights
        if self.mixture_weights_learned:
            raw_weights = self.weights_layer(x)  # (batch_size, num_gaussians)
            # use softmax to ensure they sum to 1 along dim=-1
            mixture_weights = F.softmax(raw_weights, dim=-1)
        else:
            # uniform
            mixture_weights = means.new_ones(batch_size, self.num_gaussians)
            mixture_weights = mixture_weights / float(self.num_gaussians)

        # Build the PDF object
        gmm_pdf = GMM_PDF(means, stds, mixture_weights)
        return gmm_pdf
