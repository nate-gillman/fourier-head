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

import torch
from torch import nn
from torch.nn.functional import conv1d


class Fourier_PDF(nn.Module):

    def __init__(self, fourier_coeffs_normalized):
        super().__init__()

        assert len(fourier_coeffs_normalized.shape) == 2 # (batch_size, num_frequencies)

        self.fourier_coeffs_normalized = fourier_coeffs_normalized
        self.num_frequencies = fourier_coeffs_normalized.shape[-1]

        self.device = self.fourier_coeffs_normalized.device
        self.frequencies = (1.0j * torch.arange(1, self.num_frequencies + 1) * torch.pi).to(self.device)

    def evaluate_at(self, batch):
        """
        Evaluate the Fourier probability density function at the given batch.

        Parameters:
        -----------
        batch : torch.Tensor, shape (batch_size)
            Values to evaluate the PDF at. On the forward pass through the Fourier head, this will 
            default to the bin centerpoints; we only include this as an optional parameter for cases
            where you might want more fine-grained control (e.g. visualization purposes).
        
        Returns:
        --------
        scaled_likelihood : torch.Tensor, shape (batch_size,)
            Evaluated PDF values at the batch
        """

        assert len(batch.shape) == 1 # (batch_size,)

        freqs = self.frequencies.expand((batch.shape[0], self.num_frequencies))
        # Evaluate the PDF at each bin centerpoint
        scaled_likelihood = 0.5 + (self.fourier_coeffs_normalized * torch.exp(batch.unsqueeze(1) * freqs)).sum(dim=1) # (bs)
        scaled_likelihood = scaled_likelihood.real
        # NOTE: at this point, every number should be real valued and non-negative, as this is the output from the PDF

        return scaled_likelihood


    def sample_from(self, n_samples):
        
        latent_samples_from_pdf = []
        for _ in range(n_samples):
            
            samples_uniform = torch.rand(self.fourier_coeffs_normalized.shape[0]).to(self.device) # (bs,), will apply inverse CDF to this

            # numerical inverse transform sampling over entire batch
            latent_samples_from_pdf_ = self.find_inverse(lambda x : self.evaluate_cdf(x), samples_uniform) # (bs,)
            latent_samples_from_pdf.append(latent_samples_from_pdf_)

        latent_samples_from_pdf = torch.stack(latent_samples_from_pdf).to(self.device).T # (bs, num_samples)

        return latent_samples_from_pdf


    def evaluate_cdf(self, batch):
        """
        I obtained this formula by integrating equation (4) in https://arxiv.org/pdf/2402.15345
        from -1 to x; then plugging in batch for x; details are 1/2 way thru NMM 45, in purple, Jan 5
        """

        # STEP 1: COMPUTE THE TERM FROM THE UPPER LIMIT OF INTEGRATION
        freqs = self.frequencies.expand((batch.shape[0], self.num_frequencies)) # (bs, n_freqs)
        upper_limit_complex = ( (self.fourier_coeffs_normalized * torch.exp(batch.view(-1,1) * freqs)) / freqs ).sum(dim=1) # (bs,)
        upper_limit = (batch / 2) + upper_limit_complex.real # (bs,)
        # every number here should be real, and non-negative!!

        # STEP 2: COMPUTE THE TERM FROM THE LOWER LIMIT OF INTEGRATION
        exponents = torch.arange(1, self.num_frequencies+1).expand((batch.shape[0], self.num_frequencies)).to(self.device) # (bs, n_freqs)
        lower_limit_complex = ( self.fourier_coeffs_normalized * ((-1)**exponents) / freqs).sum(dim=1) # (bs,)
        lower_limit = -0.5 + lower_limit_complex.real # (bs,)
        
        return upper_limit - lower_limit


    def find_inverse(self, f, y, tol=1e-6, max_iter=25):
        """
        Bisection method. Max_iter is fine-tuned so that tol is satisfied in the assert
        """
        
        lower = -1 * torch.ones_like(y).to("cuda:0")
        upper = torch.ones_like(y).to("cuda:0")  # Set an initial guess for the upper bound

        for _ in range(max_iter):

            mid = (lower + upper) / 2
            
            with torch.no_grad():
                f_mid = f(mid)
            lower_mask = f_mid < y

            # Update tensors based on conditions
            lower = torch.where(lower_mask, mid, lower)
            upper = torch.where(lower_mask, upper, mid)
        
        assert (f(upper)-f(lower)).abs().max().item() < tol

        return (lower + upper) / 2



class Fourier_Head_MLE(nn.Module):

    def __init__(self, 
            dim_input,
            num_frequencies,
            regularizion_gamma=0, 
            init_denominator_weight=100, 
            init_denominator_bias=100,
            device="cuda"
        ):
        """
        A PyTorch implementation of a Continuous Fourier Head, which inputs a vector, 
        uses a linear layer to learn the coefficients for a truncated Fourier 
        series over [-1,1], and either trains the Fourier head using MLE training, or
        samples from the resulting Fourier head.

        Attributes:
        -----------
        dim_input : int
            Dimension of the input vector.
        num_frequencies : int
            Number of Fourier frequencies to use in the Fourier series
        regularizion_gamma : float
            Coefficient for regularization term to penalize large high-order Fourier coefficients
        init_denominator_weight : float
            Initial scaling factor for the weight of the linear layer that extracts autocorrelation parameters.
        init_denominator_bias : float
            Initial scaling factor for the bias of the linear layer that extracts autocorrelation parameters.
        device : str
            Device to run the computations on ('cpu' or 'cuda').
        """
        super().__init__()

        # Store parameters
        self.dim_input = dim_input
        self.num_frequencies = num_frequencies
        self.regularizion_gamma = regularizion_gamma
        self.init_denominator_weight = init_denominator_weight
        self.init_denominator_bias = init_denominator_bias
        self.device = device

        # Linear layer for extracting autocorrelation parameters
        self.fc_extract_autocorrelation_params = nn.Linear(
            self.dim_input, 2*(self.num_frequencies+1)
        )

        # Weight and bias initialization
        self.fc_extract_autocorrelation_params.weight = nn.Parameter(
            self.fc_extract_autocorrelation_params.weight / self.init_denominator_weight
        )
        self.fc_extract_autocorrelation_params.bias = nn.Parameter(
            self.fc_extract_autocorrelation_params.bias / self.init_denominator_bias
        )

        # Regularization scalars to penalize high frequencies
        regularizion_scalars = torch.arange(0, self.num_frequencies+1).to(self.device)
        self.regularizion_scalars =  2 * (torch.pi ** 2) * (regularizion_scalars ** 2)


    def autocorrelate(self, sequence):
        """
        Compute the autocorrelation of the input sequence using 1D convolution.

        Parameters:
        -----------
        sequence : torch.Tensor
            Input sequence tensor, shape (batch_size, sequence_length).
        
        Returns:
        --------
        autocorr : torch.Tensor
            Autocorrelation of the input sequence, shape (batch_size, sequence_length)
        """

        batch, length = sequence.shape
        input = sequence[None, :, :] # Add a batch dimension
        weight = sequence[:, None, :].conj().resolve_conj()

        # Perform 1D convolution to compute autocorrelation
        autocorr = conv1d(
            input,
            weight,
            stride=(1,),
            padding=length-1,
            groups=batch
        )

        # Extract only the right-hand side of the symmetric autocorrelation
        autocorr = autocorr[0, :, length-1:]

        return autocorr


    def compute_fourier_coefficients(self, input):
        """
        Compute the Fourier coefficients for the input sequences.
        
        Parameters:
        -----------
        input : torch.Tensor
            Input tensor, shape (batch_size, input_dim).
        
        Returns:
        --------
        fourier_coeffs : torch.Tensor
            Computed Fourier coefficients for each input vector in the batch, 
            shape (batch_size, num_frequencies + 1)
        """

        # Compute autocorrelation parameters using the linear layer
        autocorrelation_params_all = self.fc_extract_autocorrelation_params(input) # (batch_size, dim_input) --> (batch_size, 2*(num_frequencies+1))

        # Combine the separate real and imaginary parts to obtain a single complex tensor
        autocorrelation_params = torch.complex(
            autocorrelation_params_all[..., 0:self.num_frequencies+1],
            autocorrelation_params_all[..., self.num_frequencies+1:2*(self.num_frequencies+1)]
        ) # (batch_size, num_frequencies+1)

        # Compute autocorrelation
        fourier_coeffs = self.autocorrelate(autocorrelation_params) # (batch_size, num_frequencies+1)

        return fourier_coeffs


    def forward(self, batch):
        """
        Forward pass of the Fourier head. Computes the Fourier coefficients, 
        evaluates the PDF at the bin centerpoints, and applies inverse softmax 
        transformation.
        
        Parameters:
        -----------
        batch : torch.Tensor
            Input tensor of shape (... , input_dim). NOTE: the code allows for arbitrarily many batch dimensions.

        Returns:
        --------
        fourier_pdf : Fourier_PDF
            A class representing a Fourier PDF (which lets us evaluate the PDF, sample from it, graph it, etc)
        """

        shape_input = batch.shape
        # shape_output = tuple(shape_input[:-1]) + (self.dim_output,)

        # In case there are multiple batch dimensions, we want to flatten them into a single dimension
        batch = batch.view(-1, batch.shape[-1]) # (batch_size, dim_input)

        # Compute Fourier coefficients
        fourier_coeffs = self.compute_fourier_coefficients(batch) # (batch_size, num_frequencies + 1)
        fourier_coeffs_normalized = fourier_coeffs[:, 1:] / fourier_coeffs[:, 0:1].real # (batch_size, num_frequencies)

        # Compute regularization loss, save for later
        regularization_summands = self.regularizion_scalars * torch.abs(fourier_coeffs)**2
        loss_regularization = 2 * torch.mean(torch.sum(regularization_summands, dim=-1)).to(self.device)
        self.loss_regularization = self.regularizion_gamma * loss_regularization

        fourier_pdf = Fourier_PDF(fourier_coeffs_normalized)
        return fourier_pdf