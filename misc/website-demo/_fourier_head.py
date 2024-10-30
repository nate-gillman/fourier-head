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

import sys
sys.path.append("../..")

from fourier_head import Fourier_Head


class _Fourier_Head(Fourier_Head):
    """
    A version of the Fourier head class which only learns a single PDF.
    In other words, the forward pass doesn't look at the input value at all; it returns
    the discretization of the same exact Fourier PDF for every input.
    This Fourier PDF is still learned end-to-end, however.
    """

    def __init__(self, 
            dim_input,
            dim_output,
            num_frequencies,
            regularizion_gamma=0, 
            const_inverse_softmax=1e-5,
            init_denominator_weight=100, 
            init_denominator_bias=100,
            device="cuda"
        ):

        super().__init__(
            dim_input,
            dim_output,
            num_frequencies,
            regularizion_gamma=regularizion_gamma, 
            const_inverse_softmax=const_inverse_softmax,
            init_denominator_weight=init_denominator_weight, 
            init_denominator_bias=init_denominator_bias,
            device=device
        )

        self.autocorrelation_params = nn.Parameter(
            torch.randn(2*(self.num_frequencies+1)) / 10000
        )

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
        autocorrelation_params_all = self.autocorrelation_params # (2*(num_frequencies+1))

        # Combine the separate real and imaginary parts to obtain a single complex tensor
        autocorrelation_params = torch.complex(
            autocorrelation_params_all[0:self.num_frequencies+1],
            autocorrelation_params_all[self.num_frequencies+1:2*(self.num_frequencies+1)]
        ) # (num_frequencies+1)
        autocorrelation_params = autocorrelation_params.unsqueeze(0).repeat(input.shape[0], 1) # (bs, num_frequencies + 1)

        # Compute autocorrelation
        fourier_coeffs = self.autocorrelate(autocorrelation_params) # (bs, num_frequencies+1)

        return fourier_coeffs
