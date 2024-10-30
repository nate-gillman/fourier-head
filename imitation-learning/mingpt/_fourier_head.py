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
import os
import numpy as np

import math
torch.pi = math.pi # because this version of torch doesn't define pi...

import sys
for path in sys.path:
    if path.endswith("/imitation-learning"):
        sys.path.append(path.replace("/imitation-learning", "/"))

from fourier_head import Fourier_Head

class _Fourier_Head(Fourier_Head):

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

    def autocorrelate(self, sequence):
        """
        Compute the autocorrelation of the input sequence using 1D convolution.
        NOTE: this is mathematically equal to the original; but we overwrite
        it because the lower version of torch that Decision Transformer needs to run
        doesn't do the conv1d operation on complex numbers

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
        input = sequence[None, :, :]
        weight = sequence[:, None, :].conj()#.resolve_conj()

        # Decompose input into real and imaginary parts
        real_input = input.real
        imag_input = input.imag

        # Decompose weight into real and imaginary parts
        real_weight = weight.real
        imag_weight = weight.imag

        # Perform 1D convolution to compute autocorrelation
        # Perform convolutions for the real and imaginary parts
        real_output = conv1d(real_input, real_weight, stride=(1,), padding=length-1, groups=batch) - \
                    conv1d(imag_input, imag_weight, stride=(1,), padding=length-1, groups=batch)

        imag_output = conv1d(real_input, imag_weight, stride=(1,), padding=length-1, groups=batch) + \
                    conv1d(imag_input, real_weight, stride=(1,), padding=length-1, groups=batch)
        
        autocorr = torch.complex(real_output, imag_output)

        # Extract only the right-hand side of the symmetric autocorrelation
        autocorr = autocorr[0, :, length-1:]

        return autocorr