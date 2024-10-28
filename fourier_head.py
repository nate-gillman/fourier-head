import torch
from torch import nn
from torch.nn.functional import conv1d

class Fourier_Head(nn.Module):

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
        """
        A PyTorch implementation of the Fourier Head, which inputs a vector, 
        uses a linear layer to learn the coefficients for a truncated Fourier 
        series over [-1,1], evaluates the learned Fourier PDF at those m bin 
        center points in that interval, and returns (the inverse softmax of) 
        those m likelihoods as a categorical distribution. 

        Attributes:
        -----------
        dim_input : int
            Dimension of the input vector.
        dim_output : int
            Number of output bins (dimensionality of the categorical distribution).
        num_frequencies : int
            Number of Fourier frequencies to use in the Fourier series
        regularizion_gamma : float
            Coefficient for regularization term to penalize large high-order Fourier coefficients
        const_inverse_softmax : float
            Constant added to avoid taking log(0) in the inverse softmax transformation.
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
        self.dim_output = dim_output
        self.num_frequencies = num_frequencies
        self.regularizion_gamma = regularizion_gamma
        self.const_inverse_softmax = const_inverse_softmax
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

        # Precompute Fourier frequencies
        self.frequencies = (1.0j * torch.arange(1, self.num_frequencies + 1) * torch.pi).to(self.device)

        # Regularization scalars to penalize high frequencies
        regularizion_scalars = torch.arange(0, self.num_frequencies+1).to(self.device)
        self.regularizion_scalars =  2 * (torch.pi ** 2) * (regularizion_scalars ** 2)

        # Bin centerpoints used to evaluate the PMF
        bin_edges = torch.linspace(-1, 1, self.dim_output + 1)
        bin_centerpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.bin_centerpoints = bin_centerpoints.to(device)


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

    def evaluate_pdf(self, fourier_coeffs):
        """
        Evaluate the probability density function (PDF) at the precomputed bin 
        centerpoints for given Fourier coefficients.

        Parameters:
        -----------
        fourier_coeffs : torch.Tensor
            Normalized fourier coefficients for the input batch, shape (batch_size, num_frequencies)
        
        Returns:
        --------
        scaled_likelihood : torch.Tensor
            Evaluated PDF values at the centerpoints of each bin, shape (batch_size, dim_output)
        """

        freqs = self.frequencies.expand((self.bin_centerpoints.shape[0], self.num_frequencies))

        # Evaluate the PDF at each bin centerpoint
        scaled_likelihood = 0.5 + (fourier_coeffs.unsqueeze(1) * torch.exp(self.bin_centerpoints.unsqueeze(1) * freqs).unsqueeze(0)).sum(dim=-1)
        scaled_likelihood = scaled_likelihood.real
        # NOTE: at this point, every number should be real valued and non-negative, 
        # as this is the output from the PDF

        return scaled_likelihood

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
        inverse_softmax_logits : torch.Tensor
            Logarithmic likelihoods (inverse softmax) for the predicted PMF.
        """

        shape_input = batch.shape
        shape_output = tuple(shape_input[:-1]) + (self.dim_output,)

        # In case there are multiple batch dimensions, we want to flatten them into a single dimension
        batch = batch.view(-1, batch.shape[-1]) # (batch_size, dim_input)

        # Compute Fourier coefficients
        fourier_coeffs = self.compute_fourier_coefficients(batch) # (batch_size, num_frequencies + 1)
        fourier_coeffs_normalized = fourier_coeffs[:, 1:] / fourier_coeffs[:, 0:1].real # (batch_size, num_frequencies)

        # Evaluate PDF at bin centerpoints
        scaled_likelihood = self.evaluate_pdf(fourier_coeffs_normalized) # (batch_size, dim_output)

        # Normalize to get categorical distributions
        normalization_constant = scaled_likelihood.sum(dim=-1).unsqueeze(-1) # (batch_size, 1); in expectation, should be uniformly m/2
        categorical_distributions = scaled_likelihood / normalization_constant # (batch_size, dim_output)
        
        # NOTE: at this point, we have assert(sum(categorical_distributions[..., i]) == 1 for every i)

        # In case there are multiple batch dimensions, this un-flattens them
        categorical_distributions = categorical_distributions.view(shape_output)

        # Apply inverse softmax transformation
        inverse_softmax_logits = torch.log(categorical_distributions+self.const_inverse_softmax) # (batch_size, dim_output)

        # Compute regularization loss, save for later
        regularization_summands = self.regularizion_scalars * torch.abs(fourier_coeffs)**2
        loss_regularization = 2 * torch.mean(torch.sum(regularization_summands, dim=-1)).to(self.device)
        self.loss_regularization = self.regularizion_gamma * loss_regularization

        return inverse_softmax_logits