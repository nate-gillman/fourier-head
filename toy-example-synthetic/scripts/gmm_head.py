import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class GMM_Head(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians, learn_stds=True):
        """
        Initialize the GMM Classification Head
        
        Args:
            input_dim (int): Dimension of input vectors (n)
            output_dim (int): Dimension of output vectors (m)
            num_gaussians (int): Number of Gaussians in the mixture
            learn_stds (bool): Whether to learn the standard deviations
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians
        self.learn_stds = learn_stds
        
        # Linear layer to learn the means of the Gaussians
        self.mean_layer = nn.Linear(input_dim, num_gaussians)
        
        # Precompute the evaluation points
        self.register_buffer(
            'eval_points',
            torch.tensor([-1 + (2*j + 1)/output_dim for j in range(output_dim)])
        )
        
        if learn_stds:
            # Initialize learnable parameters for standard deviations
            # Initialize to produce std ≈ 2.0/output_dim after softplus
            init_value = math.log(math.exp(2.0/output_dim) - 1)
            self.log_stds = nn.Parameter(torch.full((num_gaussians,), init_value))
        else:
            # Fixed standard deviation
            self.register_buffer('std', torch.tensor(2.0 / output_dim))

    def forward(self, x):
        """
        Forward pass of the GMM Classification Head
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Get batch size
        batch_size = x.shape[0]
        
        # Compute and normalize the means using tanh
        means = torch.tanh(self.mean_layer(x))  # Shape: (batch_size, num_gaussians)
        
        # Get standard deviations
        if self.learn_stds:
            # Use softplus to ensure positive standard deviations
            stds = F.softplus(self.log_stds)  # Shape: (num_gaussians,)
            # Reshape for broadcasting
            stds = stds.view(1, -1, 1)  # Shape: (1, num_gaussians, 1)
        else:
            stds = self.std
        
        # Reshape tensors for broadcasting
        means = means.unsqueeze(-1)  # Shape: (batch_size, num_gaussians, 1)
        eval_points = self.eval_points.view(1, 1, -1)  # Shape: (1, 1, output_dim)
        
        # Compute Gaussian values
        # Using the formula: exp(-(x - μ)²/(2σ²))/(σ√(2π))
        diff = eval_points - means  # Shape: (batch_size, num_gaussians, output_dim)
        exponent = -1 * (diff**2) / (2 * (stds**2))
        coefficient = 1 / (stds * math.sqrt(2 * math.pi))
        gaussians = coefficient * torch.exp(exponent)
        
        # Sum over all Gaussians to get the mixture
        # Shape: (batch_size, output_dim)
        mixture_multinomial = gaussians.mean(dim=1)

        inverse_softmax_logits = torch.log(mixture_multinomial + 1e-5)  # (batch_size, dim_output)
        
        return inverse_softmax_logits

def test_gmm_head():
    """Test function to verify the GMMClassificationHead implementation"""
    # Create sample instances with both fixed and learnable stds
    batch_size = 3
    input_dim = 5
    output_dim = 10
    num_gaussians = 4
    
    # Test with fixed stds
    model_fixed = GMM_Head(input_dim, output_dim, num_gaussians, learn_stds=False)
    # Test with learnable stds
    model_learnable = GMM_Head(input_dim, output_dim, num_gaussians, learn_stds=True)
    
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass for both models
    output_fixed = model_fixed(x)
    output_learnable = model_learnable(x)
    
    # Check shapes
    assert output_fixed.shape == (batch_size, output_dim)
    assert output_learnable.shape == (batch_size, output_dim)
    
    if model_learnable.learn_stds:
        # Check if stds are being learned
        assert isinstance(model_learnable.log_stds, nn.Parameter)
        # Check if stds are positive after softplus
        assert torch.all(F.softplus(model_learnable.log_stds) > 0)
    
    print("All tests passed!")
    return output_fixed, output_learnable

if __name__ == "__main__":
    test_gmm_head()