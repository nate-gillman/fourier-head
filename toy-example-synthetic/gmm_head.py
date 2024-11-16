import torch
import torch.nn as nn
import math

class GMM_Head(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians):
        """
        Initialize the GMM Classification Head
        
        Args:
            input_dim (int): Dimension of input vectors (n)
            output_dim (int): Dimension of output vectors (m)
            num_gaussians (int): Number of Gaussians in the mixture
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians
        
        # Linear layer to learn the means of the Gaussians
        self.mean_layer = nn.Linear(input_dim, num_gaussians)
        
        # Precompute the evaluation points
        self.register_buffer(
            'eval_points',
            torch.tensor([-1 + (2*j + 1)/output_dim for j in range(output_dim)])
        )
        
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
        
        # Reshape tensors for broadcasting
        means = means.unsqueeze(-1)  # Shape: (batch_size, num_gaussians, 1)
        eval_points = self.eval_points.view(1, 1, -1)  # Shape: (1, 1, output_dim)
        
        # Compute Gaussian values
        # Using the formula: exp(-(x - μ)²/(2σ²))/(σ√(2π))
        diff = eval_points - means  # Shape: (batch_size, num_gaussians, output_dim)
        exponent = -1 * (diff**2) / (2 * (self.std**2))
        coefficient = 1 / (self.std * math.sqrt(2 * math.pi))
        gaussians = coefficient * torch.exp(exponent)
        
        # Sum over all Gaussians to get the mixture
        # Shape: (batch_size, output_dim)
        mixture_multinomial = gaussians.mean(dim=1)

        inverse_softmax_logits = torch.log(mixture_multinomial+1e-5) # (batch_size, dim_output)
        
        return inverse_softmax_logits

def test_gmm_head():
    """Test function to verify the GMMClassificationHead implementation"""
    # Create a sample instance
    batch_size = 3
    input_dim = 5
    output_dim = 10
    num_gaussians = 4
    
    model = GMM_Head(input_dim, output_dim, num_gaussians)
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output = model(x)
    
    # Check shapes
    assert output.shape == (batch_size, output_dim)
    # Check if outputs are positive (they should be, as they're Gaussian evaluations)
    assert torch.all(output > 0)
    
    print("All tests passed!")
    return output

if __name__ == "__main__":
    test_gmm_head()