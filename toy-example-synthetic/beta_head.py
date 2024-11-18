import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Beta_Head(nn.Module):
    def __init__(self, input_dim, output_dim, num_betas):
        """
        Initialize the Beta Classification Head
        
        Args:
            input_dim (int): Dimension of input vectors (n)
            output_dim (int): Dimension of output vectors (m)
            num_betas (int): Number of Beta distributions in the mixture
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_betas = num_betas
        
        # Linear layers to learn parameters
        self.midpoint_layer = nn.Linear(input_dim, num_betas)
        self.width_layer = nn.Linear(input_dim, num_betas)
        self.alpha_layer = nn.Linear(input_dim, num_betas)
        self.beta_layer = nn.Linear(input_dim, num_betas)
        
        # Initialize midpoints to 0.0 (centered) and widths to 1.0
        self.midpoint_layer.weight.data.zero_()
        self.midpoint_layer.bias.data.zero_()
        
        width_bias = math.log(math.exp(1.0) - 1)  # Initialize to output 1.0 after softplus
        self.width_layer.weight.data.zero_()
        self.width_layer.bias.data.fill_(width_bias)
        
        # Initialize alpha and beta to output 2.0 after softplus (symmetric bell shape)
        shape_bias = math.log(math.exp(2.0) - 1)
        self.alpha_layer.weight.data.zero_()
        self.alpha_layer.bias.data.fill_(shape_bias)
        self.beta_layer.weight.data.zero_()
        self.beta_layer.bias.data.fill_(shape_bias)
        
        # Precompute the evaluation points in [-1, 1]
        self.register_buffer(
            'eval_points',
            torch.tensor([-1 + (2*j + 1)/output_dim for j in range(output_dim)])
        )

    def forward(self, x):
        """
        Forward pass of the Beta Classification Head
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        
        # Get parameters for each beta distribution
        midpoints = torch.tanh(self.midpoint_layer(x))     # center point in [-1, 1]
        widths = F.softplus(self.width_layer(x))           # positive width
        alphas = F.softplus(self.alpha_layer(x))           # positive shape parameter
        betas = F.softplus(self.beta_layer(x))             # positive shape parameter
        
        # Reshape for broadcasting
        midpoints = midpoints.unsqueeze(-1)  # Shape: (batch_size, num_betas, 1)
        widths = widths.unsqueeze(-1)        # Shape: (batch_size, num_betas, 1)
        alphas = alphas.unsqueeze(-1)        # Shape: (batch_size, num_betas, 1)
        betas = betas.unsqueeze(-1)          # Shape: (batch_size, num_betas, 1)
        eval_points = self.eval_points.view(1, 1, -1)  # Shape: (1, 1, output_dim)
        
        # Scale the input to beta PDF from [midpoint-width/2, midpoint+width/2] to [0,1]
        scaled_points = (eval_points - (midpoints - widths/2)) / widths
        
        # Compute Beta PDF for points in valid range, 0 otherwise
        valid_mask = (eval_points >= (midpoints - widths/2)) & (eval_points <= (midpoints + widths/2))
        
        # For valid points, compute beta PDF and scale by 1/width
        log_beta = (
            (alphas - 1) * torch.log(scaled_points + 1e-6) +
            (betas - 1) * torch.log(1 - scaled_points + 1e-6) -
            torch.lgamma(alphas) - torch.lgamma(betas) + torch.lgamma(alphas + betas)
        )
        beta_pdf = torch.exp(log_beta) / widths
        
        # Zero out invalid points
        beta_pdf = torch.where(valid_mask, beta_pdf, torch.zeros_like(beta_pdf))
        
        # Average over all beta distributions
        mixture_multinomial = beta_pdf.mean(dim=1)
        
        # Convert to logits
        inverse_softmax_logits = torch.log(mixture_multinomial + 1e-5)
        
        return inverse_softmax_logits

def test_beta_head():
    """Test function to verify the Beta Classification Head implementation"""
    batch_size = 32
    input_dim = 5
    output_dim = 10
    num_betas = 2
    
    model = Beta_Head(input_dim, output_dim, num_betas)
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output = model(x)
    
    # Check shapes
    assert output.shape == (batch_size, output_dim)
    
    print("All tests passed!")
    return output

if __name__ == "__main__":
    test_beta_head()