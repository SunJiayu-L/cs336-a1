import torch
import torch.nn as nn
class Silu(nn.Module):
    def __init__(self):
        """
        Constructs a SiLU activation layer.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SiLU activation function.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying SiLU activation.
        """
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        SwiGLU activation function.
        
        Args:
            d_model (int): Dimension of the input.
            d_ff (int): Dimension of the feed-forward layer.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Initialize as parameters that can be loaded via state_dict
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))
        
        self.silu = Silu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((Silu().forward(x@self.w1.T)) * (x@self.w3.T))@self.w2.T