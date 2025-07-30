import torch
import torch.nn as nn
import numpy as np

def init_weights(out_dim: int, in_dim: int, device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
    """
    Initialize weights using truncated normal distribution.
    
    Args:
        out_dim (int): Output dimension (number of rows)
        in_dim (int): Input dimension (number of columns)  
        device (torch.device, optional): Device to use for the parameters. Defaults to None.
        dtype (torch.dtype, optional): Data type of the parameters. Defaults to None.
    
    Returns:
        torch.Tensor: Initialized weight tensor of shape (out_dim, in_dim)
    """
    W = torch.empty(out_dim, in_dim, device=device, dtype=dtype)
    mean = 0
    std = np.sqrt(2 / (out_dim + in_dim))
    nn.init.trunc_normal_(W, mean, std, -3*std, 3*std)
    return W
