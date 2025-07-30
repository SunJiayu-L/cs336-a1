import numpy as np
import torch 
import torch.nn as nn
from .weight_init import init_weights

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 device: torch.device = None, dtype: torch.dtype = None):
        """
        Constructs a linear layer.
        Args:
            in_features (int): final dimension of input features.
            out_features (int): final dimension  of output features.
            device (torch.device, optional): Device to use for the parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        w_init = init_weights(out_features, in_features, self.device, self.dtype)
        self.weight = nn.Parameter(w_init)

    def forward (self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear layer.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying the linear transformation.
        """
        # 使用 PyTorch 的 einsum
        # 权重形状: (d_out, d_in) -> (o, i)
        # 输入形状: (..., d_in) -> (..., i)  
        # 输出形状: (..., d_out) -> (..., o)
        return torch.einsum("...i, oi -> ...o", x, self.weight)

# 测试代码
if __name__ == "__main__":
    # 创建线性层实例
    liner = Linear(10, 5)  # 输入10个特征，输出5个特征
    print("Linear layer created successfully!")
    print(f"Weight shape: {liner.weight.shape}")
    
    # 测试前向传播
    x = torch.randn(3, 10)  # 批次大小3，输入特征10
    output = liner(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Forward pass completed successfully!")
