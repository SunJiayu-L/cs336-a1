import numpy as np
import torch 
import torch.nn as nn

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

        w_init= self.init_weights(out_features, in_features, self.device, self.dtype)
        self.weight = nn.Parameter(w_init)  # PyTorch 专门用来**“声明这是一个可训练参数”**的类。

    def init_weights(self,out_dim:int, in_dim:int, device:torch.device, dtype:torch.dtype)->torch.Tensor:
        """
        Initializes the weights of the linear layer.
        """
        # torch.empty() 创建一个未初始化的张量（tensor），里面的值是随机的内存垃圾值。 
        W=torch.empty(out_dim, in_dim, device=device,dtype=dtype)
        mean=0
        std= np.sqrt(2/(out_dim+in_dim))
        nn.init.trunc_normal_(W, mean, std,-3*std, 3*std)
        return W

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
        # 正确的单字符下标格式
        return torch.einsum("...i, oi -> ...o", x, self.weight)

