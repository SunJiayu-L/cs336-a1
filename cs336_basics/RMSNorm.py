import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self,d_model:int ,eps:float=1e-5,device=None,dtype=None):
        """
            Constructs a RMSNorm layer.
        Args:
            d_model (int): Hidden dimension of the model.
            eps (float): 1e-5 Episilon value for numerical stability.
            device (torch.device): Device to use for the parameters. Defaults to None.
            dtype (torch.dtype): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype            
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype)) #gain parameter
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # Compute the RMS normalization
        in_dtype = x.dtype
        x=x.to(torch.float32)
        # 计算均方根（Root Mean Square）范数：对张量x的每个元素进行平方，然后在最后一个维度上计算均值，
        # 加上一个小的常数eps以保证数值稳定性，最后取平方根得到norm
        norm = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x / norm
        result = result * self.weight
        return result.to(in_dtype)