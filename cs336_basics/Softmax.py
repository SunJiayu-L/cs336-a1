import torch
import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self,dim:int):
        super().__init__()
        self.dim = dim

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=x-torch.max(x,dim=self.dim,keepdim=True)[0]
        # 2. 计算指数
        exp_x = torch.exp(x)
        # 3. 计算softmax
        softmax = exp_x / torch.sum(exp_x, dim=self.dim, keepdim=True)
        return softmax