import torch
import torch.nn as nn
import numpy as np

#todo 有问题，等下一步调试
class MyRoPE(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device


    def create_R_ik(self, theta: float, k: int, token_position: int, d_k: int) -> torch.Tensor:
        """
        创建R_ik矩阵  2x2 的矩阵
        """
        theta_ik=torch.tensor(token_position/(theta**((2*k)/d_k)))
        R_ik = torch.tensor([[torch.cos(theta_ik),-torch.sin(theta_ik)],[torch.sin(theta_ik),torch.cos(theta_ik)]])
        return R_ik

    def create_rotation_matrix_table(self,token_position:int) -> torch.Tensor:
        rotation_matrix_table = torch.zeros(self.d_k,self.d_k)
        blocks=[self.create_R_ik(self.theta,k,token_position,self.d_k) for k in range(self.d_k//2)]
        rotation_matrix_table=torch.block_diag(*blocks)
        return rotation_matrix_table

    
    def forward(self, x: torch.Tensor, token_position: torch.Tensor) -> torch.Tensor:
        R_i=self.create_rotation_matrix_table(token_position)
        x_rotated = torch.matmul(R_i, x.unsqueeze(-1)).squeeze(-1)
        return x_rotated
