import torch
import torch.nn as nn
import numpy as np


"""
使用预计算实现方式，Rope是固定的，不可学习。只与token_position有关，与输入无关。
"""
class RoPE(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device=None):
        """
        Constructs a RoPE layer.
        Args:
            theta: float, the base frequency of the sine and cosine waves.
            d_k: int, the dimension of the model.
            max_seq_len: int, the maximum sequence length.
            device: torch.device, the device to use for the parameters.
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.rotation_matrix_table = self.create_rotation_matrix(theta,d_k,max_seq_len)
        self.register_buffer("rotation_matrix",self.rotation_matrix_table) #注册为缓冲区，不会参与梯度计算

    def create_rotation_matrix(self,theta:float,d_k:int,max_seq_len:int)->torch.Tensor:
        rotation_matrix_table = torch.zeros(max_seq_len,d_k,d_k) #有max_seq_len 个 d_k*d_k 的矩阵
        for i in range(max_seq_len): #对每一个位置，创建一个d_k*d_k 的矩阵
            blocks=[self.create_R_ik(theta,k,i,d_k) for k in range(d_k//2)] #对每一个位置，创建一个d_k//2 个 2x2 的矩阵,用列表存储
            rotation_matrix_table[i, :, :]=torch.block_diag(*blocks)  #选择 rotation_matrix_table 中第 i 个位置的整个 d_k x d_k 矩阵”。
        return rotation_matrix_table

    """
    # 1. 创建一个空列表来存放生成的旋转矩阵
    blocks = [] #

    # 2. 遍历从 0 到 d_k//2 - 1 的整数 k
    for k in range(d_k // 2):
    # 3. 为当前的 k 值调用 create_R_ik 函数
    rotation_block = self.create_R_ik(theta, k, i, d_k)
    
    # 4. 将生成的旋转矩阵添加到列表中
    blocks.append(rotation_block)
    """

    def create_R_ik(self, theta: float, block_index: int, seq_pos: int, d_k: int) -> torch.Tensor:
        """
        创建R_ik矩阵  2x2 的矩阵
        """
        theta_ik=torch.tensor(seq_pos/(theta**((2*block_index)/d_k)))
        R_ik = torch.tensor([[torch.cos(theta_ik),-torch.sin(theta_ik)],[torch.sin(theta_ik),torch.cos(theta_ik)]])
        return R_ik
        
    def forward(self, x: torch.Tensor, token_position: torch.Tensor) -> torch.Tensor:
        *prefix_dims, seq_len, d_k = x.shape
        rotation_matrix = self.rotation_matrix_table[token_position]  # [seq_len, d_k, d_k]
        # 使用torch.matmul进行批量矩阵乘法
        x_rotated = torch.matmul(rotation_matrix, x.unsqueeze(-1)).squeeze(-1)
        return x_rotated
    

    