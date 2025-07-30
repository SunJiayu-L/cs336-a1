import torch
import torch.nn as nn
import numpy as np
from .weight_init import init_weights

class Embedding(nn.Module):
    def __init__(self,num_embeddings:int, embedding_dim:int,device=None,dtype=None):
        """
        Constructs an embedding layer.
        
        Args:
            num_embeddings (int): size of vocabulary.(词汇表中最大词)
            embedding_dim (int): Dimension of the embedding vectors.    （嵌入向量的维度）
            device (torch.device, optional): Device to use for the parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        # Initialize the embedding weights
        w_init = init_weights(num_embeddings, embedding_dim, device, dtype)
        self.weight = nn.Parameter(w_init)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
