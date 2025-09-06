import torch
from einops import einsum, rearrange
import math
from cs336_basics.Softmax import Softmax


def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of scaled dot product attention.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # 获取键向量的维度，用于缩放注意力分数
    d_k = K.shape[-1]
    
    # 计算查询和键之间的点积注意力分数
    # einsum: 对Q和K进行矩阵乘法，结果形状为 [... queries keys]
    # 除以sqrt(d_k)进行缩放，防止梯度消失
    attention_scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k) 

    # 如果提供了mask，将mask为0的位置的注意力分数设为负无穷
    # 这样在softmax后这些位置的概率会变为0
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
    
    # 对注意力分数应用softmax，得到注意力权重（概率分布）
    # dim=-1表示在最后一个维度（keys维度）上应用softmax
    attention_probs = Softmax(dim=-1).forward(attention_scores)
    
    # 使用注意力权重对值向量进行加权求和
    # einsum: attention_probs [queries keys] @ V [keys d_v] -> output [queries d_v]
    output = einsum(attention_probs, V, "... queries keys, ... keys d_v -> ... queries d_v")
    
    # 返回加权后的值向量，形状为 [... queries d_v]
    return output 

