import torch
import torch.nn as nn

class MyAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MyAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        x, _ = self.multihead_attention(x, x, x)
        return x
