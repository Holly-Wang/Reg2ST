from turtle import forward
import torch.nn as nn
import math

class FeedForwardBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.linear_1 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.linear_2(x)
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.2) -> None:
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.dropout = dropout
        
        self.d_k = dim // heads
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def attention(self, query, key, value, mask=None, dropout=0.2):
        d_k = query.shape[-1]
        attention_weights = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
        attention_weights = attention_weights.softmax(dim=-1)
        if dropout > 0:
            attention_weights = self.dropout_layer(attention_weights)
        
        return (attention_weights @ value), attention_weights
    
    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        query = query.view(query.shape[0], self.heads, self.d_k).transpose(0, 1)
        key = key.view(key.shape[0], self.heads, self.d_k).transpose(0, 1)
        value = value.view(value.shape[0], self.heads, self.d_k).transpose(0, 1)
        
        x, self.attention_weights = self.attention(query, key, value, mask, dropout=self.dropout)
        
        # print(x.shape)
        x = x.contiguous().view(x.shape[1], -1)
        # print(x.shape)
        return self.w_o(x)
        
class ResidualConnection(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
        
    def forward(self, x, sublayer):
        # print(x is None)
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.2):
        super().__init__()
        self.attention_block = MultiHeadAttentionBlock(dim, heads, dropout)
        self.feed_forward_block = FeedForwardBlock(dim, mlp_dim, dropout)
        self.residual_block = nn.ModuleList([ResidualConnection(dim, dropout) for _ in range(2)])
        
    def forward(self, x, mask=None):
        x = self.residual_block[0](x, lambda x: self.attention_block(x, x, x, mask))
        x = self.residual_block[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, dim, layers, heads, mlp_dim, dropout) -> None:
        super().__init__()
        encoder = []
        for i in range (layers):
            encoder.append(EncoderBlock(dim, heads, mlp_dim, dropout))
        # self.layers = layers
        self.layers = nn.ModuleList(encoder)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__ (self, dim, heads, mlp_dim, dropout=0.2) -> None:
        super().__init__()
        self.attention_block_1 = MultiHeadAttentionBlock(dim, heads, dropout)
        self.cross_attention_block = MultiHeadAttentionBlock(dim, heads, dropout)
        self.feed_forward_block = FeedForwardBlock(dim, mlp_dim, dropout)
        self.residual_block = nn.ModuleList([ResidualConnection(dim, dropout) for _ in range(2)])
        
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # x = self.residual_block[0](x, lambda x: self.attention_block_1(x, x, x, tgt_mask))
        x = self.residual_block[0](x, lambda x: self.cross_attention_block(x, memory, memory, src_mask))
        x = self.residual_block[1](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, dim, layers, heads, mlp_dim, dropout=0.2) -> None:
        super().__init__()
        decoder = []
        for i in range (layers):
            decoder.append(DecoderBlock(dim, heads, mlp_dim, dropout))
        
        self.layers = nn.ModuleList(decoder)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)