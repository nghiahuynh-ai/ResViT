import torch
import math
import torch.nn as nn
from einops import rearrange


def build_bottleneck(bottleneck_cfg):

    bottleneck = Bottleneck(
        feat_in=bottleneck_cfg.feat_in,
        feat_out=bottleneck_cfg.feat_out,
        n_layers=bottleneck_cfg.n_layers,
        d_model=bottleneck_cfg.d_model,
        n_heads=bottleneck_cfg.n_heads,
        ff_expansion_factor=bottleneck_cfg.ff_expansion_factor,
        patch_size=bottleneck_cfg.patch_size,
        dropout=bottleneck_cfg.dropout,
    )

    return bottleneck


class Bottleneck(nn.Module):
    
    def __init__(
        self, 
        feat_in, 
        feat_out, 
        n_layers, 
        d_model, 
        n_heads, 
        ff_expansion_factor, 
        patch_size=3, 
        dropout=0.1
        ):
        super(Bottleneck, self).__init__()
        
        self.patch_size = patch_size
        
        self.linear_in = nn.Sequential(
            nn.LayerNorm(patch_size**2 * feat_in),
            nn.Linear(patch_size**2 * feat_in, d_model),
            nn.LayerNorm(d_model),
        )
        
        self.pos_emb = PositionalEncoding(d_model)
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=n_heads, 
                dim_feedforward=ff_expansion_factor * d_model,
                dropout=dropout,
                ),
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )
        
        self.linear_out = nn.Sequential(
            nn.Linear(d_model, patch_size**2 * feat_out),
            nn.LayerNorm(patch_size**2 * feat_out),
        )
        
    def forward(self, x):
        # X: (b, c, h, w) -> (b, l, d) -> (b, c, h, w)
        
        _, c, h, _ = x.shape
        
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x = self.linear_in(x)
        x = self.pos_emb(x)
        x = self.encoder(x)
        x = self.linear_out(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h//self.patch_size, p1=self.patch_size, c=c)
        
        return x


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x.permute(1, 0, 2) # (batch, seq_len, dim) -> (seq_len, batch, dim)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x.permute(1, 0, 2)