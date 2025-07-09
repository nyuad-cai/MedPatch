import torch
from torch import nn
from einops import rearrange
import timm

import random

# Define the transformer and its components
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
        
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# EHR-specific transformer with CLS token
class EHRTransformer(nn.Module):
    def __init__(self, args, dim, depth, heads, mlp_dim, dropout=0., dim_head=64):
        super().__init__()
        self.dim = dim
        self.args=args
        if self.args.task == 'in-hospital-mortality':
            self.to_ehr_embedding = nn.Linear(48, dim)  # Assuming EHR input size of 48 as given
        else:
            self.to_ehr_embedding = nn.Linear(2442, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 77, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # CLS token initialization
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

    def forward(self, ehr):
        
        # print(f"Shape after permute: {ehr.shape}")  # Debugging print
    
        # # Check if the input dimensions match what the embedding layer expects
        # assert ehr.shape[1] == self.to_ehr_embedding.in_features, (
        #     f"Expected {self.to_ehr_embedding.in_features} features, "
        #     f"but got {ehr.shape[1]}"
        # )
        
        ehr = ehr.to(self.to_ehr_embedding.weight.device)
        ehr = ehr.permute(0, 2, 1)
        ehr = self.to_ehr_embedding(ehr)  # Embed raw EHR data
        b, n, _ = ehr.shape
        cls_tokens = self.cls_token.expand(b, -1, -1).to(ehr.device)  # Repeat CLS token for batch
        ehr = torch.cat((cls_tokens, ehr), dim=1)  # Prepend CLS token to embedded EHR
        ehr += self.pos_embedding[:, :(n + 1)]
        v_ehr = self.transformer(ehr)  # Pass through transformer
        cls = self.to_latent(v_ehr[:, 0])  # Extract the representation from the CLS token
        return v_ehr, cls