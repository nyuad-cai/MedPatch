import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import timm

import random

class LSTM(nn.Module):

    def __init__(self, args, input_dim=76, output_dim=512, batch_first=True, dropout=0.0, layers=1):
        super(LSTM, self).__init__()
        self.args = args
        self.output_dim = args.output_dim
        self.layers =  layers
        input_dim = input_dim
        batch_first =  batch_first
        dropout =  dropout
        
        for layer in range(layers):
            setattr(self, f'layer{layer}', nn.LSTM(
                input_dim, output_dim,
                batch_first=batch_first,
                dropout = dropout)
            )
            input_dim = output_dim
        self.do = None
        if dropout > 0.0:
            self.do = nn.Dropout(dropout)
        self.feats_dim = output_dim
        self.full_feats_dim = output_dim
        self.initialize_weights()
        # self.activation = torch.sigmoid
    def initialize_weights(self):
        for model in self.modules():

            if type(model) in [nn.Linear]:
                nn.init.xavier_uniform_(model.weight)
                nn.init.zeros_(model.bias)
            elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
                nn.init.orthogonal_(model.weight_hh_l0)
                nn.init.xavier_uniform_(model.weight_ih_l0)
                nn.init.zeros_(model.bias_hh_l0)
                nn.init.zeros_(model.bias_ih_l0)

    def forward(self, x, seq_lengths):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        for layer in range(self.layers):
            x, (ht, _) = getattr(self, f'layer{layer}')(x)
            
        if 'c-' in self.args.fusion_type:
            # Unpack the sequence to get embeddings for all time steps
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            if self.do is not None:
                x = self.do(x)  # Apply dropout
            
        ehr_feats = ht.squeeze()
        if self.do is not None:
            ehr_feats = self.do(ehr_feats)
        
        if 'c-' in self.args.fusion_type:
            return ehr_feats, x
            
        return ehr_feats
        
        
class PreNorm(nn.Module):
    def __init__(self, output_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(output_dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, output_dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
        
class Attention(nn.Module):
    def __init__(self, output_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == output_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(output_dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, output_dim),
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
    def __init__(self, output_dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(output_dim, Attention(output_dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(output_dim, FeedForward(output_dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# EHR-specific transformer with CLS token
class EHRTransformer(nn.Module):
    def __init__(self, args, output_dim, depth, heads, mlp_dim, dropout, dim_head):
        super().__init__()
        self.args = args
        # Default values with fallback to args
        output_dim = getattr(args, 'output_dim', 512)
        self.feats_dim = output_dim
        depth = getattr(args, 'depth', 4)
        heads = getattr(args, 'heads', 4)
        mlp_dim = getattr(args, 'mlp_dim', 768)
        dropout = getattr(args, 'dropout', 0.1)
        dim_head = getattr(args, 'dim_head', 128)
        # output_dim = args.output_dim
        # self.feats_dim = output_dim 
        # depth = args.depth
        # heads = args.heads
        # mlp_dim = args.mlp_dim
        # dropout = args.dropout
        # dim_head = args.dim_head

        input_dim = 48 if args.task == 'in-hospital-mortality' else 2442
        self.to_ehr_embedding = nn.Linear(input_dim, output_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 77, output_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, output_dim))
        self.transformer = Transformer(output_dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

    def forward(self, ehr):
        ehr = ehr.to(self.to_ehr_embedding.weight.device)
        ehr = ehr.permute(0, 2, 1)
        ehr = self.to_ehr_embedding(ehr)  # Embed raw EHR data
        b, n, _ = ehr.shape
        cls_tokens = self.cls_token.expand(b, -1, -1).to(ehr.device)  # Repeat CLS token for batch
        ehr = torch.cat((cls_tokens, ehr), dim=1)  # Prepend CLS token to embedded EHR
        ehr += self.pos_embedding[:, :(n + 1)]
        ehr_feats = self.transformer(ehr)  # Pass through transformer
        self.cls = self.to_latent(ehr_feats[:, 0])  # Extract the representation from the CLS token
        return ehr_feats
        
class LinearEHR(nn.Module):
    def __init__(self, args):
        super(LinearEHR, self).__init__()
        self.args = args
        if self.args.task == 'in-hospital-mortality':
            input_dim = 48
        else:
            input_dim = 2646  # Adjust based on your data
        output_dim = args.output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.feats_dim = output_dim
    def forward(self, ehr):
        ehr = ehr.to(self.linear.weight.device)
        ehr = ehr.permute(0, 2, 1)
        
        # print(ehr.shape)  # Debug the shape of input
        # print(self.linear.weight.shape)  # Debug the shape of weight matrix
        
        ehr_feats = self.linear(ehr)  # Embed raw EHR data
        return ehr_feats
        
        
class EHR_encoder(nn.Module):
    def __init__(self, args):
        super(EHR_encoder, self).__init__()
        self.args = args
        if args.ehr_encoder == 'lstm':
            self.ehr_encoder = LSTM(args)
        elif args.ehr_encoder == 'linear':
            self.ehr_encoder = LinearEHR(args)
        elif args.ehr_encoder == 'transformer':
            self.ehr_encoder = EHRTransformer(args)
        else:
            raise ValueError(f"Unknown ehr_encoder: {args.ehr_encoder}")
        self.feats_dim = self.ehr_encoder.feats_dim
        self.full_feats_dim = self.ehr_encoder.full_feats_dim

    def forward(self, x, seq_lengths):
        if isinstance(self.ehr_encoder, LSTM):
            return self.ehr_encoder(x, seq_lengths)
        return self.ehr_encoder(x)