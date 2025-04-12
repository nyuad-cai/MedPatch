import torch
from torch import nn

class CustomTransformerLayer(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_layers=1, dropout=0.1):
        super(CustomTransformerLayer, self).__init__()
        #self.input_proj = nn.Linear(input_dim, model_dim)
        
        #max_len = 16600
        max_len = 5269
        
        # Initialize the CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        # Initialize positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, model_dim))
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=nhead, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        #self.output_proj = nn.Linear(model_dim, model_dim)

    def forward(self, src):
        b, n, _ = src.shape  # Batch size, sequence length, input dimension

        # Prepend the CLS token to the input sequence
        cls_tokens = self.cls_token.expand(b, -1, -1)
        src = torch.cat((cls_tokens, src), dim=1)
        
         # Ensure that positional embeddings are expanded or repeated if necessary
        if n + 1 > self.pos_embedding.size(1):
            # Expand positional embeddings if src is longer
            repeat_factor = (n + 1) // self.pos_embedding.size(1) + 1
            expanded_pos_embedding = self.pos_embedding.repeat(1, repeat_factor, 1)
            pos_embedding = expanded_pos_embedding[:, :n + 1, :]
        else:
            # Use the existing positional embeddings
            pos_embedding = self.pos_embedding[:, :n + 1, :]

        # Add positional embeddings
        src += pos_embedding

        # Since nn.Transformer expects (S, N, E) format, we permute src to fit this
        src = src.permute(1, 0, 2)  # (N, S, E) -> (S, N, E)
        
        # Pass through the transformer
        src = self.transformer_encoder(src)
        
        # Permute back to original format
        src = src.permute(1, 0, 2)  # (S, N, E) -> (N, S, E)
        
        return src