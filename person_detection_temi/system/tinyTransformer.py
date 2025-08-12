import torch
import numpy as np
import torch.nn as nn


def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoidal positional encoding: shape (1, n_position, d_hid)"""
    position = torch.arange(n_position, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2, dtype=torch.float32) * -(np.log(10000.0) / d_hid))
    
    pe = torch.zeros(n_position, d_hid)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # shape: (1, n_position, d_hid)

class SingleTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, batch_first = True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, key_padding_mask=None):
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src, attn_weights

class TinyTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=2, num_layers=2, seq_len=7, num_classes=1, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.register_buffer("pos_embed", get_sinusoid_encoding(seq_len + 1, d_model))

        # Stack of custom encoder layers
        self.encoder_layers = nn.ModuleList([
            SingleTransformerEncoderLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])

        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )


    def forward(self, x, visibility_mask=None, return_mask=False):
        B = x.size(0)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # x: [B, seq_len+1, d_model]

        # Positional encoding
        x = x + self.pos_embed[:, :x.size(1), :]

        attention_maps = []

        # Process visibility mask → invert to key_padding_mask
        key_padding_mask = None
        if visibility_mask is not None:
            # visibility_mask: [B, seq_len] with True = visible
            # Pad with True for CLS token
            cls_visible = torch.ones((B, 1), dtype=visibility_mask.dtype, device=visibility_mask.device)
            visibility_mask = torch.cat([cls_visible, visibility_mask], dim=1)  # shape: [B, seq_len+1]
            key_padding_mask = ~visibility_mask  # invert: True = mask

        for layer in self.encoder_layers:
            x, attn = layer(x, key_padding_mask=key_padding_mask)
            attention_maps.append(attn)

        cls_output = x[:, 0]

        if return_mask:
            return self.mlp_head(cls_output), attention_maps
        else:
            return self.mlp_head(cls_output)


    def save_weights(self, path):
        """Save model weights to a file."""
        torch.save(self.state_dict(), path)

    def load_weights(self, path, map_location=None):
        """Load model weights from a file."""
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.eval()  # Optional: switch to eval mode


# ---- Model ----
# class TinyTransformer(nn.Module):
#     def __init__(self, d_model=512, nhead=2, num_layers=2, seq_len=7, num_classes=1, dropout = 0.0):
#         super().__init__()
#         self.d_model = d_model
#         self.seq_len = seq_len

#         # CLS token (1, 1, 512)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
#         # Positional embeddings for CLS + 6 tokens → total 7
#         self.register_buffer("pos_embed", get_sinusoid_encoding(seq_len + 1, d_model))  # [1, 7, 512]

#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         self.dropout = dropout

#         # MLP classifier
#         self.mlp_head = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(d_model, num_classes)
#         )

    # def forward(self, x, attention_mask=None):  # x: (B, 6, 512)
    #     B = x.size(0)

    #     # Prepend CLS token
    #     cls_token = self.cls_token.expand(B, 1, self.d_model)  # (B, 1, 512)
    #     x = torch.cat([cls_token, x], dim=1)  # (B, 7, 512)

    #     # Add positional embeddings
    #     x = x + self.pos_embed[:, :x.size(1), :]

    #     # If attention_mask is provided, adjust it
    #     if attention_mask is not None:
    #         # attention_mask is visibility: shape (B, 6), dtype=bool, True=visible
    #         # CLS token is always visible
    #         cls_mask = torch.ones((B, 1), dtype=torch.bool, device=attention_mask.device)
    #         extended_mask = torch.cat([cls_mask, attention_mask], dim=1)  # (B, 7)
    #         # Transformer expects True for *masked* tokens
    #         key_padding_mask = ~extended_mask  # (B, 7)
    #     else:
    #         key_padding_mask = None

    #     # Transformer
    #     x = self.transformer(x, src_key_padding_mask = key_padding_mask)  # (B, 7, 512)
    #     cls_output = x[:, 0]     # Take CLS token output

    #     return self.mlp_head(cls_output)  # (B, num_classes), (B, 512)


