import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask.gt(0), -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output


class MHA(nn.Module):
    def __init__(self, num_heads, dim_model, dropout=0.1):
        super(MHA, self).__init__()

        self.num_heads = num_heads

        self.dim_model = dim_model
        self.dim_key = dim_model
        self.dim_value = dim_model

        self.query_linear = nn.Linear(dim_model, num_heads * self.dim_key)
        self.key_linear = nn.Linear(dim_model, num_heads * self.dim_key)
        self.value_linear = nn.Linear(dim_model, num_heads * self.dim_value)

        nn.init.normal_(self.query_linear.weight, mean=0, std=np.sqrt(2.0 / (self.dim_model + self.dim_key)))
        nn.init.normal_(self.key_linear.weight, mean=0, std=np.sqrt(2.0 / (self.dim_model + self.dim_key)))
        nn.init.normal_(self.value_linear.weight, mean=0, std=np.sqrt(2.0 / (self.dim_model + self.dim_value)))

        self.attention = ScaledDotProductAttention(temperature=np.power(self.dim_key, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

        self.output_linear = nn.Linear(num_heads * self.dim_value, dim_model)
        nn.init.xavier_normal_(self.output_linear.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, len_query, _ = query.size()
        batch_size, len_key, _ = key.size()
        batch_size, len_value, _ = value.size()

        residual = query

        query = self.query_linear(query).view(batch_size, len_query, self.num_heads,
                                              self.dim_key)  # B x T_Q x num_heads x H_K
        key = self.key_linear(key).view(batch_size, len_key, self.num_heads, self.dim_key)
        value = self.value_linear(value).view(batch_size, len_value, self.num_heads,
                                              self.dim_value)

        query = query.permute(2, 0, 1, 3).contiguous().view(-1, len_query, self.dim_key)
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, len_key, self.dim_key)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, len_value,
                                                            self.dim_value)

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)

        output = self.attention(query, key, value, mask=mask)

        output = output.view(self.num_heads, batch_size, len_query, self.dim_value)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_query, -1)

        output = self.dropout(self.output_linear(output))
        output = self.layer_norm(output + residual)

        return output


class FeedForward(nn.Module):
    def __init__(self, emb_dim, ff_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.ff_dim = ff_dim
        self.squeeze = nn.Linear(self.emb_dim, self.ff_dim)
        self.ReLU = nn.ReLU()
        self.unsqueeze = nn.Linear(self.ff_dim, self.emb_dim)

    def forward(self, x):
        x1 = self.squeeze(x)
        x1 = self.ReLU(x1)
        x1 = self.unsqueeze(x1)
        return x + x1


class TrainablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, emb_dim):
        super().__init__()

        pe = torch.zeros(seq_len, emb_dim, requires_grad=False)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        exp_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(np.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * exp_term)
        pe[:, 1::2] = torch.cos(position * exp_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        return self.pe[:, :input.size(1)]
