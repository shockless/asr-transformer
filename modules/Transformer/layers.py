import torch
from torch import nn
from math import log


class MHAHead(nn.Module):
    def __init__(self, emb_dim, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.v = nn.Linear(self.emb_dim, self.emb_dim)
        self.q = nn.Linear(self.emb_dim, self.emb_dim)
        self.k = nn.Linear(self.emb_dim, self.emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_x=None, attention_mask=None):
        if enc_x is None:
            v = self.v(x)
            q = self.q(x)
            k = self.k(x)
        else:
            v = self.v(enc_x)
            q = self.q(x)
            k = self.k(enc_x)
        temp = q.bmm(k.transpose(1, 2)) * (self.emb_dim ** (-0.5))  # B, seq_len, seq_len

        if attention_mask is not None:
            temp = temp.masked_fill(attention_mask.gt(0), float('-inf'))

        temp = torch.softmax(temp, dim=-1)
        temp = self.dropout(temp)
        temp = temp.bmm(v)
        return temp


class MHA(nn.Module):
    def __init__(self, num_heads, emb_dim, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([MHAHead(emb_dim, dropout) for i in range(self.num_heads)])
        self.out = nn.Linear(self.emb_dim * self.num_heads, self.emb_dim)

    def forward(self, x, enc_x=None, attention_mask=None):
        heads = torch.cat([self.heads[i](x, enc_x, attention_mask) for i in range(self.num_heads)], dim=-1)
        return self.dropout(self.out(heads)) + x


class FeedForward(nn.Module):
    def __init__(self, emb_dim, ff_dim, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.ff_dim = ff_dim
        self.squeeze = nn.Linear(self.emb_dim, self.ff_dim)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.unsqueeze = nn.Linear(self.ff_dim, self.emb_dim)

    def forward(self, x):
        x1 = self.squeeze(x)
        x1 = self.ReLU(x1)
        x1 = self.dropout(x1)
        x1 = self.unsqueeze(x1)
        return x + x1


class TrainablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, emb_dim):
        super().__init__()

        pe = torch.zeros(seq_len, emb_dim, requires_grad=False)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        exp_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * exp_term)
        pe[:, 1::2] = torch.cos(position * exp_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
