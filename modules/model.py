import torch
from torch import nn
from torchaudio import transforms
from torch.autograd import Variable


class MHAHead(nn.Module):
    def __init__(self, emb_dim, r_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.r_dim = r_dim

        self.v = nn.Sequential(nn.Linear(self.emb_dim, self.r_dim),
                               nn.Linear(self.r_dim, self.emb_dim))
        self.q = nn.Sequential(nn.Linear(self.emb_dim, self.r_dim),
                               nn.Linear(self.r_dim, self.emb_dim))
        self.k = nn.Sequential(nn.Linear(self.emb_dim, self.r_dim),
                               nn.Linear(self.r_dim, self.emb_dim))

    def forward(self, x, enc_x=None, attention_mask=None):
        if not enc_x:
            v = self.v(x)
            q = self.q(x)
            k = self.k(x)
        else:
            v = self.v(enc_x)
            q = self.q(x)
            k = self.k(enc_x)
        temp = q.bmm(k.transpose(1, 2)) * (self.emb_dim ** (-0.5))
        temp = temp.bmm(v)
        if not isinstance(attention_mask, type(None)):
            temp *= attention_mask
            temp *= attention_mask.T
        return torch.softmax(temp, dim=-1), q


class MHA(nn.Module):
    def __init__(self, num_heads, emb_dim, r_dim):
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.r_dim = r_dim
        self.heads = nn.ModuleList([MHAHead(emb_dim, r_dim) for i in range(self.num_heads)])
        self.led = nn.Sequential(nn.Linear(self.emb_dim * self.num_heads, self.r_dim),
                                 nn.ReLU()
                                 nn.Linear(self.r_dim, self.emb_dim))
        self.out = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, x, enc_x=None, attention_mask=None):
        temp = torch.cat([self.heads[i](x, attention_mask) for i in range(self.num_heads)], dim=-1)
        return self.out(self.led(temp[0]) + temp[1])
        '''FIX THIS'''


class FeedForward(nn.Module):
    def __init__(self, emb_dim, ff_dim, r_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.ff_dim = ff_dim
        self.r_dim = r_dim
        self.squeeze = nn.Sequential(nn.Linear(self.emb_dim, self.r_dim),
                                     nn.Linear(self.r_dim, self.ff_dim))

        self.unsqueeze = nn.Sequential(nn.Linear(self.ff_dim, self.r_dim),
                                       nn.Linear(self.r_dim, self.emb_dim))

    def forward(self, x):
        x = self.squeeze(x)
        x = nn.GELU(x)
        x = self.unsqueeze(x)
        return x


class TrainablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, emb_dim):
        super().__init__()
        self.pe = Variable(torch.ones((seq_len, emb_dim)), requires_grad=True)

    def forward(self, x):
        return x + self.pe


class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, r_dim, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.r_dim = r_dim

        self.attention = MHA(self.num_heads, self.emb_dim, self.r_dim)
        self.norm1 = nn.LayerNorm(self.emb_dim)
        self.ff = FeedForward(self.emb_dim, self.ff_dim, self.r_dim)
        self.norm2 = nn.LayerNorm(self.emb_dim)

    def forward(self, x):
        temp = self.attention(x)
        temp += x
        x = self.norm1(temp)
        temp = self.ff(x)
        temp += x
        x = self.norm2(temp)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_dim, num_layers, num_heads, ff_dim, r_dim, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.r_dim = r_dim
        self.emb = nn.Embedding(vocab_size, self.emb_dim)  # FIX

        self.pe = TrainablePositionalEncoding(self.seq_len, self.emb_dim)

        self.layers = nn.ModuleList([EncoderLayer(self.emb_dim,
                                                  self.num_heads,
                                                  self.ff_dim,
                                                  self.r_dim,
                                                  self.dropout) for i in range(self.num_layers)])

    def forward(self, x, attention_mask):
        x = self.emb(x)
        x = self.pe(x)
        for i in range(self.num_layers):
            x = self.layers[i](x, attention_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, r_dim, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.r_dim = r_dim
        self.mask_attention = MHA(self.num_heads, self.emb_dim, self.r_dim)
        self.norm1 = nn.LayerNorm(self.emb_dim)
        self.attention = MHA(self.num_heads, self.emb_dim, self.r_dim)
        self.norm2 = nn.LayerNorm(self.emb_dim)
        self.ff = FeedForward(self.emb_dim, self.ff_dim, self.r_dim)
        self.norm3 = nn.LayerNorm(self.emb_dim)

    def forward(self, x, attention_mask, enc_x):
        temp = self.mask_attention(x, attention_mask=attention_mask)
        temp += x
        x = self.norm1(temp)
        temp = self.attention(x, enc_x=enc_x)
        temp += x
        x = self.norm2(temp)
        temp = self.ff(x)
        temp += x
        x = self.norm3(temp)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_dim, num_layers, num_heads, ff_dim, r_dim, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.r_dim = r_dim
        self.vocab_size = vocab_size
        # self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pe = TrainablePositionalEncoding(self.seq_len, self.emb_dim)

        self.layers = nn.ModuleList([DecoderLayer(self.emb_dim,
                                                  self.num_heads,
                                                  self.ff_dim,
                                                  self.r_dim,
                                                  self.dropout) for i in range(self.num_layers)])
        self.classifier = nn.Linear(self.emb_dim, self.vocab_size)
    def forward(self, x, attention_mask, enc_x):
        # x = self.emb(x)
        x = self.pe(x)
        for i in range(self.num_layers):
            x = self.layers[i](x, attention_mask, enc_x)
        return self.classifier(x)
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, enc_seq_len, dec_seq_len, 
                 emb_dim, 
                 enc_num_layers, dec_num_layers, 
                 num_heads, 
                 ff_dim, r_dim, 
                 dropout=0.1, 
                 sr=16000, 
                 n_fft=1024):
        super().__init__()
        self.vgg = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        enc_seq_len = int(enc_seq_len*sr/n_fft*2+1)
        self.encoder = Encoder(vocab_size=vocab_size, seq_len=enc_seq_len, emb_dim=emb_dim, num_layers=enc_num_layers, num_heads=num_heads, ff_dim=ff_dim, r_dim=r_dim, dropout=dropout)
        self.decoder = Decoder(vocab_size=vocab_size, seq_len=dec_seq_len, emb_dim=emb_dim, num_layers=dec_num_layers, num_heads=num_heads, ff_dim=ff_dim, r_dim=r_dim, dropout=dropout)
    def forward(self, batch):
        enc_x = self.encoder(batch['spectre'])
        logits = self.decoder(batch['encoded_text'], None, enc_x)
        return logits
        
