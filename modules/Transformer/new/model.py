from math import ceil
import torch
from torch import nn

from modules.Transformer.layers import MHA, FeedForward, TrainablePositionalEncoding
from modules.Transformer.masking import padding_mask, encoder_mask, decoder_mask


class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.attention = MHA(self.num_heads, self.emb_dim, self.dropout)
        self.norm1 = nn.LayerNorm(self.emb_dim)
        self.ff = FeedForward(self.emb_dim, self.ff_dim, self.dropout)
        self.norm2 = nn.LayerNorm(self.emb_dim)

    def forward(self, x, attention_mask, non_pad_mask):
        x = self.attention(x, attention_mask=attention_mask)
        x = self.norm1(x)
        x *= non_pad_mask
        x = self.ff(x)
        x = self.norm2(x)
        x *= non_pad_mask
        return x


class Encoder(nn.Module):
    def __init__(self, seq_len, emb_dim, num_layers, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.lin_in = nn.Linear(emb_dim, emb_dim)
        self.norm_in = nn.LayerNorm(emb_dim)
        self.pe = TrainablePositionalEncoding(self.seq_len, self.emb_dim)

        self.layers = nn.ModuleList([EncoderLayer(self.emb_dim,
                                                  self.num_heads,
                                                  self.ff_dim,
                                                  self.dropout) for _ in range(self.num_layers)])

    def forward(self, x, lens):
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])

        x = x.transpose(1, 2).contiguous()

        non_pad_mask = padding_mask(x, input_lengths=lens)
        self_attn_mask = encoder_mask(x, lens, self.seq_len)

        x = self.norm_in(self.lin_in(x)) + self.pe(x)

        for i in range(self.num_layers):
            x = self.layers[i](x, self_attn_mask, non_pad_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.mask_attention = MHA(self.num_heads, self.emb_dim, dropout)
        self.norm1 = nn.LayerNorm(self.emb_dim)
        self.attention = MHA(self.num_heads, self.emb_dim, dropout)
        self.norm2 = nn.LayerNorm(self.emb_dim)
        self.ff = FeedForward(self.emb_dim, self.ff_dim, dropout)
        self.norm3 = nn.LayerNorm(self.emb_dim)

    def forward(self, x, attention_mask, enc_x, enc_mask, non_pad_mask):
        x = self.mask_attention(x, attention_mask=attention_mask)
        x = self.norm1(x)
        x *= non_pad_mask
        x = self.attention(x, enc_x=enc_x, attention_mask=enc_mask)
        x = self.norm2(x)
        x *= non_pad_mask
        x = self.ff(x)
        x = self.norm3(x)
        x *= non_pad_mask

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_dim, num_layers, num_heads, ff_dim, eos_token, bos_token, dropout=0.1,
                 padding_idx=0, ):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.pe = TrainablePositionalEncoding(self.seq_len, self.emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(self.emb_dim,
                                                  self.num_heads,
                                                  self.ff_dim,
                                                  dropout) for i in range(self.num_layers)])
        self.classifier = nn.Linear(self.emb_dim, self.vocab_size, bias=False)

    def forward(self, x, enc_x, enc_lens):
        non_pad_mask = padding_mask(x, padding_token_id=self.eos_token)
        attention_mask = decoder_mask(x, padding_token_id=self.eos_token).gt(0)
        dec_enc_attn_mask = encoder_mask(enc_x, enc_lens, self.seq_len)
        x = self.dropout(self.emb(x) + self.pe(x))
        for i in range(self.num_layers):
            x = self.layers[i](x, attention_mask, enc_x, dec_enc_attn_mask, non_pad_mask)
        return self.classifier(x)

    def evaluate(self, enc_x, device):
        dec_in = torch.full((enc_x.shape[0], 1), self.bos_token, dtype=torch.int32).to(device)
        eoses = torch.full((enc_x.shape[0],), self.seq_len - 1)
        for i in range(self.seq_len):
            non_pad_mask = torch.ones_like(dec_in).float().unsqueeze(-1)
            attention_mask = decoder_mask(dec_in)
            prob = self.dropout(self.emb(dec_in) + self.pe(dec_in))

            for j in range(self.num_layers):
                prob = self.layers[j](prob, attention_mask, enc_x, None, non_pad_mask)
            prob = self.classifier(prob)
            next_word = prob[:, -1].argmax(dim=-1)
            for j in range(len(next_word)):
                if next_word[j] == self.eos_token:
                    eoses[j] = i
            next_word = next_word.unsqueeze(-1)
            dec_in = torch.cat([dec_in, next_word.to(device)], dim=1).to(device)
        return dec_in, prob, eoses


class Transformer(nn.Module):
    def __init__(self, vocab_size,
                 n_mels,
                 enc_seq_len, dec_seq_len,
                 hidden_dim,
                 enc_num_layers, dec_num_layers,
                 num_heads,
                 ff_dim,
                 device,
                 dropout=0.1,
                 sr=16000,
                 n_fft=1024,
                 padding_idx=4,
                 eos_token=2,
                 bos_token=1):
        super().__init__()
        self.n_mels = n_mels
        self.enc_seq_len = ceil(enc_seq_len * sr / n_fft * 2)
        self.vgg = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.seq_len = dec_seq_len
        self.vocab_size = vocab_size
        self.emb_dim = self.n_mels  # hidden_dim * 2 * (self.n_mels // 4)
        self.vgg_seq_out = self.enc_seq_len
        self.eos_token = eos_token
        self.bos_token = bos_token
        print(self.vgg_seq_out, self.emb_dim)
        self.encoder = Encoder(seq_len=self.vgg_seq_out,
                               emb_dim=self.emb_dim,
                               num_layers=enc_num_layers,
                               num_heads=num_heads,
                               ff_dim=ff_dim,
                               dropout=dropout)
        self.decoder = Decoder(vocab_size=vocab_size,
                               seq_len=self.seq_len,
                               emb_dim=self.emb_dim,
                               num_layers=dec_num_layers,
                               num_heads=num_heads,
                               ff_dim=ff_dim,
                               eos_token=self.eos_token,
                               bos_token=self.bos_token,
                               dropout=dropout,
                               padding_idx=padding_idx)
        self.device = device

    def forward(self, batch):
        # enc_x = self.vgg(batch['spectre'])
        enc_x = self.encoder(batch['spectre'], batch['spectrogram_len'])
        logits = self.decoder(batch['encoded_text'], enc_x, batch['spectrogram_len'])
        return logits

    def evaluate(self, batch):
        enc_x = self.encoder(batch['spectre'], batch['spectrogram_len'])
        preds, logits, eoses = self.decoder.evaluate(enc_x, self.device)
        return preds, logits, eoses
