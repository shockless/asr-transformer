from math import ceil
import torch
from torch import nn

from modules.Transformer.layers import MHA, FeedForward, TrainablePositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self._attention = MHA(num_heads, emb_dim, dropout)
        self._norm1 = nn.LayerNorm(emb_dim)
        self._feedforward = FeedForward(emb_dim, ff_dim, dropout)
        self._norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        res_x = x
        x = self._norm1(x)
        x = self._attention(x)
        x += res_x

        res_x = x
        x = self._norm2(x)
        x = self._feedforward(x)
        x += res_x
        return x


class Encoder(nn.Module):
    def __init__(self, seq_len, emb_dim, input_dim, num_layers, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self._lin_in = nn.Linear(input_dim, emb_dim)
        self._norm_in = nn.LayerNorm(emb_dim)
        self._pe = TrainablePositionalEncoding(seq_len, emb_dim)

        self._layers = nn.ModuleList([EncoderLayer(emb_dim,
                                                   num_heads,
                                                   ff_dim,
                                                   dropout) for _ in range(num_layers)])

    def forward(self, x):
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2).contiguous()
        
        x = self._norm_in(self._lin_in(x)) + self._pe(x)

        for encoder_layer in self._layers:
            x = encoder_layer(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self._mask_attention = MHA(num_heads, emb_dim, dropout)
        self._norm1 = nn.LayerNorm(emb_dim)
        self._cross_attention = MHA(num_heads, emb_dim, dropout)
        self._norm2 = nn.LayerNorm(emb_dim)
        self._feedforward = FeedForward(emb_dim, ff_dim, dropout)
        self._norm3 = nn.LayerNorm(emb_dim)

    def forward(self, x, attention_mask, enc_x):
        res_x = x
        x = self._norm1(x)
        x = self._mask_attention(x, attention_mask=attention_mask)
        x += res_x

        res_x = x
        x = self._norm2(x)
        x = self._cross_attention(x, enc_x=enc_x)
        x += res_x

        res_x = x
        x = self._norm3(x)
        x = self._feedforward(x)
        x += res_x

        return x


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 seq_len,
                 emb_dim,
                 num_layers,
                 num_heads,
                 ff_dim,
                 eos_token_id,
                 dropout=0.1,
                 pad_token_id=0):
        super().__init__()

        self._seq_len = seq_len
        self._eos_token_id = eos_token_id

        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_token_id)
        self._pe = TrainablePositionalEncoding(seq_len, emb_dim)
        self._dropout = nn.Dropout(dropout)
        self._layers = nn.ModuleList([DecoderLayer(emb_dim,
                                                   num_heads,
                                                   ff_dim,
                                                   dropout) for _ in range(num_layers)])
        self._classifier = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, x, mask, enc_x):
        x = self._dropout(self._embedding(x) + self._pe(x))
        for decoder_layer in self._layers:
            x = decoder_layer(x, mask, enc_x)
        return self._classifier(x)

    def evaluate(self, x, enc_x):
        batch_size, seq_len = x.shape
        eoses = torch.full((batch_size,), self._seq_len - 1)
        for i in range(1, self._seq_len + 1):
            attention_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.uint8), diagonal=1)
            attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1, -1)
            
            prob = self._dropout(self._embedding(x) + self._pe(x))
            
            for decoder_layer in self._layers:
                prob = decoder_layer(prob, attention_mask, enc_x)

            prob = self._classifier(prob)
            next_word = prob[:, -1].argmax(dim=-1)

            for j in range(len(next_word)):
                if next_word[j] == self._eos_token_id:
                    eoses[j] = i

            next_word = next_word.unsqueeze(-1)
            x = torch.cat([x, next_word.to(x.device)], dim=1).to(enc_x.device)
        return x, prob, eoses


class Transformer(nn.Module):
    def __init__(self, vocab_size,
                 nfft,
                 embedding_dim,
                 decoder_seq_len,
                 encoder_seq_len,
                 encoder_num_layers,
                 decoder_num_layers,
                 num_heads,
                 ff_dim,
                 dropout=0.1,
                 pad_token_id=4,
                 eos_token_id=2,
                 vgg=True):
        super().__init__()

        if vgg:
            self.vgg = nn.Sequential(nn.Conv2d(1, 64, 3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2))
            # emb_dim = hidden_dim * 2 * (n_fft // 4)
            encoder_seq_len = encoder_seq_len // 4
            vgg_dim = (nfft // 2 + 1) // 4
        else:
            raise NotImplementedError()

        self.encoder = Encoder(seq_len=encoder_seq_len,
                               input_dim=vgg_dim*128,
                               emb_dim=embedding_dim,
                               num_layers=encoder_num_layers,
                               num_heads=num_heads,
                               ff_dim=ff_dim,
                               dropout=dropout)

        self.decoder = Decoder(vocab_size=vocab_size,
                               seq_len=decoder_seq_len,
                               emb_dim=embedding_dim,
                               num_layers=decoder_num_layers,
                               num_heads=num_heads,
                               ff_dim=ff_dim,
                               eos_token_id=eos_token_id,
                               dropout=dropout,
                               pad_token_id=pad_token_id)

    def forward(self, spectrum, text, mask):
        vgg_out = self.vgg(spectrum)
        enc_x = self.encoder(vgg_out)
    
        batch_size, seq_len = text.size()
        
        mask = mask.lt(1)
        mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
        
        attention_mask = torch.triu(torch.ones((seq_len, seq_len), device=mask.device, dtype=torch.uint8), diagonal=1)
        attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        mask = torch.logical_or(mask, mask.mT)
        mask = torch.logical_or(mask, attention_mask)
        
        logits = self.decoder(text, mask, enc_x)
        return logits

    def evaluate(self, spectrum, text):
        vgg_out = self.vgg(spectrum)
        enc_x = self.encoder(vgg_out)
        preds, logits, eoses = self.decoder.evaluate(text, enc_x)
        return preds, logits, eoses
