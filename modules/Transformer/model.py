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

    def forward(self, x, mask):
        res_x = x
        x = self._attention(x, attention_mask=mask)
        x = self._norm1(x + res_x)
        
        res_x = x
        x = self._feedforward(x)
        x = self._norm2(x + res_x)
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

    def forward(self, x, mask):
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        
        x = x.transpose(1, 2).contiguous()
        
        batch_size, seq_len, input_dim = x.shape
        
        encoder_mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
        encoder_mask = encoder_mask.lt(1)
        
        encoder_mask = torch.logical_or(encoder_mask, encoder_mask.mT)
        
        x = self._norm_in(self._lin_in(x)) + self._pe(x)

        for encoder_layer in self._layers:
            x = encoder_layer(x, encoder_mask)
            
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

    def forward(self, x, mask, enc_x, enc_mask):
        # print(x.shape)
        res_x = x
        x = self._mask_attention(x, attention_mask=mask)
        x = self._norm1(x + res_x)
                
        res_x = x
        x = self._cross_attention(x, enc_x=enc_x, attention_mask=enc_mask)
        x = self._norm2(x + res_x)

        res_x = x
        x = self._feedforward(x)
        x = self._norm3(x + res_x)
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

    def forward(self, x, mask, enc_x, enc_mask):
        batch_size, seq_len = x.size()
        _, enc_len, emb_dim = enc_x.size()
        
        mask_neg = mask.lt(1)
        enc_mask = enc_mask.lt(1)
        
        dec_mask = mask_neg.unsqueeze(1).expand(-1, enc_len, -1)
        enc_mask = enc_mask.unsqueeze(1).expand(-1, seq_len, -1)
        
        enc_mask = torch.logical_or(enc_mask, dec_mask.mT)
        
        mask_neg = mask_neg.unsqueeze(1).expand(-1, seq_len, -1)
        
        attention_mask = torch.triu(torch.ones((seq_len, seq_len), device=mask_neg.device, dtype=torch.uint8), diagonal=1)
        attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1, -1)
        mask_neg = torch.logical_or(mask_neg, mask_neg.mT)
        mask_neg = torch.logical_or(mask_neg, attention_mask)
        
        x = self._dropout(self._embedding(x) + self._pe(x))
        
        for decoder_layer in self._layers:
            x = decoder_layer(x, mask_neg, enc_x, enc_mask)
            
        return self._classifier(x)

    def evaluate(self, x, enc_x, enc_mask):
        eoses = torch.full((x.shape[0],), self._seq_len - 1)
        for i in range(1, self._seq_len + 1):
            batch_size, seq_len = x.shape
            encoder_mask = enc_mask.unsqueeze(1).expand(-1, seq_len, -1)
            attention_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.uint8), diagonal=1)
            attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1, -1)
            
            prob = self._dropout(self._embedding(x) + self._pe(x))
            
            for decoder_layer in self._layers:
                prob = decoder_layer(prob, attention_mask, enc_x, encoder_mask)

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
                 input_dim,
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
                 vgg=False):
        super().__init__()
        self._vgg = None
        if vgg:
            raise NotImplementedError()
            self._vgg = nn.Sequential(nn.Conv2d(1, 64, 3, stride=1, padding=1),
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
            input_dim = input_dim // 4 * 128            

        self.encoder = Encoder(seq_len=encoder_seq_len,
                               input_dim=input_dim,
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

    def forward(self, spectrum, spectrum_mask, text, mask):
        enc_x = self.encoder(spectrum, spectrum_mask)
        logits = self.decoder(text, mask, enc_x, spectrum_mask)
        return logits

    def evaluate(self, spectrum, spectrum_mask, text):
        # vgg_out = self.vgg(spectrum)
        enc_x = self.encoder(spectrum, spectrum_mask)
        preds, logits, eoses = self.decoder.evaluate(text, enc_x, spectrum_mask)
        return preds, logits, eoses
