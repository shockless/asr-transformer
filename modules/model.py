from math import ceil, floor
from tqdm.auto import tqdm
import torch
from torchmetrics.functional import word_error_rate
from torch import nn
from torch.autograd import Variable

import numpy as np

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
            attention_mask = attention_mask.unsqueeze(dim=2)
            temp *= attention_mask
            temp *= attention_mask.transpose(1, 2)

        temp = temp.bmm(v)

        return torch.softmax(temp, dim=-1), q


class MHA(nn.Module):
    def __init__(self, num_heads, emb_dim, r_dim):
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.r_dim = r_dim
        self.heads = nn.ModuleList([MHAHead(emb_dim, r_dim) for i in range(self.num_heads)])
        self.led = nn.Sequential(nn.Linear(self.emb_dim * self.num_heads, self.r_dim),
                                 nn.Linear(self.r_dim, self.emb_dim * self.num_heads))
        self.out = nn.Linear(self.emb_dim * self.num_heads, self.emb_dim)

    def forward(self, x, enc_x=None, attention_mask=None):
        heads = [self.heads[i](x, enc_x, attention_mask) for i in range(self.num_heads)]
        outs = []
        qs = []
        for i in range(self.num_heads):
            outs.append(heads[i][0])
            qs.append(heads[i][1])

        outs = torch.cat(outs, dim=-1)
        qs = torch.cat(qs, dim=-1)

        return self.out(self.led(outs) + qs)


class FeedForward(nn.Module):
    def __init__(self, emb_dim, ff_dim, r_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.ff_dim = ff_dim
        self.r_dim = r_dim
        self.squeeze = nn.Sequential(nn.Linear(self.emb_dim, self.r_dim),
                                     nn.Linear(self.r_dim, self.ff_dim))
        self.GELU = nn.GELU()

        self.unsqueeze = nn.Sequential(nn.Linear(self.ff_dim, self.r_dim),
                                       nn.Linear(self.r_dim, self.emb_dim))

    def forward(self, x):
        x = self.squeeze(x)
        x = self.GELU(x)
        x = self.unsqueeze(x)
        return x


class TrainablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, emb_dim, device):
        super().__init__()
        self.pe = Variable(torch.ones((seq_len, emb_dim)), requires_grad=True).to(device)

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

    def forward(self, x, attention_mask):
        temp = self.attention(x, attention_mask=attention_mask)
        temp += x
        x = self.norm1(temp)
        temp = self.ff(x)
        temp += x
        x = self.norm2(temp)
        return x


class Encoder(nn.Module):
    def __init__(self, seq_len, emb_dim, num_layers, num_heads, ff_dim, r_dim, device, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.r_dim = r_dim

        self.pe = TrainablePositionalEncoding(self.seq_len, self.emb_dim, device)

        self.layers = nn.ModuleList([EncoderLayer(self.emb_dim,
                                                  self.num_heads,
                                                  self.ff_dim,
                                                  self.r_dim,
                                                  self.dropout) for i in range(self.num_layers)])

    def forward(self, x, attention_mask):
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2).contiguous()
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
    def __init__(self, vocab_size, seq_len, emb_dim, num_layers, num_heads, ff_dim, r_dim, device, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.r_dim = r_dim
        self.vocab_size = vocab_size

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pe = TrainablePositionalEncoding(self.seq_len, self.emb_dim, device)

        self.layers = nn.ModuleList([DecoderLayer(self.emb_dim,
                                                  self.num_heads,
                                                  self.ff_dim,
                                                  self.r_dim,
                                                  self.dropout) for i in range(self.num_layers)])
        self.classifier = nn.Linear(self.emb_dim, self.vocab_size)

    def forward(self, x, attention_mask, enc_x):
        x = self.emb(x)
        x = self.pe(x)
        sizes = x.size()
        if len(sizes) > 3:
            x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        for i in range(self.num_layers):
            x = self.layers[i](x, attention_mask, enc_x)
        return self.classifier(x)


class Transformer(nn.Module):
    def __init__(self, vocab_size,
                 n_mels,
                 enc_seq_len, dec_seq_len,
                 hidden_dim,
                 enc_num_layers, dec_num_layers,
                 num_heads,
                 ff_dim, r_dim,
                 device,
                 dropout=0.1,
                 sr=16000,
                 n_fft=1024):
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
        self.emb_dim = hidden_dim * 2 * (self.n_mels // 4)
        self.vgg_seq_out = self.enc_seq_len // 4
        self.encoder = Encoder(seq_len=self.vgg_seq_out,
                               emb_dim=self.emb_dim,
                               num_layers=enc_num_layers,
                               num_heads=num_heads,
                               ff_dim=ff_dim,
                               r_dim=r_dim,
                               device=device,
                               dropout=dropout)
        self.decoder = Decoder(vocab_size=vocab_size,
                               seq_len=dec_seq_len,
                               emb_dim=self.emb_dim,
                               num_layers=dec_num_layers,
                               num_heads=num_heads,
                               ff_dim=ff_dim,
                               r_dim=r_dim,
                               device=device,
                               dropout=dropout)

    def forward(self, batch):
        enc_x = self.vgg(batch['spectre'])
        enc_x = self.encoder(enc_x, None)
        logits = self.decoder(batch['encoded_text'], None, enc_x)
        return logits

    def predict(self, batch, bos_token, eos_token, device):
        enc_x = self.vgg(batch['spectre'])
        enc_x = self.encoder(enc_x, None)
        dec_in = torch.zeros((batch['spectre'].shape[0], self.seq_len), dtype=torch.int32)
        dec_in = dec_in.to(device)
        mask = torch.zeros((batch['spectre'].shape[0], self.seq_len), dtype=torch.int32)
        mask = mask.to(device)
        dec_in[:, 0] = bos_token
        for i in range(self.seq_len - 1):
            mask[:, i] = 1
            logits = self.decoder(dec_in, mask, enc_x)
            dec_in[:, i + 1] = logits.argmax(dim=1)[:, i + 1]
        return dec_in, logits


def train_epoch(model, data_loader, tokenizer, loss_function, optimizer, scheduler, device):
    model.to(device)
    model.train()
    total_train_loss = 0

    dl_size = len(data_loader)

    preds = []
    targets = []

    for batch in tqdm(data_loader):
        batch['encoded_text'] = batch['encoded_text'].to(device)
        batch['spectre'] = batch['spectre'].to(device)
        batch['ohe_text'] = batch['ohe_text'].to(device)

        optimizer.zero_grad()
        logits = model(batch)

        pred = logits.argmax(dim=-1).to('cpu')
        pred = [tokenizer.decode(i) for i in pred]
        preds.append(pred)
        targets.append(batch['text'])
        loss = loss_function(logits.transpose(1, 2), batch['encoded_text'].squeeze())
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    acc_t = 0
    wer = 0
    for batch in range(dl_size):
        acc = 0
        for sample in range(len(preds[batch])):
            acc += int(preds[batch][sample] == targets[batch][sample])

        acc /= sample + 1
        wer += word_error_rate(preds[batch], targets[batch])
        acc_t += acc
    acc_t = acc_t / dl_size
    wer = wer / dl_size
    metrics = {
        "Train Loss": total_train_loss / dl_size,
        "Train WAcc": 1 - wer.item(),
        "Train Accuracy": acc,
    }

    return metrics


def eval_epoch(model, data_loader, tokenizer, loss_function, device):
    model.to(device)
    model.eval()
    total_train_loss = 0

    preds = []
    targets = []

    dl_size = len(data_loader)

    for batch in tqdm(data_loader):
        batch['encoded_text'] = batch['encoded_text'].to(device)
        batch['spectre'] = batch['spectre'].to(device)
        batch['ohe_text'] = batch['ohe_text'].to(device)

        with torch.no_grad():
            pred, logits = model.predict(batch, tokenizer.bos_token_id, tokenizer.eos_token_id, device)
            pred = [tokenizer.decode(i) for i in pred]
            preds.append(pred)
            targets.append(batch['text'])

        loss = loss_function(logits.transpose(1, 2), batch['encoded_text'].squeeze())
        total_train_loss += loss.item()

    acc_t = 0
    wer = 0
    for batch in range(dl_size):
        acc = 0
        for sample in range(len(preds[batch])):
            acc += int(preds[batch][sample] == targets[batch][sample])

        acc /= sample + 1
        wer += word_error_rate(preds[batch], targets[batch])
        acc_t += acc
    acc_t = acc_t / dl_size
    wer = wer / (dl_size)
    metrics = {
        "Val Loss": total_train_loss / dl_size,
        "Val WAcc": 1 - wer.item(),
        "Val Accuracy": acc,
    }

    return metrics, preds[-1][-1]