import torch
from torchmetrics.functional import word_error_rate
from tqdm.auto import tqdm


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
        batch['spectrogram_len'] = batch['spectrogram_len'].to(device)
        batch['text_len'] = batch['text_len'].to(device)
        batch['true_text'] = batch['true_text'].to(device)

        optimizer.zero_grad()
        logits = model(batch)
        pred = logits.argmax(dim=-1).to('cpu')
        pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
        preds.append(pred)
        targets.append(batch['text'])
        loss = loss_function(logits.transpose(1, 2), batch['true_text'])
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
        "Train Word Accuracy": 1 - wer.item(),
        "Train Accuracy": acc_t,
    }

    return metrics, preds[-1][-1]


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
        batch['spectrogram_len'] = batch['spectrogram_len'].to(device)
        batch['text_len'] = batch['text_len'].to(device)
        batch['true_text'] = batch['true_text'].to(device)

        with torch.no_grad():
            pred, logits, eoses = model.evaluate(batch)
            pred, logits = remove_after_eos(pred, logits, eoses, tokenizer.eos_token_id)
            pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
            preds.append(pred)
            targets.append(batch['text'])

        loss = loss_function(logits.transpose(1, 2), batch['true_text'])
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
    wer = wer / dl_size
    metrics = {
        "Val Loss": total_train_loss / dl_size,
        "Val Word Accuracy": 1 - wer.item(),
        "Val Accuracy": acc_t,
    }

    return metrics, preds[-1][-1]


def remove_after_eos(pred, logits, eoses, eos_token):
    for i in range(eoses.shape[0]):
        pred[i, eoses[i]:] = eos_token
        eos = torch.zeros((logits.shape[2]))
        eos[eoses[i]] = 1
        logits[i, eoses[i]:] = eos
    return pred, logits
