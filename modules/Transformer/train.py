import torch
from torchmetrics.functional import word_error_rate
from tqdm.auto import tqdm


def train_epoch(model, data_loader, loss_function, optimizer, device):
    model.to(device)
    model.train()
    total_train_loss = 0

    dl_size = len(data_loader)

    preds = []
    targets = []

    for batch in tqdm(data_loader):
        spectrum = batch['spectrum'].to(device)
        text = batch['text'].to(device)
        mask = batch['mask'].to(device)
        spectrum_mask = batch['spectrum_mask'].to(device)
        
        input_text = text.detach().clone()
        input_text[:, mask.sum(dim=-1)-1] = input_text[:, -1]
        mask[:, mask.sum(dim=-1)-1] = mask[:, -1]
        input_text, mask = input_text[:, :-1], mask[:, :-1]
        optimizer.zero_grad()
        

        logits = model(spectrum, spectrum_mask, input_text, mask)
        pred = logits.argmax(dim=-1)
        preds.append(pred.to('cpu'))
        targets.append(text[:, :-1].to('cpu'))
        loss = loss_function(logits.transpose(1, 2), text[:, 1:])
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()

    acc_t = 0
    wer = 0
#     for batch in range(dl_size):
#         acc = 0
#         for sample in range(len(preds[batch])):
#             acc += int(preds[batch][sample] == targets[batch][sample])

#         acc /= sample + 1
#         # wer += word_error_rate(preds[batch], targets[batch])
#         acc_t += acc
#     acc_t = acc_t / dl_size
#     # wer = wer / dl_size
    metrics = {
        "Train Loss": total_train_loss / dl_size,
        # "Train Word Accuracy": 1 - wer.item(),
        # "Train Accuracy": acc_t,
    }

    return metrics, preds, targets


def eval_epoch(model, data_loader, eos_token_id, bos_token_id, loss_function, device):
    model.to(device)
    model.eval()
    total_train_loss = 0

    preds = []
    targets = []
    dl_size = len(data_loader)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            spectrum = batch['spectrum'].to(device)
            text = batch['text'].to(device)
            spectrum_mask = batch['spectrum_mask'].to(device)

            text_in = torch.full((text.shape[0], 1), bos_token_id, dtype=torch.int32).to(device)
            pred, logits, eoses = model.evaluate(spectrum, spectrum_mask, text_in)
            pred, logits = remove_after_eos(pred, logits, eoses, eos_token_id)
            # pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
            preds.append(pred.to('cpu'))
            targets.append(text.to('cpu'))

            loss = loss_function(logits.transpose(1, 2), text[:, 1:])
            total_train_loss += loss.item()

#     acc_t = 0
#     wer = 0
#     for batch in range(dl_size):
#         acc = 0
#         for sample in range(len(preds[batch])):
#             acc += int(preds[batch][sample] == targets[batch][sample])

#         acc /= sample + 1
#         wer += word_error_rate(preds[batch], targets[batch])
#         acc_t += acc
#     acc_t = acc_t / dl_size
#     wer = wer / dl_size
    metrics = {
        "Val Loss": total_train_loss / dl_size,
        # "Val Word Accuracy": 1 - wer.item(),
        # "Val Accuracy": acc_t,
    }

    return metrics, preds[-1][-1]


def remove_after_eos(pred, logits, eoses, pad_token_id):
    for i in range(eoses.shape[0]):
        pred[i, eoses[i]:] = pad_token_id
        eos = torch.zeros((logits.shape[2]))
        eos[eoses[i]] = 1
        logits[i, eoses[i]:] = eos
    return pred, logits
