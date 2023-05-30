import torch


def padding_mask(padded_input, input_lengths=None, padding_token_id=None):
    if input_lengths is not None:
        mask = padded_input.new_ones(padded_input.size()[:-1])
        for i in range(padded_input.size(0)):
            mask[i, input_lengths[i]:] = 0
    if padding_token_id is not None:
        mask = padded_input.ne(padding_token_id).float()
    return mask.unsqueeze(-1)


def decoder_mask(x, padding_token_id=None):
    batch_size, seq_len = x.size()
    mask1 = torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.uint8), diagonal=1)
    mask1 = mask1.unsqueeze(0).expand(batch_size, -1, -1)
    if padding_token_id is not None:
        mask2 = x.eq(padding_token_id)
        mask2 = mask2.unsqueeze(1).expand(-1, seq_len, -1)
        mask1 = mask1 + mask2
    return mask1


def encoder_mask(padded_input, input_lengths, expand_length):
    non_pad_mask = padding_mask(padded_input, input_lengths=input_lengths)
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return mask
