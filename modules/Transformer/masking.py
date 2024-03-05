import torch


def padding_mask(padded_input, input_lengths=None, padding_token_id=None):
    if input_lengths is not None:
        
        if len(padded_input.size()) == 2:
            mask = padded_input.new_ones(padded_input.size())
        elif len(padded_input.size()) == 3:
            mask = padded_input.new_ones(padded_input.size()[:-1])
            
        for i in range(padded_input.size(0)):
            mask[i, input_lengths[i]:] = 0
            
    if padding_token_id is not None:
        mask = padded_input.ne(padding_token_id).float()
    return mask


def decoder_mask(x, pad_mask):
    batch_size, seq_len = x.size()
    mask = torch.triu(torch.ones((batch_size, seq_len, seq_len), device=x.device, dtype=torch.uint8), diagonal=1)
    return mask


def encoder_mask(padded_input, input_lengths, expand_length):
    non_pad_mask = padding_mask(padded_input, input_lengths=input_lengths)
    pad_mask = non_pad_mask.lt(1)
    mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return mask
