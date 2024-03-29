import os

import torchaudio
import torch

from math import ceil, floor

from torch.utils.data import Dataset



# import IPython
#
# from typing import Dict, Tuple, List, Set
# from collections import defaultdict


class AudioDataset(Dataset):
    def __init__(self, 
                 paths,
                 texts,
                 masks,
                 n_fft=1024,
                 max_frames_len=2048,
                 sr=16000):
        super().__init__()
        
        self._paths = paths
        self._texts = texts
        self._masks = masks
        self._sr = sr
        self._max_frames_len = max_frames_len
        
        self._transform = torchaudio.transforms.Spectrogram(n_fft=n_fft,
                                                            center=False)
    
    
    def __len__(self):
        return len(self._texts)
    
    def _calculate_spectrum_len(self, x):
        return floor((x - self._transform.win_length) / self._transform.hop_length) + 1
    
    def __getitem__(self, ind):
        audio, sr = torchaudio.load(self._paths[ind])
        
        if sr != self._sr:
            audio = torchaudio.fuctional.resample(audio, sr, self._sr)
        #audio = torch.cat([audio, torch.zeros((batch_size, self.max_frames_len - frames))], dim=1)

        spectrogram = self._transform(audio)
        channels, freq, frames = spectrogram.shape
        padding_length = self._calculate_spectrum_len(self._max_frames_len) - frames
        spectrum_mask = torch.cat([torch.ones((frames,)), torch.zeros((padding_length,))])
        spectrogram = torch.cat([spectrogram, torch.zeros((channels, freq, padding_length))], dim=-1)
        return {'spectrum': spectrogram, 
                'spectrum_mask': spectrum_mask,
                'text': self._texts[ind], 
                'mask': self._masks[ind]}
