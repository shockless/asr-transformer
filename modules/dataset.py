import os

import torchaudio
import torch

from math import ceil

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
        self.max_frames_len = max_frames_len
        
        self._transform = torchaudio.transforms.Spectrogram(n_fft=n_fft,
                                                            center=False)
    
    
    def __len__(self):
        return len(self._texts)

    
    def __getitem__(self, ind):
        audio, sr = torchaudio.load(self._paths[ind])
        
        if sr != self._sr:
            audio = torchaudio.fuctional.resample(audio, sr, self._sr)
            
        batch_size, frames = audio.shape
        audio = torch.cat([audio, torch.zeros((batch_size, self.max_frames_len - frames))], dim=1)

        spectrogram = self._transform(audio)
        
        return {'spectrum': spectrogram, 
                'text': self._texts[ind], 
                'mask': self._masks[ind]}
