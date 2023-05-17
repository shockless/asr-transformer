import os
import IPython


from typing import Dict, Tuple, List, Set
from collections import defaultdict

import torchaudio
import torch

from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, df, path_to_data, tokenizer,
                 n_fft=1024, n_mels=64, center=True, 
                 max_tokenized_length=100, max_audio_len=22,
                 sr=16000):
        super().__init__()

        self.texts = list(df.text)
        self.paths = list(df.path)
        #self.rates = list(df.rate)
        self.path_to_data = path_to_data
        self.tokenizer = tokenizer
        self.n_fft=n_fft
        self.sr = sr
        self.n_mels=n_mels
        self.center=center
        
        if not max_tokenized_length:
            self.max_tokenized_length = df.text.apply(lambda x: len(self.tokenizer.encode(x))).max()
        else:
            self.max_tokenized_length = max_tokenized_length
        if not max_audio_len:            
            self.max_audio_len = int(df.frames.max())
        else:
            self.max_audio_len = max_audio_len
        self.max_frames_len = self.max_audio_len * self.sr
        self.max_spectrogram_len = self.max_frames_len / self.n_fft * 2 + 1
        
        
    def __check_tokenizer_length(self, x):
        return 
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, ind):
        text = self.texts[ind]
        audio, sr = torchaudio.load(os.path.join(self.path_to_data, self.paths[ind]))
        
        if sr!=self.sr:
            audio = torchaudio.fuctional.resample(audio, sr, self.sr)
        
        audio = audio[:,:self.max_frames_len]
        audio = torch.cat([audio, torch.zeros((audio.shape[0], self.max_frames_len - audio.shape[1]))], dim=1)
        #print(spectrogram.shape[0]/sr)
        
        spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=self.sr,
                                                           n_fft=self.n_fft,
                                                           n_mels=self.n_mels,
                                                           center=self.center)
          
        #transform = torchaudio.transforms.AmplitudeToDB(stype="amplitude", top_db=80)
        #audio = transform(audio)
        
        return {'text': text, 
                'encoded_text': self.tokenizer.encode(text, padding='max_length', truncation=True, max_length=self.max_tokenized_length, return_tensors='pt'),
                'spectre': spectrogram(audio),
                'audio': audio,
                'sr': sr}
    
    def listen(self, ind):
        plt.figure(figsize=(20, 5))
        plt.imshow(self[ind]['spectre'].squeeze().log())
        plt.xlabel('Time', size=20)
        plt.ylabel('Mels', size=20)
        plt.show()
        print(ind)
        print(self.texts[ind])
        return IPython.display.Audio(os.path.join(self.path_to_data, self.paths[ind]))