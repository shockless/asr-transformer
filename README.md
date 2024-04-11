# ASR-Transformer
My implementation of Speech-Transformer model using PyTorch

## Technologies used
- Torch 2.0
- Torchaudio
- Spectrogramms
- Transformer

## Citations

```

@INPROCEEDINGS{8462506,
  author={Dong, Linhao and Xu, Shuang and Xu, Bo},
  booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition}, 
  year={2018},
  volume={},
  number={},
  pages={5884-5888},
  keywords={Hidden Markov models;Encoding;Training;Decoding;Speech recognition;Time-frequency analysis;Spectrogram;Speech Recognition;Sequence-to-Sequence;Attention;Transformer},
  doi={10.1109/ICASSP.2018.8462506}}
```


### TODO:
- Train on big corpus and evaluate metrics
- Implement 2d-attention proposed in the paper (it has not been implemented due to minor changes in metrics in the evaluation results in the paper)
- Make beamsearch instead of greedy-search
