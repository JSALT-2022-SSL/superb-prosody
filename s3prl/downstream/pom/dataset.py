from pathlib import Path
import os
import pandas as pd

import random
import torch
import torchaudio

from torch.utils.data.dataset import Dataset
from itertools import accumulate

class POMDataset(Dataset):
    def __init__(self, split, dialogue_dir, id_dir, label_path, sample_rate):
        self.split = split  # train, val, or test
        self.dialogue_dir = dialogue_dir
        self.sample_rate = sample_rate

        with open(f'{id_dir}/{self.split}.txt', 'r') as f:
            self.ids = [idx.rstrip() for idx in f.readlines()]
        
        df = pd.read_csv(label_path, dtype=str)
        self.wav_files, self.labels = [], []
        for idx in self.ids:
            for wav_file in os.listdir(dialogue_dir):
                if f'{idx}-0.wav' in wav_file:
                    self.wav_files.append(wav_file)
                    self.labels.append(int(df[df['id'] == idx]['label']))
        
    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_file = self.wav_files[idx]
        label = self.labels[idx]
        
        # load waveform
        wav, sample_rate = torchaudio.load(f"{self.dialogue_dir}/{wav_file}")

        return wav, label

    def collate_fn(self, samples):
        wavs, labels = zip(*samples)
        
        wav_list = [wav for wav in wavs]
        label_list = [label for label in labels]
        
        return (
            torch.stack(wav_list).squeeze(1),
            label_list
        )