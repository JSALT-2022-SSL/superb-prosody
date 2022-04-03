import json
import pickle
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchaudio
from torchaudio import transforms

BASE_PATH = Path('./downstream/mustard/data')
DATA_PATH = BASE_PATH / 'sarcasm_data.json'
SPLIT_PATH = BASE_PATH / 'split.json'
AUDIO_PATH = BASE_PATH / 'audios' / 'utterances_final'
SAMPLE_RATE = 16000

class SarcasmDataset(Dataset):
    def __init__(self, mode):
        with DATA_PATH.open() as file:
            self.dataset_dict = json.load(file)

        with SPLIT_PATH.open() as file:
            self.split_dict = json.load(file)

        self.mode = mode

    def __getitem__(self, idx):
        file_id = self.split_dict[self.mode][idx]
        wav_path = AUDIO_PATH / f'{file_id}.wav'
        wav, sr = torchaudio.load(wav_path)
        
        transform = transforms.Resample(sr, SAMPLE_RATE)
        resampled_wav = transform(wav)

        wav = resampled_wav.squeeze(0)
        label = int(self.dataset_dict[file_id]['sarcasm'])

        return wav.numpy(), label, file_id

    def __len__(self):
        return len(self.split_dict[self.mode])

    def collate_fn(self, samples):
        return zip(*samples)