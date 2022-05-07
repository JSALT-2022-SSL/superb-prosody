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
from audiomentations import Compose, AddGaussianNoise, AddGaussianSNR, TimeStretch, Shift, PitchShift, Gain

BASE_PATH = Path('./downstream/mustard/data')
DATA_PATH = BASE_PATH / 'sarcasm_data.json'
SPLIT_PATH = BASE_PATH / 'split.json'
AUDIO_PATH = BASE_PATH / 'audios' / 'utterances_final'
SAMPLE_RATE = 16000

class SarcasmDataset(Dataset):
    def __init__(self, mode, aug_config=None):
        with DATA_PATH.open() as file:
            self.dataset_dict = json.load(file)

        with SPLIT_PATH.open() as file:
            self.split_dict = json.load(file)

        self.mode = mode
        self.aug_config = aug_config

    def __getitem__(self, idx):
        file_id = self.split_dict[self.mode][idx]
        wav_path = AUDIO_PATH / f'{file_id}.wav'
        wav, sr = torchaudio.load(wav_path)
        
        transform = transforms.Resample(sr, SAMPLE_RATE)
        resampled_wav = transform(wav)

        wav = resampled_wav.squeeze(0).numpy()
        label = int(self.dataset_dict[file_id]['sarcasm'])

        if self.aug_config:
            # Reference: https://github.com/iver56/audiomentations#waveform
            aug_list = []
            for aug, params in self.aug_config.items():
                aug_func = eval(aug)
                aug_list.append(aug_func(**params))

            augment = Compose(aug_list)
            wav = augment(samples=wav, sample_rate=SAMPLE_RATE)

        return wav, label, file_id

    def __len__(self):
        return len(self.split_dict[self.mode])

    def collate_fn(self, samples):
        return zip(*samples)