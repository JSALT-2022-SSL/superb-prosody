import json
import pickle
import os
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchaudio
from torchaudio import transforms
from audiomentations import Compose, AddGaussianNoise, TimeStretch, Shift, Gain

BASE_PATH = Path('./downstream/mustard/data')
DATA_PATH = BASE_PATH / 'sarcasm_data.json'
SPLIT_PATH = BASE_PATH / 'split_indices.p'
AUDIO_PATH = BASE_PATH / 'audios' / 'utterances_final'
SAMPLE_RATE = 16000

class SarcasmDataset(Dataset):
    def __init__(self, mode, speaker_dependent, split_no=None, aug_config=None):
        with DATA_PATH.open() as file:
            self.dataset_dict = json.load(file)

        with SPLIT_PATH.open(mode='rb') as file:
            split = pickle.load(file, encoding="latin1")

        if speaker_dependent:
            split_idx = split[split_no][mode != 'train']
            file_ids = list(self.dataset_dict.keys())
            self.file_ids = [file_ids[_id] for _id in split_idx]
        else:
            self.file_ids = [
                k for k, v in self.dataset_dict.items()
                if (mode == 'train' and v['show'] != 'FRIENDS') or
                   (mode == 'dev' and v['show'] == 'FRIENDS')
            ]

        self.aug_config = aug_config

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
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
        return len(self.file_ids)

    def collate_fn(self, samples):
        return zip(*samples)


# training_data = SarcasmDataset('train', 0)
# dev_data = SarcasmDataset('dev', 0)
# print(len(training_data))
# print(len(dev_data))

# train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
# for i, data in enumerate(train_dataloader):
#     if i > 10:
#         break
#     print(data[2])