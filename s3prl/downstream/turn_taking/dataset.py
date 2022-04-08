from pathlib import Path
import os
import pandas as pd

import random
import torch
from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file
from itertools import accumulate

class MaptaskDataset(Dataset):
    # need to pass frame size
    def __init__(
            self, 
            targets_path, 
            dialogues_path, 
            id_list, 
            mode='train',
            predict_size=3,
            frame_size=50, 
            sample_rate=20000, 
            wav_size=60
        ):

        self.targets_path = Path(targets_path)
        self.dialogues_path = Path(dialogues_path)
        self.id_list = id_list

        self.wav_files = []
        for idx in self.id_list:
            for f in os.listdir(dialogues_path):
                if idx in f:
                    self.wav_files.append(f)

        self.frame_size = frame_size    # in ms
        self.sample_rate = sample_rate
        self.wav_size = wav_size    # in sec
        self.predict_size = predict_size
        # number of frames in a 60s wav
        self.n_wav_frame = int(self.wav_size * 1000 / self.frame_size)
        # number of frames in a prediction window
        self.n_predict_frame = int(self.predict_size * 1000 / self.frame_size)
        self.stride = int(sample_rate * (self.frame_size/1000))
        
        self.time_units_f = []
        self.time_units_g = []
        for idx in self.id_list:
            for f in os.listdir(targets_path):
                if idx in f:
                    # each window contains 60 frames
                    prediction_windows = self.sliding_window(
                        pd.read_csv(f"{targets_path}/{f}")['target'].tolist() 
                                    + [0] * self.n_predict_frame,
                        self.n_predict_frame
                    )
                    # split labels of whole wav into 60s (1200 frames) per window
                    splitted_prediction_windows = [
                        prediction_windows[i : min(i + self.n_wav_frame, len(prediction_windows))] 
                        for i in range(0, len(prediction_windows), self.n_wav_frame)
                    ]
                    if 'f' in f:
                        self.time_units_f += splitted_prediction_windows
                    else:
                        self.time_units_g += splitted_prediction_windows

    def __len__(self):
        return len(self.wav_files)
    
    def __getitem__(self, idx):
        """
        Split a 60s wav into 1200 frame (50ms each)
        Return: list of frames, list of labels
        """
        wav_file = self.wav_files[idx]
        # list of labels
        labels_f = self.time_units_f[idx]
        labels_g = self.time_units_g[idx]
        labels = labels_g + labels_f
        
        # load waveform
        wav, sample_rate = apply_effects_file(
            f"{self.dialogues_path}/{wav_file}",
            [
                ["rate", str(self.sample_rate)],
                ["norm"]
            ],
        )
        # pad if needed
        # if wav.shape[-1] % self.stride != 0:
        #     wav = self.padding(wav, self.stride)
        # shape: (num_channels * n_frame, frame_size)
        wav = wav.view(-1, self.stride)
        
        return wav, labels


    def collate_fn(self, samples):
        wavs, labels = zip(*samples)
        
        wav_list = [wav for wav in wavs]
        label_list = [label for label in labels]
        
        return (
            torch.stack(wav_list).view(-1, self.stride),
            label_list
        )

    def padding(self, wav, stride):
        pad_len = (wav.shape[-1]//stride + 1) * stride - wav.shape[-1]
        # shape of padding tensor
        zeros_tensor_shape = [shape for shape in wav.shape]
        zeros_tensor_shape[-1] = pad_len
        zeros_tensor_shape = tuple(zeros_tensor_shape)
        # pad to length divisible by stride
        padded_wav = torch.cat([wav, torch.zeros(zeros_tensor_shape)], dim=-1)

        return padded_wav

    def sliding_window(self, labels, stride):
        prediction_windows = []
        for i in range(0, len(labels) - stride, 1):
            next_n_frame_labels = labels[i + 1 : i + 1 + stride]
            prediction_windows.append(next_n_frame_labels)

        return prediction_windows
