import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np 
import librosa
from librosa.util import find_files
from torchaudio import load
from torch import nn
import os 
import re
import random
import pickle
import torchaudio
import sys
import time
import glob
import tqdm
import json
import pyworld as pw
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic


CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')
SAMPLE_RATE = 16000

DEBUG = False
# LibriTTS energy reconstruction
class EnergyDataset(Dataset):
    def __init__(self, mode, file_path, meta_data, upstream_rate=160):
        
        self.mode = mode
        self.root = file_path
        self.meta_data = meta_data
        self.upstream_rate = upstream_rate
        self.fp = 1000 // (SAMPLE_RATE // self.upstream_rate)
        if DEBUG:
            print("Frame Period: ", self.fp)
        self.usage_list = []
        with open(f"{self.meta_data}/{mode}-filtered.txt", "r", encoding="utf-8") as f:
            for line in f:
                if line == "\n":
                    continue
                self.usage_list.append(line.strip())

        cache_path = os.path.join(CACHE_PATH, f'{mode}-{self.fp}.pkl')
        if os.path.isfile(cache_path):
            print(f'[Energy Dataset] - Loading file paths from {cache_path}')
            with open(cache_path, 'rb') as cache:
                dataset = pickle.load(cache)
        else:
            dataset  = getattr(self, f"build_{mode}_dataset")()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as cache:
                pickle.dump(dataset, cache)
        print(f'[Energy Dataset] - there are {len(dataset)} files found')

        self.dataset = dataset

        cache_path = os.path.join(CACHE_PATH, f'{mode}-labels-{self.fp}.pkl')
        if os.path.isfile(cache_path):
            print(f'[Energy Dataset] - Loading labels from {cache_path}')
            with open(cache_path, 'rb') as cache:
                all_labels = pickle.load(cache)
        else:
            all_labels = self.build_label(self.dataset)
            with open(cache_path, 'wb') as cache:
                pickle.dump(all_labels, cache)

        self.label = all_labels

        cache_path = os.path.join(CACHE_PATH, f'{mode}-stats-{self.fp}.pkl')
        if os.path.isfile(cache_path):
            print(f'[Energy Dataset] - Loading stats from {cache_path}')
            with open(cache_path, 'rb') as cache:
                all_stats = pickle.load(cache)
        else:
            all_stats = self.build_stats()
            with open(cache_path, 'wb') as cache:
                pickle.dump(all_stats, cache)

        self.norm_stat = all_stats

    def name2path(self, name):
        prefix = name.split('_')
        if self.mode == "train":
            return f"{self.root}/train-clean-100/{prefix[0]}/{prefix[1]}/{name}"
        elif self.mode == "dev":
            return f"{self.root}/dev-clean/{prefix[0]}/{prefix[1]}/{name}"
        elif self.mode == "test":
            return f"{self.root}/test-clean/{prefix[0]}/{prefix[1]}/{name}"
        else:
            raise NotImplementedError

    def build_label(self, path_list):
        y = {}
        for path in tqdm.tqdm(path_list, desc="Pitch extraction"):
            wav_path = self.name2path(path)
            wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
            # dio_len = int(wav.size * 1000 / SAMPLE_RATE / self.fp) + 1
            mag, phase = librosa.magphase(librosa.stft(wav, n_fft=1024, hop_length=self.upstream_rate))
            energy = np.linalg.norm(mag, axis=0)
            y[path] = energy

        return y
    
    def build_stats(self):
        energies = []
        for path, energy in tqdm.tqdm(self.label.items()):
            for p in energy:
                if p == 0:
                    continue
                energies.append(p)
        n = len(energies)
        mean = sum(energies) / n
        variance = sum([((x - mean) ** 2) for x in energies]) / n
        std = variance ** 0.5
        return (mean, std)
    
    def build_train_dataset(self):
        dataset = self.usage_list
        if DEBUG:
            dataset = dataset[:100]
        print("finish searching training set wav")
        return dataset

    def build_dev_dataset(self):
        dataset = self.usage_list
        if DEBUG:
            dataset = dataset[:100]
        print("finish searching dev set wav")
        return dataset

    def build_test_dataset(self):
        dataset = self.usage_list
        if DEBUG:
            dataset = dataset[:100]
        print("finish searching test set wav")
        return dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        path = self.dataset[idx]
        wav, _ = librosa.load(self.name2path(path), sr=SAMPLE_RATE)
        
        return np.float32(wav), torch.FloatTensor(self.label[path]).view(-1, 1)
        
    def collate_fn(self, samples):
        return zip(*samples)
