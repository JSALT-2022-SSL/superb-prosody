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
USEYAAPT = False
# LJSpeech pitch reconstruction
class PitchDataset(Dataset):
    def __init__(self, mode, file_path, meta_data, upstream_rate=160):

        self.root = file_path
        self.meta_data = meta_data
        self.upstream_rate = upstream_rate
        self.fp = 1000 // (SAMPLE_RATE // self.upstream_rate)
        if DEBUG:
            print("Frame Period: ", self.fp)
        with open(self.meta_data, "r", encoding="utf-8") as f:
            self.usage_list = json.load(f)

        cache_path = os.path.join(CACHE_PATH, f'{mode}-{self.fp}.pkl')
        if os.path.isfile(cache_path):
            print(f'[Pitch Dataset] - Loading file paths from {cache_path}')
            with open(cache_path, 'rb') as cache:
                dataset = pickle.load(cache)
        else:
            dataset  = getattr(self, f"build_{mode}_dataset")()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as cache:
                pickle.dump(dataset, cache)
        print(f'[Pitch Dataset] - there are {len(dataset)} files found')

        self.dataset = dataset

        cache_path = os.path.join(CACHE_PATH, f'{mode}-labels-{self.fp}.pkl')
        if USEYAAPT:
            cache_path = os.path.join(CACHE_PATH, f'{mode}-yaapt-labels-{self.fp}.pkl')
        if os.path.isfile(cache_path):
            print(f'[Pitch Dataset] - Loading labels from {cache_path}')
            with open(cache_path, 'rb') as cache:
                all_labels = pickle.load(cache)
        else:
            all_labels = self.build_label(self.dataset)
            with open(cache_path, 'wb') as cache:
                pickle.dump(all_labels, cache)

        self.label = all_labels

        cache_path = os.path.join(CACHE_PATH, f'{mode}-stats-{self.fp}.pkl')
        if USEYAAPT:
            cache_path = os.path.join(CACHE_PATH, f'{mode}-yaapt-stats-{self.fp}.pkl')
        if os.path.isfile(cache_path) and 1 == 2:  # force not loading
            print(f'[Pitch Dataset] - Loading stats from {cache_path}')
            with open(cache_path, 'rb') as cache:
                all_stats = pickle.load(cache)
        else:
            all_stats = self.build_stats()
            with open(cache_path, 'wb') as cache:
                pickle.dump(all_stats, cache)

        self.norm_stat = all_stats
        if DEBUG:
            print("Stat: ", self.norm_stat)

    def name2path(self, name):
        return f"{self.root}/wavs/{name}"

    def build_label(self, path_list):
        y = {}
        for path in tqdm.tqdm(path_list, desc="Pitch extraction"):
            wav_path = self.name2path(path)

            if USEYAAPT:
                # pYAAPT
                signal = basic.SignalObj(wav_path)
                dio_len = int(signal.size * 1000 / SAMPLE_RATE / self.fp) + 1
                f0_pad = np.zeros(dio_len, dtype=np.float64)
                try:
                    pitch = pYAAPT.yaapt(signal, **{'f0_min' : 71.0, 'f0_max' : 800.0, 'frame_space' : self.fp})
                    f0 = pitch.samp_values.astype(np.float64)
                    f0_pad[:len(f0)] = f0
                    y[path] = f0_pad
                except:
                    print("Error detected: ", wav_path)
                    y[path] = f0_pad
            else:
                # pyWorld
                wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
                pitch, t = pw.dio(wav.astype(np.float64), SAMPLE_RATE, frame_period=self.fp)
                pitch = pw.stonemask(wav.astype(np.float64), pitch, t, SAMPLE_RATE)
                y[path] = pitch
        
        return y

    def build_stats(self):
        pitches = []
        # mx, mi = 2e9
        for path, pitch in tqdm.tqdm(self.label.items()):
            for p in pitch:
                if p == 0:
                    continue
                pitches.append(p)
        n = len(pitches)
        mean = sum(pitches) / n
        variance = sum([((x - mean) ** 2) for x in pitches]) / n
        std = variance ** 0.5
        return (mean, std)
    
    def build_train_dataset(self):
        dataset = []
        wav_list = glob.glob(f"{self.root}/wavs/*.wav")
        for wav_path in wav_list:
            wavname = os.path.basename(wav_path)
            if wavname.split("-")[0] in self.usage_list["train"]:
                dataset.append(wavname)
        if DEBUG:
            dataset = dataset[:100]
        print("finish searching training set wav")
        return dataset

    def build_dev_dataset(self):
        dataset = []
        wav_list = glob.glob(f"{self.root}/wavs/*.wav")
        for wav_path in wav_list:
            wavname = os.path.basename(wav_path)
            if wavname.split("-")[0] in self.usage_list["dev"]:
                dataset.append(wavname)
        if DEBUG:
            dataset = dataset[:100]
        print("finish searching dev set wav")
        return dataset

    def build_test_dataset(self):
        dataset = []
        wav_list = glob.glob(f"{self.root}/wavs/*.wav")
        for wav_path in wav_list:
            wavname = os.path.basename(wav_path)
            if wavname.split("-")[0] in self.usage_list["test"]:
                dataset.append(wavname)
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
