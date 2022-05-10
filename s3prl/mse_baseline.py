import os
import glob
import math
import numpy as np
import librosa
import pyworld as pw
from tqdm import tqdm
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
import pickle


PATH = '/mnt/d/Data/LibriTTS/train-clean-100/78/368/78_368_000017_000005.wav'
SAMPLE_RATE = 24000


def matching(path):
    signal = basic.SignalObj(path)
    pitch = pYAAPT.yaapt(signal, **{'f0_min' : 71.0, 'f0_max' : 800.0, 'frame_space' : 20.0})
    f0 = pitch.samp_values.astype(np.float64)
    dio_len = int(signal.size * 1000 / SAMPLE_RATE / 10) + 1
    f0_pad = np.zeros(dio_len, dtype=np.float64)
    f0_pad[:len(f0)] = f0
    # print(f0)

    wav, _ = librosa.load(path, sr=SAMPLE_RATE)
    pitch, t = pw.dio(wav.astype(np.float64), SAMPLE_RATE, frame_period=10)
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, SAMPLE_RATE)
    # print(pitch)
    print(len(f0), len(f0_pad), len(pitch))
    assert dio_len == len(pitch) == len(f0_pad)


def main():
    pass


if __name__ == "__main__":
    with open("./downstream/energy/.cache/test-labels-10.pkl", 'rb') as f:
        label = pickle.load(f)
    with open("./downstream/energy/.cache/test-stats-10.pkl", 'rb') as f:
        (mean, std) = pickle.load(f)
    pitches = {}
    for path, pitch in tqdm(label.items()):
        # spk = "all"
        spk = path.split('_')[0]
        # spk = path
        if spk not in pitches:
            pitches[spk] = []
        
        for p in pitch:
            if p == 0:
                continue
            pitches[spk].append(p)
            # pitches[spk].append((p - mean) / std)
            # pitches[spk].append(math.log(p))
    total = 0
    variance = 0.0
    for spk in pitches:
        n = len(pitches[spk])
        if n == 0:
            continue
        total += n
        mean = sum(pitches[spk]) / n
        variance += sum([((x - mean) ** 2) for x in pitches[spk]])
    std =  variance / total
    print(std)
