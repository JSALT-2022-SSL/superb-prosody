import os
import glob
import numpy as np
import librosa
import pyworld as pw
from tqdm import tqdm
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic


# path = '/mnt/d/Data/LibriTTS/dev-clean/84/121123/84_121123_000007_000001.wav'
SAMPLE_RATE = 24000


def matching(path):
    signal = basic.SignalObj(path)
    pitch = pYAAPT.yaapt(signal, **{'f0_min' : 71.0, 'f0_max' : 800.0, 'frame_space' : 10.0})
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
    for path in tqdm(glob.glob("/mnt/d/Data/LibriTTS/dev-clean/84/121123/*.wav")):
        matching(path)
