from typing import Callable

from ..structures.audiosignal import AudioSignal
from scipy.fftpack import fft as fft_
import numpy as np
import librosa


def __timeToSamples(audio: AudioSignal) -> Callable[[float], int]:
    return lambda v: int(audio.getSamplerate() * v)


def fft(audio: AudioSignal, nfft: int = None) -> np.ndarray:
    if not nfft:
        nfft = len(audio.signal)
    T = fft_(audio.signal, n=nfft)
    return T[:len(T) // 2]


def stft(audio: AudioSignal, winLen: float, hopLen: float) -> np.ndarray:
    ts = __timeToSamples(audio)
    return librosa.stft(
        audio.signal,
        n_fft=ts(winLen),
        win_length=ts(winLen),
        hop_length=ts(hopLen)
    )


def mfcc(audio: AudioSignal, winLen: float, hopLen: float) -> np.ndarray:
    ts = __timeToSamples(audio)
    return librosa.feature.mfcc(
        y=audio.signal,
        sr=audio.samplerate,
        n_fft=ts(winLen),
        win_length=ts(winLen),
        hop_length=ts(hopLen)
    )


def getAmplitude(T: np.ndarray) -> np.ndarray:
    return np.abs(T)


def amplitudeToDB(T: np.ndarray) -> np.ndarray:
    return librosa.amplitude_to_db(T)

