from ..structures.audiosignal import AudioSignal
from scipy.fftpack import fft as fft_
import numpy as np


def fft(audio: AudioSignal, nfft: int = None) -> np.ndarray:
    if not nfft:
        nfft = len(audio.signal)
    T = fft_(audio.signal, n=nfft)
    return T[:len(T) // 2]


def getAmplitude(T: np.ndarray) -> np.ndarray:
    return np.abs(T)


def amplitudeToDB(T: np.ndarray) -> np.ndarray:
    return 10 * np.log10(T)
