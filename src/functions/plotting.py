import numpy as np
import matplotlib.pyplot as plt

from ..structures.audiosignal import AudioSignal


def plotSignal(audio: AudioSignal) -> None:
    assert isinstance(audio, AudioSignal)
    plt.figure()
    x_axis = np.linspace(0, (1 / audio.samplerate) * len(audio.signal), len(audio.signal))
    plt.plot(x_axis, audio.signal)
    plt.grid()
    plt.title(audio.name)
    plt.show(block=False)


def plotSpectrum(fft_: np.ndarray, fs: float, title="") -> None:
    plt.figure()
    plt.plot(np.linspace(0, fs / 2, len(fft_)), fft_)
    plt.grid()
    plt.title(title)
    plt.show(block=False)


def plotSpectrogram():
    pass


def keepPlotsOpen():
    plt.show()
