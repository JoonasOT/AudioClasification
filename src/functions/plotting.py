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


def plotSpectrogram(spectrogram: np.ndarray, maxF: float, maxT: float, title: str = ""):
    plt.figure()
    plt.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        cmap="inferno",
        interpolation='none',
        extent=(0.0, maxT, 0.0, maxF)
    )
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")


def keepPlotsOpen():
    plt.show()
