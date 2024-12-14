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
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show(block=False)


def plotSpectrum(fft_: np.ndarray, fs: float, title="", isDB=True) -> None:
    plt.figure()
    plt.plot(np.linspace(0, fs / 2, len(fft_)), fft_)
    plt.grid()
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(f"Amplitude{' (dB)' if isDB else ''}")
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
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar()
    plt.title(title)


def keepPlotsOpen():
    plt.show()


def save(where: str):
    plt.savefig("./img/" + where)
