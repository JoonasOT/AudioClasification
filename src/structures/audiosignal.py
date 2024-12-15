from __future__ import annotations

from scipy.io.wavfile import read as read_wav, write as write_wav
from numpy import ndarray, sqrt, mean, float64
import librosa
from sounddevice import play


class AudioSignal:
    def __init__(self, file: str):
        self.name = file
        content = read_wav(file)
        self.samplerate: int = content[0]
        self.signal: ndarray = float64(content[1])

    def rmse(self) -> float:
        return sqrt(mean(self.signal**2))

    def normalize(self) -> AudioSignal:
        self.signal = librosa.util.normalize(self.signal)
        return self

    def __str__(self) -> str:
        return f"AudioSignal[{self.name}, Signal: {self.signal}, Samplerate: {self.samplerate}]"

    def getName(self) -> str:
        return self.name

    def getSignal(self) -> ndarray:
        return self.signal

    def getSamplerate(self) -> int:
        return self.samplerate

    def write(self, file: str):
        return write_wav(file, self.samplerate, self.signal)

    def play(self, wait=True) -> None:
        play(self.signal, self.samplerate, blocking=wait)

    def getTime(self) -> float:
        return (1.0 / self.samplerate) * len(self.signal)