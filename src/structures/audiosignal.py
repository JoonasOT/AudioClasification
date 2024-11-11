from scipy.io.wavfile import read as read_wav, write as write_wav
from numpy import ndarray
from sounddevice import play as playSound, wait as waitForSound


class AudioSignal:
    def __init__(self, file: str):
        content = read_wav(file)
        self.samplerate: int = content[0]
        self.signal: ndarray = content[1]

    def __str__(self):
        return f"AudioSignal[Signal: {self.signal}, Samplerate: {self.samplerate}]"

    def write(self, file: str):
        return write_wav(file, self.samplerate, self.signal)

    def play(self, wait=True):
        playSound(self.signal, self.samplerate, blocking=wait)
