from .dependencies import *


class AudioSignal:
    def __init__(self, file):
        self.sampleRate, self.signal = read_wav(file)

    def __str__(self):
        return f"AudioSignal[{self.signal}, {self.sampleRate}]"
