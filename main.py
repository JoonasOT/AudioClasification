# Require Structures:
from typing import Final

# Required functions:
from src.functions.file_management import *
from src.functions.audio_manipulation import FreqType, getNormalizedAudio, getSpectrum, getSpectrogram, getMFCC
from src.functions.plotting import keepPlotsOpen


DIRECTORY: Final[str] = "./data"
WIN_SIZE: Final[float] = 0.032
HOP_SIZE: Final[float] = WIN_SIZE / 2
N_MFCC: Final[int] = 40


def main():
    for file in onlyWavFiles(getFilesInDir("./data")):
        
        # Create a normalized AudioSignal
        audio = getNormalizedAudio(file, plot=False)

        # Form spectrums
        spectrum = getSpectrum(audio, FreqType.DECIBEL, plot=False)

        # Form spectrograms
        spectrogram = getSpectrogram(audio, FreqType.DECIBEL, WIN_SIZE, HOP_SIZE, plot=False)

        mfcc = getMFCC(audio, N_MFCC, WIN_SIZE, HOP_SIZE, plot=True)

    keepPlotsOpen()


if __name__ == '__main__':
    main()
