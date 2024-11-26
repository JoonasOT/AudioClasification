# Require Structures:
from typing import Final
from src.dependencies.dependencies import *

# Required functions:
from src.functions.transfroms import *
from src.functions.plotting import *
from src.functions.file_management import *
from src.functions.audio_manipulation import *

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

        mel = getMFCC(audio, N_MFCC, WIN_SIZE, HOP_SIZE, plot=True)
        print(audio.transform(AudioSignal.getName).unwrap(), "\n", mel)

    keepPlotsOpen()


if __name__ == '__main__':
    main()
