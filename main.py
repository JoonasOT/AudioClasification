# Require Structures:
from typing import Final
from src.dependencies.dependencies import *

# Required functions:
from src.functions.transfroms import *
from src.functions.plotting import *
from src.functions.file_management import *


DIRECTORY: Final[str] = "./data"
WIN_SIZE: Final[float] = 0.032
HOP_SIZE: Final[float] = WIN_SIZE / 2


def main():
    for file in onlyWavFiles(getFilesInDir("./data")):
        # Create a normalized AudioSignal
        audio =\
            Maybe(file)\
            .construct(AudioSignal, False)\
            .transform(AudioSignal.normalize)

        # Form spectrums
        spectrum =\
            audio\
            .run(plotSignal)\
            .transform(fft)\
            .transform(getAmplitude)\
            .transform(amplitudeToDB)\
            .run(plotSpectrum, False,
                 audio.transform(AudioSignal.getSamplerate).orElse(44100),
                 "Spectrum -- " + audio.transform(AudioSignal.getName).orElse(None),
            )

        # Form spectrograms
        spectrogram =\
            audio\
            .transform(stft, False, WIN_SIZE, HOP_SIZE) \
            .transformers(getAmplitude, amplitudeToDB) \
            .run(
                plotSpectrogram,
                False,
                audio.transform(AudioSignal.getSamplerate).orElse(2.0) / 2,
                audio.transform(lambda as_: (1 / as_.getSamplerate()) * len(as_.getSignal())).orElse(1.0),
                "Spectrogram -- " + audio.transform(AudioSignal.getName).orElse(None)
            )
        mel =\
            audio\
            .transform(mfcc, False, WIN_SIZE, HOP_SIZE) \
            .run(
                plotSpectrogram,
                False,
                audio.transform(AudioSignal.getSamplerate).orElse(2.0) / 2,
                audio.transform(lambda as_: (1 / as_.getSamplerate()) * len(as_.getSignal())).orElse(1.0),
                "Mel -- " + audio.transform(AudioSignal.getName).orElse("")
            )
        print(audio.transform(AudioSignal.getName).unwrap(), "\n", mel)

    keepPlotsOpen()


if __name__ == '__main__':
    main()
