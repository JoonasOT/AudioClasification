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


def main():
    for file in onlyWavFiles(getFilesInDir("./data")):
        # Create a normalized AudioSignal
        audio = getNormalizedAudio(file, plot=False)

        # Form spectrums
        spectrum = getSpectrum(audio, FreqType.DECIBEL, plot=True)

        # Form spectrograms
        spectrogram = getSpectrogram(audio, FreqType.DECIBEL, WIN_SIZE, HOP_SIZE, plot=True)

        continue
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
