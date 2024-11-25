# Require Structures:
from typing import Final
from src.dependencies.dependencies import *

# Required functions:
from src.functions.transfroms import *
from src.functions.plotting import *


FILES: Final[list[str]] = ["./data/audio1.wav", "./data/audio2.wav", "./data/x.wav"]


def main():
    for file in FILES:
        audio = Maybe(file).construct(AudioSignal, False)
        audio\
            .run(plotSignal)\
            .transform(fft)\
            .transform(getAmplitude)\
            .transform(amplitudeToDB)\
            .run(plotSpectrum, False,
                 audio.transform(AudioSignal.getSamplerate).orElse(44100),
                 audio.transform(AudioSignal.getName).orElse(None)
             )

    keepPlotsOpen()


if __name__ == '__main__':
    main()
