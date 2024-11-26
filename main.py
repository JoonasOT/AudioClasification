# Require Structures:
from typing import Final
from src.dependencies.dependencies import *

# Required functions:
from src.functions.transfroms import *
from src.functions.plotting import *
from src.functions.file_management import *


DIRECTORY: Final[str] = "./data"


def main():
    for file in onlyWavFiles(getFilesInDir("./data")):
        # Create a normalized AudioSignal
        audio =\
            Maybe(file)\
            .construct(AudioSignal, False)\
            .transform(AudioSignal.normalize)
        
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
