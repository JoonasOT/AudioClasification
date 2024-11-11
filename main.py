# Require Structures:
from src.structures.audiosignal import *
from src.structures.maybe import Maybe

# Required functions:
from src.functions.fft import fft, amplitudeToDB
from src.functions.plotting import plotSignal, plotTransfrom, keepPlotsOpen


def main():
    print(
        Maybe("./data/audio2.wav")
            .transform(getAudioSignal)
            .run(AudioSignal.play)
            .transform(fft)
            .run(plotSignal)
            .run(plotTransfrom)
            .transform(amplitudeToDB)
            .run(plotTransfrom)
    )
    keepPlotsOpen()


if __name__ == '__main__':
    main()
