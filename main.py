# Require Structures:
from src.structures.audiosignal import *
from src.structures.maybe import Maybe

# Required functions:
from src.functions.fft import fft, amplitudeToDB
from src.functions.plotting import plotSignal, plotTransfrom, keepPlotsOpen


def main():
    [
        Maybe(file)
        .transform(getAudioSignal)
        .run(AudioSignal.play)
        .transform(fft)
        .run(plotSignal)
        .run(plotTransfrom)
        .transform(amplitudeToDB)
        .run(plotTransfrom)
        for file in ["./data/audio1.wav", "./data/audio2.wav", "./data/x.wav"]
    ]
    keepPlotsOpen()


if __name__ == '__main__':
    main()
