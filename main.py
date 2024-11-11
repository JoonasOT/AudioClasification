# Require Structures:
from src.structures.audiosignal import *
from src.structures.maybe import Maybe

# Required functions:
from src.functions.transfroms import fft, amplitudeToDB
from src.functions.plotting import plotSignal, plotTransfrom, keepPlotsOpen


def main():
    [
        Maybe(file)
        .transform(getAudioSignal)  # Get the actual audio
        # .run(AudioSignal.play)    # Play the recording
        .transform(fft)             # Get the DFT of the signal
        .run(plotSignal)            # Plot the signal in the time domain
        # .run(plotTransfrom)       # Plot the signal in the frequency domain
        .transform(amplitudeToDB)   # Transfrom the DFT values to DB scale
        .run(plotTransfrom)         # Plot the signal in the frequency domain
        for file in ["./data/audio1.wav", "./data/audio2.wav", "./data/x.wav"]
    ]
    keepPlotsOpen()


if __name__ == '__main__':
    main()
