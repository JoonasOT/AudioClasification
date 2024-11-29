from main import SAMPLERATE, N_SAMPLES, WIN_SIZE, HOP_SIZE
from src.functions.audio_manipulation import getNormalizedAudio, sampleTo, getSpectrum, getSpectrogram, FreqType, \
    limitSamplesTo
from src.functions.file_management import onlyWavFiles, getFilesInDir
from src.functions.plotting import keepPlotsOpen


def viz():
    for file in onlyWavFiles(getFilesInDir("./data")):
        # Create a normalized AudioSignal
        audio = getNormalizedAudio(file, plot=True).transformers(sampleTo(SAMPLERATE), limitSamplesTo(N_SAMPLES))

        # Form spectrums
        spectrum = getSpectrum(audio, FreqType.DECIBEL, plot=True)

        # Form spectrograms
        spectrogram = getSpectrogram(audio, FreqType.DECIBEL, WIN_SIZE, HOP_SIZE, plot=True)
        break

    keepPlotsOpen()


if __name__ == "__main__":
    viz()