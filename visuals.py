import matplotlib.pyplot as plt
from random import shuffle

# Get settings
from main import SAMPLERATE, N_SAMPLES, WIN_SIZE, HOP_SIZE, N_MFCC
# Get file management functions and plotting convenience functions
from src.functions.file_management import onlyWavFiles, getFilesInDir
from src.functions.plotting import keepPlotsOpen, save

# Functions for getting and transforming audio
from src.functions.audio_manipulation import getNormalizedAudio, limitSamplesTo, sampleTo
# Functions for getting fft, and stft of an audio
from src.functions.audio_manipulation import FreqType, getSpectrum, getSpectrogram
# Function for getting mfcc
from src.functions.audio_manipulation import getMFCC, getSpectralCentroid


def getImages(dir_: str) -> dict[str, list[str]]:
    tmp = {file: file.split("/")[-2] for file in onlyWavFiles(getFilesInDir(dir_))}
    images: dict[str, list[str]] = {label: [] for _, label in tmp.items()}
    for file, label in tmp.items():
        images[label].append(file)
    return images


SAVE_DIR = "./img/"
TAKE = 1


def viz():
    for label, files in getImages("./data/").items():
        print(f"Label: {label}")
        shuffle(files)
        for index, file in enumerate(files[:TAKE]):
            # Create a normalized AudioSignal
            plot = False
            audio = getNormalizedAudio(file, plot=plot).transformers(sampleTo(SAMPLERATE), limitSamplesTo(N_SAMPLES))
            if plot:
                save(f"audio_{label}_{index}")

            # Form spectrums
            plot = False
            spectrum = getSpectrum(audio, FreqType.DECIBEL, plot=plot)
            if plot:
                save(f"spectrum_{label}_{index}")

            # Form spectrograms
            plot = False
            spectrogram = getSpectrogram(audio, FreqType.DECIBEL, WIN_SIZE, HOP_SIZE, plot=plot)
            if plot:
                save(f"spectrogram_{label}_{index}")

            # Form MFCCs
            plot = False
            mfcc = getMFCC(audio, N_MFCC, WIN_SIZE, HOP_SIZE, plot=plot)
            if plot:
                save(f"mfcc_{label}_{index}")

            # Form spectral centroid
            plot = True
            sc = getSpectralCentroid(audio, int(WIN_SIZE * SAMPLERATE), WIN_SIZE, HOP_SIZE, plot=plot)
            if plot:
                pass
                # save(f"spectralCentroid_{label}_{index}")


    keepPlotsOpen()


if __name__ == "__main__":
    viz()
