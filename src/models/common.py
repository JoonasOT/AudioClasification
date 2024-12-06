from typing import Callable, NamedTuple
import numpy as np


from src.functions.audio_manipulation import getNormalizedAudio, getMFCC, conditionalRunner, sampleTo, limitSamplesTo, \
    getSpectralCentroid
from src.functions.file_management import onlyWavFiles, getFilesInDir


class Settings(NamedTuple):
    samplerate: int
    samples: int
    winLen: float
    hopSize: float
    binCount: int


"""
Extracts MFCCs (Mel-Frequency Cepstral Coefficients) from audio files in a given directory.
Args:
    path (str): The directory path containing audio files.
    modelNameGetter (Callable[[str], str]): A function that takes a file path and returns the label name of the file.
    nMFCC (int): The number of MFCCs to extract.
    winLen (float): The window length for MFCC extraction.
    hopSize (float): The hop size for MFCC extraction.
    samplerate (int, optional): The target sample rate for audio files. Defaults to None.
    samples (int, optional): The maximum number of samples to consider from each audio file. Defaults to None.

Returns:
    dict[str, list[np.ndarray]]: A dictionary where keys are model names and values are lists of MFCC arrays.
"""
def getMFCCs(path: str, modelNameGetter: Callable[[str], str], nMFCC: int, winLen: float, hopSize: float,
             samplerate: int = None, samples: int = None) -> dict[str, list[np.ndarray]]:
    MFCCs: dict[str, list[np.ndarray]] = {}
    for file in onlyWavFiles(getFilesInDir(path)):
        # Create a normalized AudioSignal
        audio = getNormalizedAudio(file, plot=False).transformers(
            conditionalRunner(samplerate is not None, sampleTo(samplerate), lambda v: v),
            conditionalRunner(samples is not None, limitSamplesTo(samples), lambda v: v),
        )

        assert bool(audio), "File reading errornous!"
        model = modelNameGetter(file)
        mfcc = getMFCC(audio, nMFCC, winLen, hopSize, plot=False)
        if model in MFCCs:
            MFCCs[model].append(mfcc.unwrap())
        else:
            MFCCs[model] = [mfcc.unwrap()]
    return MFCCs


def getSpectralCentroids(path: str, modelNameGetter: Callable[[str], str], nFFT: int, winLen: float, 
                        hopSize: float, samplerate: int = None, samples: int = None):
    for file in onlyWavFiles(getFilesInDir(path)):
        audio = getNormalizedAudio(file, plot=False).transformers(
            conditionalRunner(samplerate is not None, sampleTo(samplerate), lambda v: v),
            conditionalRunner(samples is not None, limitSamplesTo(samples), lambda v: v),
        )
        assert bool(audio), "File reading errornous!"
        label = modelNameGetter(file)
        spectralCentroids = getSpectralCentroid(audio, nFFT, winLen, hopSize, plot=False)
        if label in spectralCentroids:
            spectralCentroids[label].append(spectralCentroids.unwrap())
        else:
            spectralCentroids[label] = [spectralCentroids.unwrap()]
        