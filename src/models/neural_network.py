import numpy as np

from src.functions.audio_manipulation import getMFCC, getNormalizedAudio
from src.functions.file_management import onlyWavFiles, getFilesInDir
from src.models.common import Settings


def normalize__(arr: np.ndarray) -> np.ndarray:
    max_ = np.max([np.abs(np.min(arr)), np.max(arr)])
    arr /= 2 * max_
    return arr + 0.5


class Model:
    def __init__(self, settings: Settings):
        self.settings: Settings = settings
        self.trainLabels: list[int] = []
        self.labels: dict[str, int] = {}

    def train(self, dir_: str):
        __labelMax = 0
        mfccs: list[np.ndarray] = []
        spectralBands: list[np.ndarray] = []
        spectralCentroids: list[np.ndarray] = []

        for file in onlyWavFiles(getFilesInDir(dir_)):
            label = file.split("/")[-2]
            audio = getNormalizedAudio(file)
            mfcc = audio.transform(
                getMFCC, False,
                self.settings.binCount,
                self.settings.winLen,
                self.settings.hopSize
            ).unwrap().unwrap()

            mfcc = normalize__(mfcc)

            if label not in self.labels:
                self.labels[label] = __labelMax
                __labelMax += 1

            self.trainLabels.append(self.labels[label])
            mfccs.append(mfcc)

            # spectralBands = ...
            # spectralCentroids = ...


