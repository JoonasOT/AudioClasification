from typing import Callable, NamedTuple
import numpy as np


from src.functions.audio_manipulation import getNormalizedAudio, getMFCC, conditionalRunner, sampleTo, limitSamplesTo
from src.functions.file_management import onlyWavFiles, getFilesInDir


class Settings(NamedTuple):
    samplerate: int
    samples: int
    winLen: float
    hopSize: float
    binCount: int


def getMFCCs(path: str, modelNameGetter: Callable[[str], str], nMFCC: int, winLen: float, hopSize: float,
             samplerate: int = None, samples: int = None) -> dict[str, list[np.ndarray]]:
    MFCCs: dict[str, list[np.ndarray]] = {}
    for file in onlyWavFiles(getFilesInDir(path)):
        print(file)
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