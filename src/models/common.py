from typing import Callable

import numpy as np

from src.functions.audio_manipulation import getNormalizedAudio, getMFCC
from src.functions.file_management import onlyWavFiles, getFilesInDir


def getMFCCs(path: str, modelNameGetter: Callable[[str], str], nMFCC: int, winLen: float, hopSize: float)\
        -> dict[str, list[np.ndarray]]:
    MFCCs: dict[str, list[np.ndarray]] = {}
    for file in onlyWavFiles(getFilesInDir(path)):
        print(file)
        # Create a normalized AudioSignal
        audio = getNormalizedAudio(file, plot=False)
        assert bool(audio), "File reading errornous!"
        model = modelNameGetter(file)
        mfcc = getMFCC(audio, nMFCC, winLen, hopSize, plot=False)
        if model in MFCCs:
            MFCCs[model].append(mfcc.unwrap())
        else:
            MFCCs[model] = [mfcc.unwrap()]
    return MFCCs