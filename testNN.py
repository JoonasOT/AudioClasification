from typing import Final

import numpy as np

import src.models.neural_network as NN
from src.models.common import Settings

DATA_DIR: Final[str] = "./data"
TEST_DIR: Final[str] = "/test"
TRAIN_DIR: Final[str] = "/train"

SAMPLERATE: Final[int] = 22100
N_SAMPLES: Final[int] = int(2.0 * SAMPLERATE)

WIN_SIZE: Final[float] = 0.032
HOP_SIZE: Final[float] = WIN_SIZE / 2
N_MFCC: Final[int] = 40


SETTINGS: Final[Settings] = Settings(SAMPLERATE, N_SAMPLES, WIN_SIZE, HOP_SIZE, N_MFCC)


def main():
    USE_SAVE = True
    model = NN.Model(SETTINGS, "./models/checkpoint.keras", USE_SAVE)
    model.importTrain(DATA_DIR + TRAIN_DIR)
    model.importValidation(DATA_DIR + TEST_DIR)

    for md in [model.trainData, model.validationData]:
        print(np.shape(md.data))
        print(md.labels)

    if not USE_SAVE:
        history = model.train(10, 100)

    preds = model.predictionsFor(DATA_DIR + TEST_DIR)
    print([f"{preds[0][i]} got {model.preditionToLabel(preds[1][i])}" for i in range(len(preds[0]))])


if __name__ == '__main__':
    main()
