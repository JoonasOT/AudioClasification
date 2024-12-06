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
    model = NN.Model(SETTINGS, "./models/checkpoint.keras", False)
    model.importTrain(DATA_DIR + TRAIN_DIR)
    model.importValidation(DATA_DIR + TEST_DIR)

    for md in [model.trainData, model.validationData]:
        print(np.shape(md.data))
        print(md.labels)

    history = model.train(10).history
    print(history)


if __name__ == '__main__':
    main()
