from typing import Final

import src.models.neural_network as NN
from src.models.common import Settings

DATA_DIR: Final[str] = "./data"
TEST_DIR: Final[str] = "/test"
TRAIN_DIR: Final[str] = "/train"

SAMPLERATE: Final[int] = 22100
DURATION: Final[float] = 4.0
N_SAMPLES: Final[int] = int(DURATION * SAMPLERATE)

WIN_SIZE: Final[float] = 0.032
HOP_SIZE: Final[float] = WIN_SIZE / 2
N_MFCC: Final[int] = 40


SETTINGS: Final[Settings] = Settings(SAMPLERATE, N_SAMPLES, WIN_SIZE, HOP_SIZE, N_MFCC)


def main():
    USE_SAVE = False
    model = NN.Model(SETTINGS, "./models/checkpoint.keras", USE_SAVE)

    if not USE_SAVE:
        model.importTrain(DATA_DIR + TRAIN_DIR)
        model.importValidation(DATA_DIR + TEST_DIR)
        history = model.train(20, 400)
    else:
        model.importLabelsFrom(DATA_DIR + TEST_DIR)

    for prediction in model.predictionsFor(DATA_DIR + TEST_DIR):
        print(prediction)


if __name__ == '__main__':
    main()
