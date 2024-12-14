import sys

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
    # Load model from memory or train a new one?
    USE_SAVE = False
    # Compute MFCCs or try to first get them from cache?
    COMPUTE_MFCCS_EVERYTIME = True

    HIGHLIGHT_INCORRECT = True

    model = NN.Model(
        SETTINGS,
        "./models/checkpoint.keras",
        useCachedValues=not COMPUTE_MFCCS_EVERYTIME,
        useSave=USE_SAVE
    )

    # Logic for using the model saved in Memory or training one from scratch
    if not USE_SAVE:
        model.importTrain(DATA_DIR + TRAIN_DIR)
        model.importValidation(DATA_DIR + TEST_DIR)
        model.train(13, 100, saveHistory="./models/history")
    else:
        # If we imported from memory we still have to initialize the labels
        # to get predictions etc
        model.importLabelsFrom(DATA_DIR + TEST_DIR)

    # Predictions:
    for prediction in model.predictionsFor(DATA_DIR + TEST_DIR):
        correct = prediction.gotLabel == prediction.correctLabel if HIGHLIGHT_INCORRECT else True
        print(prediction, file=sys.stdout if correct else sys.stderr)


if __name__ == '__main__':
    main()