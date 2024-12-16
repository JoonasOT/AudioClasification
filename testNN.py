import sys

from typing import Final

from src.functions.file_management import WORKING_DIR, getFullPath
import src.models.neural_network as NN
from src.models.common import Settings

DATA_DIR: Final[str] = "data"
TEST_DIR: Final[str] = "test"
OWN_DIR: Final[str] = "own"
TRAIN_DIR: Final[str] = "train"

MODEL_DIR: Final[str] = "models"
MODEL_NAME: Final[str] = "model.keras"
HISTORY_NAME: Final[str] = "history.json"

FINAL_DIR: Final[str] = "final"


SAMPLERATE: Final[int] = 22100
DURATION: Final[float] = 4.0
N_SAMPLES: Final[int] = int(DURATION * SAMPLERATE)

WIN_SIZE: Final[float] = 0.032
HOP_SIZE: Final[float] = WIN_SIZE / 2
N_MFCC: Final[int] = 40


SETTINGS: Final[Settings] = Settings(SAMPLERATE, N_SAMPLES, WIN_SIZE, HOP_SIZE, N_MFCC)


def main():
    # Load model from memory or train a new one?
    USE_SAVE = True
    # Compute MFCCs or try to first get them from cache?
    COMPUTE_MFCCS_EVERYTIME = False

    HIGHLIGHT_INCORRECT = True

    # Use the final folders?
    USE_FINAL = True

    model = NN.Model(
        SETTINGS,
        getFullPath(WORKING_DIR, MODEL_DIR, FINAL_DIR, MODEL_NAME) if USE_FINAL else
        getFullPath(WORKING_DIR, MODEL_DIR, MODEL_NAME),
        useCachedValues=not COMPUTE_MFCCS_EVERYTIME,
        useSave=USE_SAVE
    )

    # Get Paths to folders containing data
    trainDirPath = getFullPath(WORKING_DIR, DATA_DIR, FINAL_DIR, TRAIN_DIR) if USE_FINAL else \
        getFullPath(WORKING_DIR, DATA_DIR, TRAIN_DIR)

    validateDirPath = getFullPath(WORKING_DIR, DATA_DIR, FINAL_DIR, TEST_DIR) if USE_FINAL else \
        getFullPath(WORKING_DIR, DATA_DIR, TEST_DIR)

    # Logic for using the model saved in Memory or training one from scratch
    if not USE_SAVE:
        model.importTrain(trainDirPath)
        model.importValidation(validateDirPath)

        historyPath = getFullPath(WORKING_DIR, MODEL_DIR, FINAL_DIR, HISTORY_NAME) if USE_FINAL else \
            getFullPath(WORKING_DIR, MODEL_DIR, HISTORY_NAME)

        model.train(13, 100, saveHistory=historyPath)
    else:
        # If we imported from memory we still have to initialize the labels
        # to get predictions etc
        model.importLabelsFrom(trainDirPath)

    # Predictions:
    testPath = getFullPath(WORKING_DIR, DATA_DIR, FINAL_DIR, OWN_DIR) if USE_FINAL else validateDirPath
    for prediction in model.predictionsFor(testPath):
        correct = prediction.gotLabel == prediction.correctLabel if HIGHLIGHT_INCORRECT else True
        print(prediction, file=sys.stdout if correct else sys.stderr)


if __name__ == '__main__':
    main()