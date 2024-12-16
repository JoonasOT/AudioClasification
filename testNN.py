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
PREDICTION_NAME: Final[str] = "predictions.csv"

FINAL_DIR: Final[str] = "final"


SAMPLERATE: Final[int] = 22100#Hz
DURATION: Final[float] = 4.0#s
N_SAMPLES: Final[int] = int(DURATION * SAMPLERATE)#kpl

WIN_SIZE: Final[float] = 0.032#s
HOP_SIZE: Final[float] = WIN_SIZE / 2#s
N_MFCC: Final[int] = 40#kpl


SETTINGS: Final[Settings] = Settings(SAMPLERATE, N_SAMPLES, WIN_SIZE, HOP_SIZE, N_MFCC)


def main():
    # Load model from memory or train a new one?
    USE_SAVE = True
    # Compute MFCCs or try to first get them from cache?
    COMPUTE_MFCCS_EVERYTIME = False

    # Use the final folders?
    USE_FINAL = True
    FINAL_ = FINAL_DIR if USE_FINAL else ""

    # Save prediction results to file?
    SAVE_PREDICTIONS = True

    model = NN.Model(
        SETTINGS,
        getFullPath(WORKING_DIR, MODEL_DIR, FINAL_, MODEL_NAME),
        useCachedValues=not COMPUTE_MFCCS_EVERYTIME,
        useSave=USE_SAVE
    )

    # Get Paths to folders containing data
    trainDirPath = getFullPath(WORKING_DIR, DATA_DIR, FINAL_, TRAIN_DIR)
    validateDirPath = getFullPath(WORKING_DIR, DATA_DIR, FINAL_, TEST_DIR)

    # Logic for using the model saved in Memory or training one from scratch
    if not USE_SAVE:
        model.importTrain(trainDirPath)
        model.importValidation(validateDirPath)

        historyPath = getFullPath(WORKING_DIR, MODEL_DIR, FINAL_, HISTORY_NAME)
        model.train(13, 100, saveHistory=historyPath)
    else:
        # If we imported from memory we still have to initialize the labels
        # to get predictions etc
        model.importLabelsFrom(trainDirPath)

    # Predictions:
    testPath = getFullPath(WORKING_DIR, DATA_DIR, FINAL_DIR, OWN_DIR) if USE_FINAL else validateDirPath
    results: list[NN.Prediction] = []
    for prediction in model.predictionsFor(testPath):
        results.append(prediction)
        print(prediction)

    # Save the predictions
    if SAVE_PREDICTIONS:
        OUT_PATH = getFullPath(WORKING_DIR, MODEL_DIR, FINAL_, PREDICTION_NAME)
        HEADERS = ["Got", "Was", "Confidence", "File"]
        DELIM = ";"
        out = DELIM.join(HEADERS) + "\n"
        for pred in results:
            out += DELIM.join([pred.gotLabel, pred.correctLabel, pred.getConfidence(), pred.file]) + "\n"
        with open(OUT_PATH, "w") as f:
            f.write(out)


if __name__ == '__main__':
    main()