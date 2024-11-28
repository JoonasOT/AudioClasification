# Require Structures:
import json
from typing import Final, Callable

from src.models.common import getMFCCs
import src.models.nearest_neighbour as NearestNeighbour

DATA_DIR: Final[str] = "./data"
TEST_DIR: Final[str] = "/test"
TRAIN_DIR: Final[str] = "/train"

SAMPLERATE: Final[int] = 22100
N_SAMPLES: Final[int] = int(2.0 * SAMPLERATE)

WIN_SIZE: Final[float] = 0.032
HOP_SIZE: Final[float] = WIN_SIZE / 2
N_MFCC: Final[int] = 40

NEAREST_NEIGHBOUR_N: Final[int] = 20

MODEL_GETTER: Final[Callable[[str], str]] = lambda file: file.split("/")[-2]

PRINT_CONFIDENCES: Final[bool] = True
PRINT_RESULTS: Final[bool] = False
WRITE_RESULTS: Final[bool] = False


def main():
    trainData = getMFCCs(DATA_DIR + TRAIN_DIR, MODEL_GETTER, N_MFCC, WIN_SIZE, HOP_SIZE)
    testData = getMFCCs(DATA_DIR + TEST_DIR, MODEL_GETTER, N_MFCC, WIN_SIZE, HOP_SIZE)

    nearest = NearestNeighbour.Model(trainData, NEAREST_NEIGHBOUR_N)

    tests: list = []
    for m, mffcs in testData.items():
        for mfcc in mffcs:
            test = nearest.test(m, mfcc)
            tests.append(test)

            if PRINT_CONFIDENCES:
                print(json.dumps({
                    test.label:
                        {
                            'constant': test.getConfidence(NearestNeighbour.biasWithCutoff(NEAREST_NEIGHBOUR_N)),
                            'linear': test.getConfidence(NearestNeighbour.linearBias)
                        }
                    }, indent=2))

    results = json.dumps(tests, indent=4)

    if PRINT_RESULTS:
        print(results)

    if WRITE_RESULTS:
        with open("model.json", "w") as f:
            f.write(results)


if __name__ == '__main__':
    main()
