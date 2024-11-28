import numpy as np
from typing import NamedTuple, Callable


biasWithCutoff = lambda N: lambda rd: np.concatenate([np.ones(N) / N, np.zeros(len(rd) - N)])
linearBias = lambda rd: np.linspace(1, -1, len(rd))


class Model:
    class Result(NamedTuple):
        label: str
        nearestLabels: list[tuple[str, float]]

        def getConfidence(self, bias: Callable[[list[float]], np.ndarray]) -> dict[str, float]:
            N: int = len(self.nearestLabels)
            assert N > 0, "No results stored!"
            minDist = self.nearestLabels[0][1]
            relDists = [tup[1] / minDist for tup in self.nearestLabels]
            certanties = np.full((N,), 1.0) * bias(relDists)

            out: dict[str, float] = {}
            for i in range(len(self.nearestLabels)):
                model = self.nearestLabels[i][0]
                if model in out:
                    out[model] += certanties[i]
                else:
                    out[model] = certanties[i]

            return out

    def __init__(self, data: dict[str, list[np.ndarray]], N: int = None):
        self.N = N if N is not None else sum([len(v) for k, v in data.items()], 0)
        self.data = {label: [np.mean(mfcc.T, axis=0) for mfcc in MFCCs] for label, MFCCs in data.items()}

    def test(self, label: str, mfcc: np.ndarray) -> Result:
        distsPerM: list[tuple[str, float]] = [
            (m, np.linalg.norm(np.mean(mfcc.T, axis=0) - mfcc_))
            for m, MFCCs in self.data.items() for mfcc_ in MFCCs
        ]
        distsPerM.sort(key=lambda tup: tup[1])
        return Model.Result(label, [tup for tup in distsPerM][:self.N])
