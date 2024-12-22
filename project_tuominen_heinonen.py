from __future__ import annotations

from enum import Enum
from typing import NamedTuple, Final, Callable
import sys
import os
from os import walk

from scipy import signal

from src.functions.plotting import plotSignal, plotSpectrum, plotSpectrogram, plotSpectralCentroid

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow
import keras

import numpy as np
import json
from scipy.io.wavfile import read as read_wav, write as write_wav
from numpy import ndarray, sqrt, mean, float64, shape
import librosa
from sounddevice import play
from scipy.fftpack import fft as fft_
from FunctionalMaybe import FunctionalMaybe as Maybe

def __timeToSamples(audio: AudioSignal) -> Callable[[float], int]:
    return lambda v: int(audio.getSamplerate() * v)


def fft(audio: AudioSignal, nfft: int = None) -> np.ndarray:
    if not nfft:
        nfft = len(audio.signal)
    T = fft_(audio.signal, n=nfft)
    return T[:len(T) // 2]


def stft(audio: AudioSignal, winLen: float, hopLen: float) -> np.ndarray:
    ts = __timeToSamples(audio)
    return librosa.stft(
        audio.signal,
        n_fft=ts(winLen),
        win_length=ts(winLen),
        hop_length=ts(hopLen)
    )


def mfcc(audio: AudioSignal, nMFCC, winLen: float, hopLen: float) -> np.ndarray:
    ts = __timeToSamples(audio)
    return librosa.feature.mfcc(
        y=audio.signal,
        sr=audio.samplerate,
        n_mfcc=nMFCC,
        n_fft=ts(winLen),
        win_length=ts(winLen),
        hop_length=ts(hopLen)
    )


def spectralCentroid(audio: AudioSignal, nFFT: int, winLen: float, hopLen: float) -> np.ndarray:
    ts = __timeToSamples(audio)
    return librosa.feature.spectral_centroid(
        y=audio.signal,
        sr=audio.getSamplerate(),
        win_length=ts(winLen),
        hop_length=ts(hopLen),
        n_fft=nFFT
    )

def spectralBandwidth(audio: AudioSignal, nFFT: int, winLen: float, hopLen: float):
    ts = __timeToSamples(audio)
    return librosa.feature.spectral_bandwidth(
        y=audio,
        sr=audio.getSamplerate,
        win_length=ts(winLen),
        hop_length=ts(hopLen),
        n_fft=nFFT
    )


def getAmplitude(T: np.ndarray) -> np.ndarray:
    return np.abs(T)


def amplitudeToDB(T: np.ndarray) -> np.ndarray:
    return librosa.amplitude_to_db(T)


def getFilesInDir(directory: str) -> list:
    f = []
    for (dirpath, dirnames, filenames) in walk(directory):
        f.extend([dirpath + "/" + file for file in filenames])
    f = [path.replace("\\", "/") for path in f]
    f.sort()
    return f


def removeDotFiles(dirs: list[str]) -> list[str]:
    return list(filter(lambda path: path.split("/")[-1][0] != ".", dirs))


def onlyWavFiles(dirs: list[str]) -> list[str]:
    assert sum([int(len(p) > 4) for p in dirs]) == len(dirs)
    return list(filter(lambda path: path[-4:] == ".wav", dirs))


class FreqType(Enum):
    BASE = 0
    AMPLITUDE = 1
    DECIBEL = 2


def conditionalRunner(cond: bool, func: Callable, funcOther=lambda *args, **kvargs: None):
    return func if cond else funcOther


def getNormalizedAudio(file: str, plot: bool = False) -> Maybe[AudioSignal]:
    return Maybe(file) \
        .construct(AudioSignal, False) \
        .transform(AudioSignal.normalize) \
        .run(conditionalRunner(cond=plot, func=plotSignal))


def sampleTo(sampleRate: int) -> Callable[[AudioSignal], AudioSignal]:
    def __(as_: AudioSignal):
        as_.signal = signal.resample(as_.getSignal(), int(sampleRate / as_.getSamplerate() * len(as_.getSignal())))
        as_.samplerate = sampleRate
        return as_

    return __


def limitSamplesTo(N: int) -> Callable[[AudioSignal], AudioSignal]:
    def __(as_: AudioSignal):
        as_.signal = as_.getSignal()[:N]
        return as_

    return __


def getSpectrum(audio: Maybe[AudioSignal], freqT: FreqType = FreqType.DECIBEL, plot: bool = False) -> Maybe[np.ndarray]:
    return audio \
        .transformers(*((fft, getAmplitude, amplitudeToDB)[:freqT.value + 1])) \
        .run(
            conditionalRunner(cond=plot, func=plotSpectrum),
            False,
            audio.transform(AudioSignal.getSamplerate).orElse(44100),
            "Spectrum -- " + audio.transform(AudioSignal.getName).orElse(None),
            isDB=freqT == FreqType.DECIBEL
        )


def getSpectrogram(audio: Maybe[AudioSignal], freqT: FreqType = FreqType.DECIBEL,
                   winSize: float = 0.030, hopSize: float = 0.015, plot: bool = False) -> Maybe[np.ndarray]:
    return audio \
        .transform(stft, False, winSize, hopSize) \
        .transformers(*((getAmplitude, amplitudeToDB)[:freqT.value])) \
        .run(
            conditionalRunner(cond=plot, func=plotSpectrogram),
            False,
            audio.transform(AudioSignal.getSamplerate).orElse(2.0) / 2,
            audio.transform(lambda as_: (1 / as_.getSamplerate()) * len(as_.getSignal())).orElse(1.0),
            "Spectrogram -- " + audio.transform(AudioSignal.getName).orElse(None)
        )


def getMFCC(audio: Maybe[AudioSignal], nMFFC: int = 20, winSize: float = 0.030, hopSize: float = 0.015,
            plot: bool = False) \
        -> Maybe[np.ndarray]:
    return audio \
        .transform(mfcc, False, nMFFC, winSize, hopSize) \
        .run(
            conditionalRunner(cond=plot, func=plotSpectrogram),
            False,
            audio.transform(AudioSignal.getSamplerate).orElse(2.0) / 2,
            audio.transform(lambda as_: (1 / as_.getSamplerate()) * len(as_.getSignal())).orElse(1.0),
            "MFCC -- " + audio.transform(AudioSignal.getName).orElse("")
        )


def getSpectralCentroid(audio: Maybe[AudioSignal], nFFT: int = 20, winSize: float = 0.030,
                        hopSize: float = 0.015, plot: bool = False) \
        -> Maybe[np.ndarray]:
    return audio \
        .transform(spectralCentroid, False, nFFT, winSize, hopSize) \
        .run(
            conditionalRunner(cond=plot, func=plotSpectralCentroid),
            False,
            audio.transform(AudioSignal.getTime).orElse(1.0),
            "Spectral Centroid -- " + audio.transform(AudioSignal.getName).orElse("")
        )


class AudioSignal:
    def __init__(self, file: str):
        self.name = file
        content = read_wav(file)
        self.samplerate: int = content[0]
        if content[1].ndim > 1:
            self.signal: ndarray = float64(content[1].T[0])
        else:
            self.signal: ndarray = float64(content[1])

    def rmse(self) -> float:
        return sqrt(mean(self.signal**2))

    def normalize(self) -> AudioSignal:
        self.signal = librosa.util.normalize(self.signal)
        return self

    def __str__(self) -> str:
        return f"AudioSignal[{self.name}, Signal: {self.signal}, Samplerate: {self.samplerate}]"

    def getName(self) -> str:
        return self.name

    def getSignal(self) -> ndarray:
        return self.signal

    def getSamplerate(self) -> int:
        return self.samplerate

    def write(self, file: str):
        return write_wav(file, self.samplerate, self.signal)

    def play(self, wait=True) -> None:
        play(self.signal, self.samplerate, blocking=wait)

    def getTime(self) -> float:
        return (1.0 / self.samplerate) * len(self.signal)

class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ModelData(NamedTuple):
    labels: np.ndarray
    data: list[np.ndarray]
    ref: Model

    def get(self):
        # If you get an error about the data being inhomogenous it is most likely due to having
        # included a too short of an audio clip. All audio clips need to be ATLEAST as long as the
        # set length in the Settings object given to the Model constructor
        return np.array(self.data), tensorflow.one_hot(self.labels, depth=self.ref.ONE_HOT_DEPTH)


class Prediction(NamedTuple):
    file: str
    weights: np.ndarray
    gotLabel: str
    correctLabel: str

    def getConfidence(self) -> str:
        return f"{np.max(self.weights) * 100:.2f}"

    def isCorrect(self) -> bool:
        return self.gotLabel == self.correctLabel

    def __str__(self):
        HIGHLIGHT = '\033[1;4;97m'
        WRONG = '\033[1;31m'
        NORM = '\033[0;0m'
        COLOR = WRONG if self.gotLabel != self.correctLabel else NORM
        Values = [COLOR + v + NORM for v in [
            self.gotLabel.rjust(4),
            self.correctLabel.rjust(4),
            self.getConfidence().rjust(6) + "%",
            self.file
        ]]
        Labels = [HIGHLIGHT + v + NORM for v in ["Got", "Was", "Confidence", "File"]]
        return "Pred[" + ", ".join([f"{v}: {Values[i]}" for i, v in enumerate(Labels)]) + "]"


class Model:
    DEFAULT_LABEL_GETTER: Final[Callable[[str], str]] = lambda str_: str_.split("/")[-2]

    def __init__(self, settings: Settings, modelDir: str, useCachedValues: bool = False, useSave: bool = False):
        self.settings: Settings = settings

        self.labels: dict[str, int] = {}
        self.ONE_HOT_DEPTH: int = 0

        self.trainData: ModelData = None
        self.validationData: ModelData = None
        self.useCache = useCachedValues

        self.modelDir: str = modelDir
        self.modelCallbacks: list = []

        self.model: keras.models.Sequential = None
        self.useSave = useSave
        if useSave:
            self.__createModel(True)

    def __normalize(self, arr: np.ndarray) -> np.ndarray:
        max_ = np.max([np.abs(np.min(arr)), np.max(arr)])
        arr /= 2 * max_
        return arr + 0.5

    def __getInputs(self, file: str) -> np.ndarray:
        audio = getNormalizedAudio(file).transformers(
            sampleTo(self.settings.samplerate),
            limitSamplesTo(self.settings.samples)
        )
        if audio.transform(AudioSignal.getSignal).transform(lambda s: len(s) < self.settings.samples).unwrap():
            raise IOError(f"File {file} is too short! The file needs to be atleast "
                          f"{self.settings.samples * 1.0 / self.settings.samplerate:.2f} s long!")

        # Get MFCC constants
        mfcc = audio.construct(Maybe, False).transform(
            getMFCC, False,
            self.settings.binCount,
            self.settings.winLen,
            self.settings.hopSize
        )

        spectralCentroids = audio.construct(Maybe, False).transform(
            getSpectralCentroid, False,
            int(self.settings.winLen * self.settings.samplerate),
            self.settings.winLen,
            self.settings.hopSize
        )

        # Normalize the data
        mfcc = self.__normalize(mfcc.unwrap().unwrap()).T
        spectralCentroids = np.repeat(self.__normalize(spectralCentroids.unwrap().unwrap()),
                                      self.settings.binCount,
                                      axis=0).T

        # What we use as outputs or inputs for NN
        outputs = [mfcc, spectralCentroids]

        # Remap the outputs from separate matricies to a tensor
        xMax, yMax, zMax = len(mfcc), len(mfcc[0]), len(outputs)
        data = np.zeros((xMax, yMax, zMax))  # data[x][y] = list of the outputs at (x, y)
        for x in range(xMax):
            for y in range(yMax):
                for z in range(zMax):
                    data[x, y, z] = outputs[z][x][y]

        return data

    def __cacheDataRead(self, file: str) -> tuple[list[str], list]:
        with open(file, 'r') as f:
            cache = json.loads(f.read())
            labels = cache["labels"]
            out = cache["out"]
            for i in range(len(out)):
                out[i] = np.array(out[i])
            return labels, out

    def __cacheDataWrite(self, labels: list[str], out: list, file: str) -> None:
        with open(file, 'w') as f:
            f.write(json.dumps({"labels": labels, "out": out}, cls=json_serialize))

    def __importData(self, dir_: str, labelGetter=DEFAULT_LABEL_GETTER, createCache=True):
        labels: list[str] = []
        out: list = []

        cachedFile = "/".join(dir_.split("/")[:-1]) + "/cache/" + dir_.split("/")[-1] + ".json"
        if self.useCache and os.path.exists(cachedFile):
            print(f"Reading from cache {cachedFile}")
            labels, out = self.__cacheDataRead(cachedFile)
            labels = [labelGetter(label) for label in labels]

        else:
            # Flush the std streams so if we get error messages we can be quite certain
            # that they were from the following the files exectuded here
            sys.stdout.flush()
            sys.stderr.flush()

            for file in onlyWavFiles(getFilesInDir(dir_)):
                label = labelGetter(file)
                labels.append(label)

                out.append(self.__getInputs(file))
            # Return the labels, mfccs and others
            sys.stdout.flush()
            sys.stderr.flush()

            if createCache:
                print(f"Creating cache {cachedFile}")
                self.__cacheDataWrite(onlyWavFiles(getFilesInDir(dir_)), out, cachedFile)

        return labels, out

    def __createLabels(self, labels: list[str]) -> None:
        __labelMax = 0
        for label in labels:
            if label not in self.labels:
                self.labels[label] = __labelMax
                __labelMax += 1
        self.ONE_HOT_DEPTH = __labelMax

    def importLabelsFrom(self, dir_: str, labelGetter=DEFAULT_LABEL_GETTER) -> None:
        print(f"Importing labels from {dir_}")
        self.__createLabels([labelGetter(file) for file in onlyWavFiles(getFilesInDir(dir_))])

    def importTrain(self, dir_: str) -> None:
        print("Importing training data")
        labels, data = self.__importData(dir_)

        self.__createLabels(labels)
        self.trainData = ModelData(
            np.array(list(map(lambda s: self.labels[s], labels))),
            data,
            self
        )
        self.__createModel(False, np.shape(data[0]))

    def importValidation(self, dir_: str) -> None:
        print("Importing validation data")
        labels, data = self.__importData(dir_)

        self.validationData = ModelData(
            np.array(list(map(lambda s: self.labels[s], labels))),
            data,
            self
        )

    def __createModel(self, useSave: bool = True, dataDim: tuple = None) -> None:
        assert self.model is None, "The underlying model has been already constructed!"

        if useSave:
            print(f"Creating model from {self.modelDir}")
            self.model = keras.models.load_model(self.modelDir)
            return

        print(f"Creating new model based on training data")
        self.model = keras.Sequential(
            [
                keras.layers.Input(dataDim),

                keras.layers.Conv2D(256, (3, 3), padding='same', activation="relu"),
                keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'),

                keras.layers.Conv2D(512, (3, 3), padding='same', activation="relu"),
                keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'),
                keras.layers.Dropout(0.25),

                keras.layers.Conv2D(1024, (3, 3), padding='same', activation="relu"),
                keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'),
                keras.layers.Dropout(0.25),

                keras.layers.Flatten(),

                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dropout(0.25),

                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.25),

                keras.layers.BatchNormalization(),

                keras.layers.Dense(self.ONE_HOT_DEPTH, activation="softmax")
            ]
        )

        # Create a printout of the created model
        print("Created model:")
        self.model.summary()
        print()
        sys.stdout.flush()

        # Compile the model for training
        self.model.compile(
            optimizer='adam',
            loss=keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        # Function for stopping before overfitting
        self.modelCallbacks.append(keras.callbacks.EarlyStopping(patience=3))
        # Function for storing the model checkpoints to memory
        self.modelCallbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=self.modelDir,
                save_weights_only=False,
                monitor='loss',
                mode='min',
                save_best_only=True
            )
        )

    def train(self, epochs: int, steps: int, saveHistory: str = None):
        print(f"Training Neural Network (for {', '.join(self.labels.keys())}):")
        trainingDataset = tensorflow.data.Dataset.from_tensor_slices(self.trainData.get())
        validationDataset = tensorflow.data.Dataset.from_tensor_slices(self.validationData.get())
        history = self.model.fit(
            trainingDataset.shuffle(len(self.trainData.labels)).batch(128).repeat(),
            epochs=epochs,
            steps_per_epoch=steps,
            validation_data=validationDataset.shuffle(len(self.validationData.labels)).batch(128).repeat(),
            validation_steps=2,
            callbacks=self.modelCallbacks
        )
        if saveHistory is not None:
            with open(saveHistory, "w") as f:
                f.write(json.dumps(history.history))
        return history

    def predict(self, file: str) -> Prediction:
        print(f"Getting prediction for {file}")
        p = self.model.predict(np.expand_dims(self.__getInputs(file), axis=0))
        return Prediction(file, p, self.preditionToLabel(p))

    def predictionsFor(self, dir_: str) -> list[Prediction]:
        print(f"Getting predictions for files in {dir_}")
        lG = Model.DEFAULT_LABEL_GETTER if not self.useCache else lambda str_: str_.split("/")[-2]

        labels, data = self.__importData(dir_, labelGetter=lG, createCache=False)
        files = onlyWavFiles(getFilesInDir(dir_))

        ps = self.model.predict(np.array(data))
        return [Prediction(files[i], ps[i], self.preditionToLabel(ps[i]), labels[i]) for i in range(len(labels))]

    def preditionToLabel(self, pred: np.ndarray) -> str:
        return list(map(lambda t: t[0], filter(lambda t: t[1] == np.argmax(pred), self.labels.items())))[0]

WORKING_DIR: Final[str] = "."

def getFullPath(*steps) -> str:
    return "/".join(list(filter(lambda dir_: dir_ != "", steps)))

class Settings(NamedTuple):
    samplerate: int
    samples: int
    winLen: float
    hopSize: float
    binCount: int

DATA_DIR: Final[str] = "data"
TEST_DIR: Final[str] = "test"
OWN_DIR: Final[str] = "own"
TRAIN_DIR: Final[str] = "train"

MODEL_DIR: Final[str] = "models"
MODEL_NAME: Final[str] = "model.keras"
HISTORY_NAME: Final[str] = "history.json"
PREDICTION_NAME: Final[str] = "predictions.csv"

FINAL_DIR: Final[str] = "final"

SAMPLERATE: Final[int] = 22100  # Hz
DURATION: Final[float] = 4.0  # s
N_SAMPLES: Final[int] = int(DURATION * SAMPLERATE)  # kpl

WIN_SIZE: Final[float] = 0.032  # s
HOP_SIZE: Final[float] = WIN_SIZE / 2  # s
N_MFCC: Final[int] = 40  # kpl

SETTINGS: Final[Settings] = Settings(SAMPLERATE, N_SAMPLES, WIN_SIZE, HOP_SIZE, N_MFCC)


def main():
    model = Model(
        SETTINGS,
        "./models/final/model.keras",
        useCachedValues=False,
        useSave=True
    )

    # Get Paths to folders containing data
    trainDirPath = "./data/final/train"

    model.importLabelsFrom("./data/final/own")
    # Predictions:
    testPath = "./data/final/own"
    results: list[Prediction] = []
    for prediction in model.predictionsFor(testPath):
        results.append(prediction)
        print(prediction)
    print(f"\nAccuracy: {100 * (sum(map(lambda r: r.isCorrect(), results)) / len(results)):.2f}%")


if __name__ == '__main__':
    main()