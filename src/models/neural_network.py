from __future__ import annotations
from typing import NamedTuple, Final, Callable

import sys
from .environment import *

import tensorflow
import keras

import numpy as np
import json

from src.dependencies.dependencies import Maybe
from ..structures.audiosignal import AudioSignal

from src.models.common import Settings
from src.functions.file_management import onlyWavFiles, getFilesInDir
from src.functions.audio_manipulation import getNormalizedAudio, limitSamplesTo, sampleTo,\
                                             getMFCC, getSpectralCentroid


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
        # print(file)
        audio = getNormalizedAudio(file).transformers(
            sampleTo(self.settings.samplerate),
            limitSamplesTo(self.settings.samples)
        )
        if audio.transform(AudioSignal.getSignal).transform(lambda s: len(s) < self.settings.samples).unwrap():
            raise IOError(f"File {file} is too short! The file needs to be atleast "
                          f"{self.settings.samples * 1.0/self.settings.samplerate:.2f} s long!")

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
        data = np.zeros((xMax, yMax, zMax))     # data[x][y] = list of the outputs at (x, y)
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
