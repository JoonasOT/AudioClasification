from typing import NamedTuple, Final
from math import log2

import numpy as np
import os

from src.dependencies.dependencies import Maybe

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow
import keras

from src.functions.audio_manipulation import getMFCC, getNormalizedAudio, limitSamplesTo, sampleTo
from src.functions.file_management import onlyWavFiles, getFilesInDir
from src.models.common import Settings


ONE_HOT_DEPTH: Final[int] = 2


class ModelData(NamedTuple):
    labels: np.ndarray
    data: list[np.ndarray]

    def get(self):
        return np.array(self.data), tensorflow.one_hot(self.labels, depth=ONE_HOT_DEPTH)


class Model:
    def __init__(self, settings: Settings, modelDir: str, useSave: bool = False):
        self.settings: Settings = settings

        self.labels: dict[str, int] = {}

        self.trainData: ModelData = None
        self.validationData: ModelData = None

        self.modelDir: str = modelDir
        self.modelCallbacks: list = []

        self.model: keras.models.Sequential = None
        self.__createModel(useSave)

    def __normalize(self, arr: np.ndarray) -> np.ndarray:
        max_ = np.max([np.abs(np.min(arr)), np.max(arr)])
        arr /= 2 * max_
        return arr + 0.5

    def __importData(self, dir_: str):
        labels: list[str] = []
        mfccs: list[np.ndarray] = []
        spectralBands: list[np.ndarray] = []
        spectralCentroids: list[np.ndarray] = []

        out: list = []

        for file in onlyWavFiles(getFilesInDir(dir_)):
            label = file.split("/")[-2]
            audio = getNormalizedAudio(file).transformers(
                        sampleTo(self.settings.samplerate),
                        limitSamplesTo(self.settings.samples)
            )
            # Get MFCC constants
            mfcc = audio.construct(Maybe, False).transform(
                getMFCC, False,
                self.settings.binCount,
                self.settings.winLen,
                self.settings.hopSize
            )

            # Normalize the data
            mfcc = self.__normalize(mfcc.unwrap().unwrap()).T

            # Set the data
            labels.append(label)
            mfccs.append(mfcc)

            # spectralBands = ...
            # spectralCentroids = ...

            outputs = [mfcc]

            xMax, yMax, zMax = len(mfcc), len(mfcc[0]), len(outputs)
            data = np.zeros((xMax, yMax, zMax))
            for x in range(xMax):
                for y in range(yMax):
                    for z in range(zMax):
                        data[x, y, z] = outputs[z][x][y]
            out.append(data)
        # Return the labels, mfccs and others
        return labels, out

    def __createLabels(self, labels: list[str]) -> None:
        __labelMax = 0
        for label in labels:
            if label not in self.labels:
                self.labels[label] = __labelMax
                __labelMax += 1

    def importTrain(self, dir_: str) -> None:
        labels, data = self.__importData(dir_)

        self.__createLabels(labels)
        self.trainData = ModelData(
            np.array(list(map(lambda s: self.labels[s], labels))),
            data
        )

    def importValidation(self, dir_: str) -> None:
        labels, data = self.__importData(dir_)

        self.validationData = ModelData(
            np.array(list(map(lambda s: self.labels[s], labels))),
            data
        )

    def __createModel(self, useSave: bool = True) -> None:
        if useSave:
            self.model = keras.models.load_model(self.modelDir)
            return

        self.model = keras.Sequential(
            [
                keras.layers.Conv2D(126, (3, 3), padding='same', activation="relu", input_shape=(126, 40, 1)),
                keras.layers.MaxPooling2D((2, 2), strides=2),

                keras.layers.Conv2D(64, (3, 3), padding='same', activation="relu"),
                keras.layers.MaxPooling2D((2, 2), strides=2),

                keras.layers.Flatten(),
                keras.layers.Dense(100, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(2, activation="softmax")
            ]
        )
        self.model.summary()
        self.model.compile(
            optimizer='adam',
            loss=keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        # Function for stopping before overfitting
        self.modelCallbacks.append(keras.callbacks.EarlyStopping(patience=2))
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

    def train(self, epochs: int):

        return self.model.fit(
            tensorflow.data.Dataset.from_tensor_slices(self.trainData.get()).batch(128).repeat(),
            epochs=epochs,
            steps_per_epoch=500,
            validation_data=tensorflow.data.Dataset.from_tensor_slices(self.validationData.get()).batch(128).repeat(),
            validation_steps=2,
            callbacks=self.modelCallbacks
        )
