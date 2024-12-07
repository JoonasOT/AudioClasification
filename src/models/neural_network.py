from typing import NamedTuple, Final

import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow
import keras
import numpy as np


from src.dependencies.dependencies import Maybe
from src.functions.audio_manipulation import getMFCC, getNormalizedAudio, limitSamplesTo, sampleTo, getSpectralCentroid
from src.functions.file_management import onlyWavFiles, getFilesInDir
from src.models.common import Settings


ONE_HOT_DEPTH: Final[int] = 2


class ModelData(NamedTuple):
    labels: np.ndarray
    data: list[np.ndarray]

    def get(self):
        return np.array(self.data), tensorflow.one_hot(self.labels, depth=ONE_HOT_DEPTH)


class Prediction(NamedTuple):
    file: str
    weights: np.ndarray
    label: str


class Model:
    def __init__(self, settings: Settings, modelDir: str, useSave: bool = False):
        self.settings: Settings = settings

        self.labels: dict[str, int] = {}

        self.trainData: ModelData = None
        self.validationData: ModelData = None

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
        # Get MFCC constants
        mfcc = audio.construct(Maybe, False).transform(
            getMFCC, False,
            self.settings.binCount,
            self.settings.winLen,
            self.settings.hopSize
        )

        spectralCentroids = audio.construct(Maybe, False).transform(
            getSpectralCentroid, False,
            self.settings.binCount,
            self.settings.winLen,
            self.settings.hopSize
        )

        # Normalize the data
        mfcc = self.__normalize(mfcc.unwrap().unwrap()).T
        #spectralCentroids = self.__normalize(spectralCentroids.unwrap().unwrap()).T


        # What we use as outputs or inputs for NN
        outputs = [mfcc]

        xMax, yMax, zMax = len(mfcc), len(mfcc[0]), len(outputs)
        data = np.zeros((xMax, yMax, zMax))
        for x in range(xMax):
            for y in range(yMax):
                for z in range(zMax):
                    data[x, y, z] = outputs[z][x][y]

        return data

    def __importData(self, dir_: str, labelGetter=lambda str_: str_.split("/")[-2]):
        sys.stdout.flush()
        sys.stderr.flush()
        labels: list[str] = []
        out: list = []

        for file in onlyWavFiles(getFilesInDir(dir_)):
            label = labelGetter(file)
            labels.append(label)

            out.append(self.__getInputs(file))
        # Return the labels, mfccs and others
        return labels, out

    def __createLabels(self, labels: list[str]) -> None:
        __labelMax = 0
        for label in labels:
            if label not in self.labels:
                self.labels[label] = __labelMax
                __labelMax += 1

    def importLabelsFrom(self, dir_: str, labelGetter=lambda str_: str_.split("/")[-2]) -> None:
        print(f"Importing labels from {dir_}")
        self.__createLabels([labelGetter(file) for file in onlyWavFiles(getFilesInDir(dir_))])

    def importTrain(self, dir_: str) -> None:
        print("Importing training data")
        labels, data = self.__importData(dir_)

        self.__createLabels(labels)
        self.trainData = ModelData(
            np.array(list(map(lambda s: self.labels[s], labels))),
            data
        )
        self.__createModel(False, np.shape(data[0]))

    def importValidation(self, dir_: str) -> None:
        print("Importing validation data")
        labels, data = self.__importData(dir_)

        self.validationData = ModelData(
            np.array(list(map(lambda s: self.labels[s], labels))),
            data
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

                keras.layers.Conv2D(64, (3, 3), padding='same', activation="relu"),
                keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'),


                keras.layers.Conv2D(128, (3, 3), padding='same', activation="relu"),
                keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'),
                keras.layers.Dropout(0.25),

                keras.layers.Conv2D(256, (3, 3), padding='same', activation="relu"),
                keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'),
                keras.layers.Dropout(0.25),

                # Reduce the number of filters back to 128 -> Halves the number of weights in the next step
                keras.layers.Conv2D(128, (3, 3), padding='same', activation="relu"),
                keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'),
                keras.layers.Dropout(0.25),

                keras.layers.Flatten(),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.25),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(2, activation="softmax")
            ]
        )
        print("Created model:")
        self.model.summary()
        print()
        sys.stdout.flush()
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

    def train(self, epochs: int, steps: int):
        print("Training Neural Network:")
        return self.model.fit(
            tensorflow.data.Dataset.from_tensor_slices(self.trainData.get()).batch(128).repeat(),
            epochs=epochs,
            steps_per_epoch=steps,
            validation_data=tensorflow.data.Dataset.from_tensor_slices(self.validationData.get()).batch(128).repeat(),
            validation_steps=2,
            callbacks=self.modelCallbacks
        )

    def predict(self, file: str) -> Prediction:
        print(f"Getting prediction for {file}")
        p = self.model.predict(np.expand_dims(self.__getInputs(file), axis=0))
        return Prediction(file, p, self.preditionToLabel(p))

    def predictionsFor(self, dir_: str) -> list[Prediction]:
        print(f"Getting predictions for files in {dir_}")
        labels, data = self.__importData(dir_, labelGetter=lambda str_: str_.split("/")[-1])
        ps = self.model.predict(np.array(data))

        return [Prediction(labels[i], ps[i], self.preditionToLabel(ps[i])) for i in range(len(labels))]

    def preditionToLabel(self, pred: np.ndarray) -> str:
        return list(map(lambda t: t[0], filter(lambda t: t[1] == np.argmax(pred), self.labels.items())))[0]
