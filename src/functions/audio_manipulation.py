from enum import Enum

from scipy import signal

from src.dependencies.dependencies import *

# Required functions:
from src.functions.transfroms import *
from src.functions.plotting import *


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


def getSpectralBandwidth(audio: Maybe[AudioSignal], nFFT: int = 20, winSize: float = 0.030,
                         hopSize: float = 0.015, plot: bool = False) \
        -> Maybe[np.ndarray]:
    return audio \
        .transform(spectralBandwidth, False, nFFT, winSize, hopSize) \
        .run(
            conditionalRunner(cond=plot, func=plotSpectrogram),
            False,
            audio.transform(AudioSignal.getSamplerate).orElse(2.0) / 2,
            audio.transform(lambda as_: (1 / as_.getSamplerate()) * len(as_.getSignal())).orElse(1.0),
            "Spectral Bandwidth -- " + audio.transform(AudioSignal.getName).orElse("")
        )