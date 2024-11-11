from typing import Union
import numpy as np
import matplotlib.pyplot as plt

from ..structures.audiosignal import AudioSignal
from ..structures.helpers import SignalWithTransform


def plotSignal(audio: Union[AudioSignal, SignalWithTransform]) -> None:
    if isinstance(audio, SignalWithTransform):
        audio = audio.audio

    assert isinstance(audio, AudioSignal)
    plt.figure()
    x_axis = np.linspace(0, (1 / audio.samplerate) * len(audio.signal), len(audio.signal))
    plt.plot(x_axis, audio.signal)
    plt.grid()
    plt.title(audio.name)
    plt.show(block=False)


def plotTransfrom(st: SignalWithTransform, positiveOnly=True) -> None:
    assert isinstance(st, SignalWithTransform)
    audio, transform = st.audio, st.transfrom
    plt.figure()
    if positiveOnly:
        N = len(transform)
        x_axis = np.linspace(0, audio.samplerate / 2, N // 2)
        plt.plot(x_axis, transform[N // 2 if not N % 2 else N // 2 + 1:])
    else:
        x_axis = np.linspace(-audio.samplerate / 2, audio.samplerate / 2, len(transform))
        plt.plot(x_axis, transform)
    plt.grid()
    plt.title(audio.name)
    plt.show(block=False)


def keepPlotsOpen():
    plt.show()
