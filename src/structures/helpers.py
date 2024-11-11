from typing import NamedTuple

from .audiosignal import AudioSignal
from numpy import ndarray


class SignalWithTransform(NamedTuple):
    audio: AudioSignal
    transfrom: ndarray
