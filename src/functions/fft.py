from ..structures.audiosignal import AudioSignal
from ..structures.helpers import SignalWithTransform
from scipy.fftpack import fft as fft_, fftshift
from numpy import abs, log10


def fft(audio: AudioSignal) -> SignalWithTransform:
    N = len(audio.signal)
    T = fftshift(fft_(audio.signal, n=N))
    T /= N // 2
    return SignalWithTransform(audio, abs(T))


def amplitudeToDB(st: SignalWithTransform) -> SignalWithTransform:
    return SignalWithTransform(st.audio, 10 * log10(st.transfrom))
