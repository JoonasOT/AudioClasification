from src.classes import *


def main():
    m = Maybe(AudioSignal("./data/x.wav"))
    print(m)


if __name__ == '__main__':
    main()
