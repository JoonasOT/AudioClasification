from os import walk


def getFilesInDir(directory: str) -> list:
    f = []
    for (dirpath, dirnames, filenames) in walk(directory):
        f.extend([dirpath + "/" + file for file in filenames])

    return [path.replace("\\", "/") for path in f]


def removeDotFiles(dirs: list[str]) -> list[str]:
    return list(filter(lambda path: path.split("/")[-1][0] != ".", dirs))


def onlyWavFiles(dirs: list[str]) -> list[str]:
    assert sum([int(len(p) > 4) for p in dirs]) == len(dirs)
    return list(filter(lambda path: path[-4:] == ".wav", dirs))