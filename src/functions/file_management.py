from os import walk

from typing_extensions import Final

WORKING_DIR: Final[str] = "."


def getFullPath(*steps) -> str:
    return "/".join(list(filter(lambda dir_: dir_ != "", steps)))


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