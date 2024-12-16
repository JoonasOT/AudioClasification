import sys
import os
import zipfile


def zipsToCollections(*args):
    assert len(args) == 3, "Too few arguments!\nUsage: >zipsToCollections <folder containing zips> <folder for " \
                               "extraction> <classes as \"class1:class2:class3...\" "
    infolder, outfolder, classes = args
    classes = classes.split(":")
    NO_CLASS = "misc"

    for class_ in classes + [NO_CLASS]:
        try:
            os.mkdir(outfolder + "/" + class_)
        except FileExistsError:
            pass
        except PermissionError:
            print(f"Permission denied: Unable to create '{class_}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

    zips = os.listdir(infolder)
    for zip_ in zips:
        if not zipfile.is_zipfile(infolder + "/" + zip_):
            continue
        c = list(filter(lambda t: t[0] > 0, map(lambda class_: (zip_.count(class_), class_), classes)))
        c = NO_CLASS if len(c) != 1 else c[0][1]
        with zipfile.ZipFile(infolder + "/" + zip_, 'r') as zr:
            zr.extractall(outfolder + "/" + c)
    print("Extracted zips!")


def getLicences(*args):
    infolder, outFile, licenceFiles = args

    out = ""
    for zip_ in os.listdir(infolder):
        if not zipfile.is_zipfile(infolder + "/" + zip_):
            continue

        with zipfile.ZipFile(infolder + "/" + zip_, 'r') as zr:
            with zr.open(licenceFiles, "r") as lf:
                out += lf.read().decode("utf-8") + "\n"
    with open(outFile, "w") as o:
        o.write(out)
    print("Licences extracted!")


FUNCS = {
    "zipsToCollections": zipsToCollections,
    "getLicences": getLicences
}


if __name__ == '__main__':
    if sys.argv[1] not in FUNCS:
        raise ValueError(f"Incorrect function name '{sys.argv[1]}'!\nValid functions are: {', '.join(FUNCS.keys())}")
    FUNCS[sys.argv[1]](*tuple(sys.argv[2:]))