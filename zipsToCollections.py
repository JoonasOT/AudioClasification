import sys
import os
import zipfile


def main():
    assert len(sys.argv) == 4, "Too few arguments!\nUsage: >zipsToCollections <folder containing zips> <folder for " \
                               "extraction> <classes as \"class1:class2:class3...\" "
    infolder = sys.argv[1]
    outfolder = sys.argv[2]
    classes = sys.argv[3].split(":")
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


if __name__ == '__main__':
    main()
