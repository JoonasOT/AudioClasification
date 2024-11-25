from sys import stderr
try:
    from FunctionalMaybe import FunctionalMaybe as Maybe
except ModuleNotFoundError as E:
    try:
        from .FunctionalMaybe import FunctionalMaybe as Maybe
        print(f"The package functional-maybe is not installed. Using the submodule!", file=stderr)
    except ModuleNotFoundError as e:
        raise Exception("Project doesn't contain the FunctionalMaybe-module."
                        "Either clone the submodules of this repository or install "
                        "package \"functional-maybe\" with pip.")
