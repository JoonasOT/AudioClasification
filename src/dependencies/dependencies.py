try:
    from FunctionalMaybe import FunctionalMaybe as Maybe
except ModuleNotFoundError as E:
    from .FunctionalMaybe import FunctionalMaybe as Maybe