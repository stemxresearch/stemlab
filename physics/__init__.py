from .core import newtons_second_law, ohms_law


def __dir__():
    from stemlab.utils import get_public_names
    return get_public_names(globals_dict=globals())


# https://www.omnicalculator.com/physics