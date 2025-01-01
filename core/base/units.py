from stemlab.core.arraylike import list_join
from stemlab.core.display import Result
from stemlab.core.base.numbers import num_floatify
from stemlab.core.base.strings import str_separate_alphanumeric


def si_units(x: float | int | str, quantity: str, si_unit: str) -> Result:
    
    x = num_floatify(x)
    if isinstance(x, float):
        return x, 'K'
    else:
        x, units = str_separate_alphanumeric(x=x, si_units=si_unit)
    
    if quantity.startswith('t'):
        if units.startswith('k'):
            pass
        elif units.startswith('k'):
            pass
        else:
            pass
    elif quantity.startswith('l'):
        if units.startswith('k'):
            pass
        elif units.startswith('k'):
            pass
        else:
            pass
    elif quantity.startswith('p'):
        if units.startswith('k'):
            pass
        elif units.startswith('k'):
            pass
        else:
            pass
    elif quantity.startswith('v'):
        if units.startswith('k'):
            pass
        elif units.startswith('k'):
            pass
        else:
            pass
    else:
        valid_quantities = list_join(
            lst=['temperature', 'length', 'pressure']
        )
        raise ValueError(
            f"Expected 'quantity' to be {valid_quantities} but got: {quantity}"
        )
        
    result = Result(x=x, units=units)
    
    return result