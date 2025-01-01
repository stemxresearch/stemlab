from sympy import Symbol, flatten, sympify

from stemlab.core.arraylike import is_iterable
from stemlab.core.validators.errors import IterableError


def sym_get_variables(expr_array: str | list[str]) -> tuple[Symbol]:
    """
    Get unknown variables in a single or a system of equations.

    Parameters
    ----------
    expr_array : str or list of list of str
        Single or multiple equations represented as strings.

    Returns
    -------
    fvars : tuple of Symbol
        Tuple containing the unknown variables found in the equations.

    Raises
    ------
    IterableError
        If the input is not iterable.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.sym_get_expr_vars('x ** 2 + y')
    (x, y)
    >>> stm.sym_get_expr_vars(['x ** 2 + y', 'z - 1'])
    (x, y, z)

    """
    if not is_iterable(expr_array, includes_str=True):
        raise IterableError(par_name='expr_array', user_input=expr_array)
    if isinstance(expr_array, str):
        expr_array = str(expr_array).replace('=', '-')
    else:
        expr_array = [
            str(item).replace('=', '-') for item in flatten(expr_array)
        ]
    eqtns = sympify(expr_array)
    if not isinstance(eqtns, (list, tuple)):
        fvars = eqtns.free_symbols
    else:
        fvars = set(flatten([eqtn.free_symbols for eqtn in flatten(eqtns)]))
    fvars = tuple(set(fvars))
    
    return fvars