import warnings

from sympy import lambdify, solve, sympify
from scipy.optimize import fsolve

from stemlab.core.base.dictionaries import dict_none_remove


def filter_real_numbers(lst: list, is_positive: bool = True):
    """
    Filters a list of values to return only real numbers.
    Optionally filters for positive real numbers if specified.

    Parameters
    ----------
    lst : list
        The list of numbers to filter.
    is_positive : bool, optional (default=True)
        If `True`, filter for positive real numbers.

    Returns
    -------
    filtered : list
        A list of filtered real numbers.

    Examples
    --------
    >>> import stemlab as stm
    >>> numbers = [-8.045, 0.045, -7.847 - 1.25j, -7.847 + 1.25j, 3.5,
    ... 0, 2.1, -2.3]
    >>> stm.filter_real_numbers(lst=numbers)
    [0.045, 3.5, 2.1]

    >>> stm.filter_real_numbers(lst=numbers, is_positive=False)
    [-8.045, 0.045, 3.5, 0, 2.1, -2.3]
    """
    condition = lambda x: x > 0 if is_positive else True
    filtered = [
        x for x in lst if isinstance(x, (int, float)) and condition(x)
    ]
    
    return filtered


def make_subject_and_solve(dct: dict, fexpr: str, initial_guess: float = 1.0):
    
    args_dict, expr_var = dict_none_remove(dct=dct, return_keys=True)
    fexpr = sympify(fexpr)
    expr_var = sympify(expr_var)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = solve(fexpr.subs(args_dict), expr_var)
        if len(result) > 1:
            result = filter_real_numbers(lst=result, is_positive=True)
    if w or not result:
        f = lambdify(sympify(expr_var), fexpr.subs(args_dict), 'numpy')
        result, = fsolve(func=f, x0=initial_guess)

    if isinstance(result, (tuple, list)):
        result = result[0]
    
    return result