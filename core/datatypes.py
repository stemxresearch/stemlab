from typing import Sequence, TypeVar

from sympy import (
    sympify, MutableDenseMatrix, latex, Abs, Expr, flatten
)
from numpy import array, any, issubdtype, ndarray, number
from pandas import DataFrame


T = TypeVar('T')


class ArrayMatrixLike(Sequence[T]):
    pass


class NumpyArraySympyMatrix(Sequence[T]):
    pass


class NumpyArray(Sequence[T]):
    pass


class ListArrayLike(Sequence[T]):
    pass


def abs_value(value: int | float) -> str:
    """
    Returns the absolute value of an object.

    Parameters
    ----------
    value : {int, float}
        The numerical value for which the absolute value needs to be 
        determined.

    Returns
    -------
    value_cleaned : str
        A string representing the absolute value of the input value in 
        LaTeX format.

    Raises
    ------
    Exception
        If an error occurs during the calculation or conversion to 
        LaTeX.

    Examples
    --------
    >>> stm.abs_value(5)
    '5'
    >>> stm.abs_value(-3.14)
    '3.14'
    >>> stm.abs_value(0)
    '0'
    """
    try:
        value = latex(Abs(value))
        # just in case it is a symbolic expression
        value = value.replace('\\left|{', '').replace('}\\right|', '')
    except Exception as e:
        raise Exception(f'abs_value(): {e}')
    
    return value


def is_function(obj):
    """

    Check if the given object is a function.

    Parameters
    ----------
    obj: object
        The object to be checked.

    Returns
    -------
    bool: 
        True if the object is a function, False otherwise.

    """
    import numpy as np
    try:
        if callable(obj) and (
            isinstance(obj, type(lambda x: x)) or hasattr(np, obj.__name__)):
            return True
        else:
            return False
    except Exception as e:
        raise TypeError(e)


def is_any_element_negative(M: list[int | float]) -> bool:
    """
    Check if any of the elements in the array are negative, including 
    symbolic expressions or string.

    Parameters
    ----------
    array_ : array_like
        The array-like object to be checked.

    Returns
    -------
    bool
        Returns True if any element in the array is negative, 
        otherwise returns False.

    Raises
    ------
    Exception
        If the input array cannot be converted to a numpy array.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.is_any_element_negative([1, 2, 3])
    False
    >>> stm.is_any_element_negative([-1, 2, 3])
    True
    >>> stm.is_any_element_negative([['b', 3, 8], [3, 1, 9]])
    False
    >>> stm.is_any_element_negative([[4, 'a', 9], ['-g', 8, 5]])
    True
    """
    try:
        M = array(M)
    except Exception as e:
        raise Exception(f'is_any_element_negative(): {e}')
    if issubdtype(M.dtype, number): # all values are numeric
        return any(M < 0)
    else: # contains symbols
        return any([is_negative(value) for value in flatten(M)])


def is_negative(value: int| float | str | Expr) -> bool:
    """
    Checks whether a numerical or symbolic value is negative without 
    using '<' operator.

    Parameters
    ----------
    value : {int, float, str, Expr}
        The value to be checked for negativity.

    Returns
    -------
    bool
        True if the value is negative, otherwise False.

    Notes
    -----
    This function avoids using the '<' operator to determine 
    negativity. For symbolic or string expressions, the function
    checks if the expression starts with a minus, and not necessarily 
    negative.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.is_negative(-5)
    True
    >>> stm.is_negative(0)
    False
    >>> stm.is_negative(5)
    False
    >>> stm.is_negative('-3')
    True
    >>> stm.is_negative(-3 * x ** 2)
    True
    """
    return str(value).startswith('-')


def is_numeric(strng: str) -> bool:
    """
    Check if a string represents a numeric value.

    Parameters
    ----------
    strng : str
        The string to be checked.

    Returns
    -------
    bool
        True if the string represents a numeric value, otherwise False.
    """
    try:
        float(sympify(strng))
        return True
    except (ValueError, TypeError):
        return False
    

def is_nested_list(lst: list) -> bool:
    """
    Check whether or not a list is nested.

    Parameters
    ----------
    lst : list_like
        The list-like object to be checked.

    Returns
    -------
    bool
        True if the list is nested, otherwise False.
    """
    if not isinstance(lst, (list, tuple)):
        try:
            lst = lst.tolist() # from numpy / sympy
        except AttributeError:
            try:
                lst = lst.values.tolist()  # pandas DataFrame
            except AttributeError:
                return False
    try:
        nested = list(
            filter(lambda x: isinstance(x, (list, tuple, set)), lst)
        )
        return len(nested) > 0
    except Exception:
        return False
    

def conv_structure_to_list(data_structure: ArrayMatrixLike) -> tuple[list]:
    """
    Convert `data_structure` to a list, and if it is a DataFrame extract 
    the index (row names) and column names.

    Parameters
    ----------
    data_structure : {array_like, Matrix}
        The data structure to be converted to a list.

    Returns
    -------
    tuple
        A tuple containing:
        - variable_values: The converted data structure as a string 
        representation.
        - row_names: List of row names if `data_structure` is a 
        DataFrame, otherwise None.
        - column_names: List of column names if `data_structure` is a 
        DataFrame, otherwise None.
        
    Examples
    --------
    >>> stm.import numpy as np
    >>> import pandas as pd
    >>> import sympy as sym
    >>> import stemlab as stm
    
    >>> stm.conv_structure_to_list([1, 2, 3])
    {'lst': '[1, 2, 3]', 'row_names': None, 'col_names': None}
    
    >>> A = np.array([[1, 2], [3, 4]])
    >>> stm.conv_structure_to_list(A)
    {'lst': '[[1, 2], [3, 4]]', 'row_names': None, 'col_names': None}
    
    >>> B = sym.Matrix([[11, 21], [31, 41]])
    >>> stm.conv_structure_to_list(B)
    {'lst': '[[11, 21], [31, 41]]', 'row_names': None, 'col_names': None}
    
    >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, index=['x', 'y'])
    >>> stm.conv_structure_to_list(df)
    {'lst': '[[1, 3], [2, 4]]', 'row_names': ['x', 'y'], 'col_names': ['a', 'b']}
    """
    row_names, column_names = [None] * 2
    # convert the user input to a list
    if isinstance(data_structure, (list, tuple)):
        # list just incase it is a tuple
        variable_values = str(list(data_structure))
    elif isinstance(data_structure, (MutableDenseMatrix, ndarray)):
        # extract the values of the sympy matrix - avoid use of list() 
        # since it will always return a one dimensional list
        variable_values = str(data_structure.tolist())
    elif isinstance(data_structure, DataFrame):
        variable_values = str(data_structure.values.tolist())
        row_names = list(data_structure.index)
        column_names = list(data_structure.columns)
    else:
        variable_values = str(data_structure)

    result_dict = {
        'lst': variable_values,
        'row_names':row_names,
        'col_names': column_names
    }
    
    return result_dict
