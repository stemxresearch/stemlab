from typing import Any
from decimal import Decimal, ROUND_HALF_UP

from pandas import DataFrame, Series, set_option
from sympy import (
    FiniteSet, Matrix, Expr, flatten, sympify, MutableDenseMatrix, Integer, Float, denom
)
from numpy import asfarray, ndarray, round, asarray, array_equal

from stemlab.core.symbolic import is_symexpr
from stemlab.core.arraylike import is_iterable
from stemlab.core.datatypes import ArrayMatrixLike


def float15(x: float) -> float:
    """
    Return exact decimal points. It avoids round off errors, e.g. 
    0.1 being presented as 0.09999999999999987.
    """
    return round(x, 15)


def to_float_in_steps(
    result_name: str, 
    result: Expr | ArrayMatrixLike, 
    decimal_points: int
) -> dict:
    """
    Convert the results to float.

    Parameters
    ----------
    result_name : str
        Name of the result.
    result : Expr or ndarray or Matrix
        The result to be converted to float.
    decimal_points : int
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    dict
        A dictionary containing the original result and its float 
        version.

    Examples
    --------
    >>> import stemlab as stm
    >>> import sympy as sym
    >>> stm.to_float_in_steps(result_name='result', result=[1.234, 2.345, 3.456], decimal_points=2)
    {'result': [1.234, 2.345, 3.456], 'result_': [1.23, 2.35, 3.46]}
    >>> x, y, z = sym.symbols('x y z')
    >>> M = sym.Matrix([[1.234, x], [y, 3.456]])
    >>> stm.to_float_in_steps("M", M, 2) # sig. figures not decimals
    {'M': [[1.234, x], [y, 3.456]], 'M_': [[1.2, x], [y, 3.5]]}
    """
    if isinstance(result, Expr):
        try:
            result_float = needs_approximation(result, decimal_points)
        except Exception:
            result_float = None

        if result_float:
            result_dict = {
                result_name: result, 
                f'{result_name}_': result_float,
            }  # result and float
        else:
            result_dict = {result_name: result}  # result only
    else:
        try:
            # if this fails then it is not a list/tuple/matrix/array
            len(result)
        except Exception:
            result = Matrix([result])

        # convert to float where applicable
        result = asarray(result) # just incase it is not an array
        float_answer, decimal_points = float_decimals(decimal_points)
        if is_float_display(result, float_answer):
            result_float = fround(result, decimal_points)
        else:
            result_float = None

        if len(flatten(result)) == 1:
            result = result.tolist()[0][0]
            if result_float:
                result_float = result_float.tolist()[0][0]
        else:
            result = result.tolist()
            if result_float:
                result_float = result_float.tolist()

        if result_float:
            result_dict = {
                result_name: result, 
                f"{result_name}_": result_float
            }
        else:
            result_dict = {result_name: result}

    return result_dict


def needs_approximation(result: ArrayMatrixLike, decimal_points: int) -> bool:
    """
    Checks whether approximation float result is equal to original 
    results in which case there is no need for approximation.

    Parameters
    ----------
    result : {array_like, Matrix}
        The original result to be checked.
    decimal_points : int
        The number of decimal points for approximation.

    Returns
    -------
    bool
        True if approximation is needed, False otherwise.
    """
    try:
        if decimal_points == -1:
            return False
        else:
            decimal_points = min(14, decimal_points)
            result_float = result.evalf(decimal_points)
        # Check if the result is equal to its float approximation
        if isinstance(result, ndarray):
            if array_equal(result, result_float):
                return False
        elif isinstance(result, Matrix):
            if result.equals(result_float):
                return False
        elif result == result_float:
            return False
        return True
    except (TypeError, ValueError):
        return False


def float_decimals(decimal_points: int) -> tuple[bool, int]:
    """
    Check whether or not a result should be converted to float.

    Parameters
    ----------
    - decimal_points (int): The number of decimal points specified. 
    If it's -1, the result should not be converted to float.

    Returns
    -------
    tuple[bool, int]
        A tuple containing a boolean indicating whether the result 
        should be converted to float (True) or not (False), and the 
        original number of decimal points.
    """
    float_answer = False if decimal_points == -1 else True

    return float_answer, decimal_points


def float_dict(result: dict, decimal_points: int = -1) -> dict:
    """
    Convert numeric elements of a dictionary to floats, with 
    optional rounding.

    Parameters
    ----------
    result : dict
        The dictionary containing values to be converted to floats.
    decimal_points : int, optional (default=-1)
        Number of decimal points for rounding. If -1 (default), 
        no rounding will be performed.

    Returns
    -------
    result : dict
        A dictionary with numeric elements rounded to the specified 
        number of decimals.

    Examples
    --------
    >>> stm.float_dict({'a': 1.234, 'b': 2.345}, decimal_points=2)
    {'a': 1.23, 'b': 2.34}
    >>> stm.float_dict({'a': 'text', 'b': 2 * 3.14}, decimal_points=2)
    {'a': 'text', 'b': 6.28}
    >>> stm.float_dict({'value1': 'text', 'value2': 11/13}, decimal_points=4)
    {'value1': 'text', 'value2': 0.8462}
    """
    def convert_value(value, decimal_points):
        try:
            # Try to convert to float and round if needed
            if isinstance(value, (int, float)):
                return round(float(value), decimal_points) if decimal_points != -1 else float(value)
            # Check if the value can be evaluated as a numeric expression
            elif isinstance(value, Expr):
                return value.evalf(decimal_points) if decimal_points != -1 else value.evalf()
            return value
        except (ValueError, TypeError):
            return value

    return {key: convert_value(value, decimal_points) for key, value in result.items()}


def round_half_up(x: float, decimal_points: int = -1):
    """
    Rounds a floating-point number to a specified number of decimal places 
    using the ROUND_HALF_UP method from the Decimal module.

    Parameters
    ----------
    x : float
        The number to be rounded.
    decimal_points : int, optional (default=-1)
        The number of decimal places to round to.

    Returns
    -------
    number_float : float
        The rounded number as a float.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.round_half_up(0.045, decimal_points=2)
    0.05

    >>> stm.round_half_up(1.234567, decimal_points=3)
    1.235

    >>> stm.round_half_up(5.678, decimal_points=0)
    6.0

    >>> stm.round_half_up(3.14159, decimal_points=-1)
    3.14159
    """
    if decimal_points == -1:
        return x
    number = Decimal(str(x))
    zeros_str = '0' * (decimal_points - 1)
    number_float = number.quantize(
        Decimal(f'0.{zeros_str}1'), rounding=ROUND_HALF_UP
    )
    return float(number_float)


def dframe_round(dframe, decimal_points):
    """
    Applies different rounding to columns in a DataFrame based 
    on their data types.
    
    Parameters
    ----------
    dframe: pandas.DataFrame 
        The DataFrame to with values to be rounded off.
    decimal_points: int
        Number of decimal points to use for truncation and rounding.
        
    Notes
    -----
    - For columns containing strings, rounds numeric values in the 
    string to the specified decimal points.
    - For numeric columns, rounds the values to the specified decimal points.
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with values rounded off.
    """
    df = dframe.copy() # this is important to avoid overwriting original dframe
    def round_number(x, decimal_points):
        """
        Truncate objects which might contain both numbers and strings.
        """
        try:
            # Convert to float, truncate, and convert back to string with 
            # fixed decimal places
            rounded = round(float(x) * 10 ** decimal_points) / 10 ** decimal_points
            return f'{rounded:.{decimal_points}f}'
        except (ValueError, TypeError):
            return x

    # Identify string and numeric columns
    string_columns = df.select_dtypes(include=['object']).columns
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Apply truncation to string columns
    df[string_columns] = df[string_columns].map(
        lambda x: round_number(x, decimal_points=decimal_points)
    )

    # Round numeric columns
    df[numeric_columns] = df[numeric_columns].round(decimal_points)

    return df

   
def fround(
    x: int | float | list | Expr,
    decimal_points: int = -1,
    to_Matrix=True
) -> int | float | list | Expr:
    """
    Round the elements of an array or a sympy expression to the 
    specified number of decimal points.

    Parameters
    ----------
    x : {float, array_like, sym.Expr}
        The array or expression to round.
    decimal_points : int, optional (default=-1)
        Number of decimal points (or significant figures) to round the 
        results to. Default -1 means no rounding.
    to_Matrix : bool, optional (default = True)
        If `True`, result will be converted to a sympy matrix.

    Returns
    -------
    result : {int, float, list, Expr}
        The value of `x` rounded to `decimal_points`.
        
    Notes
    -----
    The result will be converted to a sympy object where applicable. 
    That is, sympy.Matrix, sympy.Expr, sympy.Int, sympy.Float, etc

    Examples
    --------
    >>> import sympy as sym
    >>> import pandas as pd
    >>> import stemlab as stm
    >>> stm.fround(x=[1.234, 2.345, 3.456], decimal_points=2)
    Matrix([[1.23000000000000], [2.34000000000000], [3.46000000000000]])
    >>> x, y = sym.symbols('x y')
    >>> matrix_result = sym.Matrix([[1.234, x], [y, 3.456]])
    >>> stm.fround(x=matrix_result, decimal_points=3) # sig. figures
    Matrix([[1.23, x], [y, 3.46]])
    >>> df_result = pd.DataFrame({'A': [1.234, 2.345], 'B': [3.456, 4.567]})
    >>> stm.fround(x=df_result, decimal_points=2)
          A     B
    0  1.23  3.46
    1  2.35  4.57
    """

    def get_numeric(x: Any) -> float:
        """
        Returns float if value is an array or list of a single 
        element.
        
        Parameters
        ----------
        x : Any
            The result to be rounded off.
        
        Returns
        -------
        x : float
            The numeric value as a float if the result has only one 
            element.
        """
        try:
            if hasattr(x, 'shape') and (x.shape == (1, 1) or len(x) == 1):
                x = x[0]
        except:
            pass
        
        return x
    
    if isinstance(x, str):
        try:
            x = sympify(x)
        except Exception:
            pass
    
    if not isinstance(decimal_points, int):
        decimal_points = -1
    
    decimal_points = 14 if decimal_points == -1 else decimal_points
    
    set_option('display.precision', decimal_points)
    
    if isinstance(x, (DataFrame, Series)):
        return dframe_round(dframe=x, decimal_points=decimal_points)

    if is_symexpr(x):
        try:
            x = x.evalf(decimal_points)
        except Exception:
            pass
    
    if isinstance(x, (set, FiniteSet)):
        x = list(x)
    
    if is_iterable(x) and not isinstance(x, dict):
        try:
            x = Matrix(x)
            try:
                N = asfarray(x.evalf(decimal_points + 10))
                x = asfarray(Matrix(round(N, decimal_points)))
            except Exception: # symbolic matrix
                try:
                    x = x.evalf(decimal_points)
                except Exception:
                    pass
        except Exception as e:
            raise e

    if isinstance(x, dict) or (isinstance(x, str) and ":" in x):
        try:
            x = float_dict(x, decimal_points=decimal_points)
        except Exception:
            pass
        

    if isinstance(x, (int, float, Integer, Float)):
        try:
            x = round_half_up(float(x), decimal_points)
        except Exception:
            pass

    if to_Matrix:
        try:
            x = Matrix(x)
        except Exception:
            pass

    x = get_numeric(x)
    
    return x


def is_float_display(
    results: ArrayMatrixLike, decimal_points: int
) -> bool:
    """
    Check if the results are float-displayable based on the specified 
    number of decimal points.

    Parameters
    ----------
    results : ArrayMatrixLike
        The results to be checked.
    decimal_points : int
        Number of decimal points to consider.

    Returns
    -------
    bool
        True if the results are float-displayable, False otherwise.
    """

    float_answer = False if decimal_points == -1 else True

    return is_any_rational(results) and float_answer


def is_any_rational(M: ArrayMatrixLike) -> bool:
    """
    Checks if an array contains at least one non-integer value.

    Parameters
    ----------
    M : ArrayMatrixLike
        The array to check.

    Returns
    -------
    bool
        True if the array contains at least one non-integer value, 
        False otherwise.

    Examples
    --------
    >>> import stemlab as stm
    >>> import sympy as sym
    >>> import numpy as np

    # Example 1: Array containing non-integer values
    >>> array1 = [1.5, 2, 3]
    >>> stm.is_any_rational(array1)
    True

    # Example 2: Array containing only integer values
    >>> array2 = [1, 2, 3]
    >>> stm.is_any_rational(array2)
    False

    # Example 3: Matrix containing non-integer values
    >>> matrix1 = sym.Matrix([[1.5, 2], [3, 4]])
    >>> stm.is_any_rational(matrix1)
    True

    # Example 4: Matrix containing only integer values
    >>> matrix2 = sym.Matrix([[1, 2], [3, 4]])
    >>> stm.is_any_rational(matrix2)
    False

    # Example 5: Matrix containing symbolic expressions
    >>> x, y = sym.symbols('x y')
    >>> matrix3 = sym.Matrix([[x, y], [3, 4]])
    >>> stm.is_any_rational(matrix3)
    False

    # Example 6: Single integer value
    >>> single_value = 5
    >>> stm.is_any_rational(single_value)
    False

    # Example 7: Single float value
    >>> single_float_value = 3.14
    >>> stm.is_any_rational(single_float_value)
    True
    """
    def _is_rational(value):
        try:
            return float(sympify(value)) % 1 != 0
        except (ValueError, TypeError):
            return denom(value) != 1
    
    if not isinstance(M, MutableDenseMatrix):
        try:
            M = Matrix(M)
        except Exception: # it could be a single value
            M = Matrix([M])

    return any(_is_rational(value) for value in flatten(M))

