from itertools import chain
import re
from typing import Any, Literal

from numpy import (
    hstack, ndarray, repeat, nan, array, asarray, float64, diff, arange, zeros,
    round, full, isnan, append
)
from pandas import DataFrame, Series
from sympy import Matrix, Rational, flatten, sympify, latex

from stemlab.core.validators.errors import (
    NumpifyError, IterableError, VectorLengthError
)
from stemlab.core.datatypes import ArrayMatrixLike, NumpyArray

VALID_OPERATIONS = ['+', '-', '\\times', '\\div', '^']


def general_array(
    element: str ='a', 
    nrows: int | None = None, 
    ncols: int | None = None, 
    rows: str = "m", 
    cols: str = "n", 
    operation: Literal['+', '-', '\\times', '\\div', '^'] | None = None, 
    constant: str = 'c'
) -> str:
    """
    Generate a general m x n array.

    Parameters
    ----------
    element : str, optional (default='a')
        The base element to fill the array with.
    nrows : int, optional (default=None)
        Number of rows in the array. If not specified, generates 
        a custom array based on `rows` and `cols`.
    ncols : int, optional (default='nrows')
        Number of columns in the array.
    rows : str, optional (default='m')
        Row variable symbol.
    cols : str, optional (default='n')
        Column variable symbol.
    operation : str, {'+', '-', '\\times', '\\div', '^'} optional (default='+')
        Math operation to perform on array elements.
    constant : str, optional (default='c')
        Constant value to apply during the operation.

    Returns
    -------
    result : str
        LaTeX representation of the generated array.

    Examples
    --------
    >>> general_array(element='x', nrows=3, ncols=3)
    '\\left[\\begin{array}{ccc} x_{11} & x_{12} & x_{13} \\\\ x_{21} & x_{22} & x_{23} \\\\ x_{31} & x_{32} & x_{33} \\end{array}\\right]'

    >>> general_array(element='y', nrows=2, ncols=2, operation='+',
    ... constant='k')
    '\\left[\\begin{array}{cc} y_{11}+k & y_{12}+k \\\\ y_{21}+k & y_{22}+k \\end{array}\\right]'

    >>> general_array(element='z', nrows=2, operation='^', constant='2')
    '\\left[\\begin{array}{cc} z_{11}^2 & z_{12}^2 \\\\ z_{21}^2 & z_{22}^2 \\end{array}\\right]'

    >>> general_array(element='b', nrows=2, ncols=2, operation='\\div',
    ... constant='2')
    '\\left[\\begin{array}{cc} b_{11}\\div2 & b_{12}\\div2 \\\\ b_{21}\\div2 & b_{22}\\div2 \\end{array}\\right]'

    >>> general_array(element='c', rows='r', cols='c', nrows=3, ncols=2)
    '\\left[\\begin{array}{cc} c_{11} & c_{12} \\\\ c_{21} & c_{22} \\\\ c_{31} & c_{32} \\end{array}\\right]'
    """
    from stemlab.core.htmlatex import color_values
    from stemlab.core.htmlatex import tex_to_latex
    from stemlab.core.validators.validate import ValidateArgs
    from sympy import zeros # should be here

    element = ValidateArgs.check_string(par_name='element', user_input=element)
    if rows is not None:
        nrows = ValidateArgs.check_numeric(par_name='nrows', limits=[1, 100], is_integer=True)
    if rows is not None:
        ncols = ValidateArgs.check_numeric(par_name='ncols', limits=[1, 100], is_integer=True)
    rows = ValidateArgs.check_string(par_name='rows', user_input=rows)
    cols = ValidateArgs.check_string(par_name='cols', user_input=cols)
    if operation is not None:
        operation = ValidateArgs.check_member(
            par_name='operation', 
            valid_items=VALID_OPERATIONS, 
            user_input=operation, 
            default='+'
        )
    constant = ValidateArgs.check_string(par_name='cols', user_input=constant)
    A = element
    if nrows is None:
        result = f"""
            \\left[
                \\begin{{array}}{{cccc}} 
                    {A}_{{11}} & {A}_{{12}} & \\cdots & {A}_{{1{cols}}} ~~ \\\\ 
                    {A}_{{21}} & {A}_{{22}} & \\cdots & {A}_{{2{cols}}} ~~ \\\\ 
                    \\vdots & \\vdots & \\ddots & \\vdots \\\\ 
                    {A}_{{{rows}1}} & {A}_{{{rows}2}} & \\cdots & {A}_{{{rows}{cols}}} ~~ 
                \\end{{array}}
            \\right]
        """
        if operation is not None:
            if operation not in ['+', '-', '\\times', '\\div', '^']:
                raise Exception(f'{general_array()} is an invalid operation.')
            constant = color_values(constant)
            operation = operation.replace('^', '\\,^')
            if operation == '\\times':
                result = result.replace(f'{A}_', f'{constant}\\,.{A}_')
            else:
                result = result.replace(' &', f'{operation}{constant} &')
                result = result.replace(f'dots{operation}{constant} &', f'dots &')
                result = result.replace('~~', f'{operation}{constant}')
    else:
        ncols = nrows if ncols is None else ncols
        M = zeros(nrows, ncols)
        for row in range(nrows):
            for col in range(ncols):
                M[row, col] = f'{element}{row + 1}{col + 1}'
        result = tex_to_latex(M)

    return result


def general_array_ab(
    A: str = 'a', 
    B: str = 'b', 
    rows: str = "m", 
    cols: str = "n", 
    operation: str = '+', 
    color: bool = True
) -> str:
    """
    Perform an arithmetic operation on two m x n matrices and color 
    elements of the second matrix.

    Generate a general m x n array.

    Parameters
    ----------
    element : str, optional (default='a')
        The base element to fill the array with.
    nrows : int, optional (default=None)
        Number of rows in the array. If not specified, generates 
        a custom array based on `rows` and `cols`.
    ncols : int, optional (default='nrows')
        Number of columns in the array.
    rows : str, optional (default='m')
        Row variable symbol.
    cols : str, optional (default='n')
        Column variable symbol.
    operation : str, {'+', '-', '\\times', '\\div', '^'} optional (default='+')
        Math operation to perform on array elements.
    color : str, optional (default=True)
        If `True`, coloring will be applied.

    Returns
    -------
    result : str
        LaTeX representation of the generated array.

    Examples
    --------
    >>> general_array_ab(rows=2, cols=2, operation='\\times', color=False)
    '\\left[\\begin{array}{cc} a_{11}\\times b_{11} & a_{12}\\times b_{12} \\\\ a_{21}\\times b_{21} & a_{22}\\times b_{22} \\end{array}\\right]'

    >>> general_array_ab(A='x', B='y', rows=2, cols=2, operation='-', color=True)
    '\\left[\\begin{array}{cc} \\color{red}x_{11}-y_{11} & \\color{red}x_{12}-y_{12} \\\\ \\color{red}x_{21}-y_{21} & \\color{red}x_{22}-y_{22} \\end{array}\\right]'

    >>> general_array_ab(rows=3, cols=3, operation='\\div', color=False)
    '\\left[\\begin{array}{ccc} \\frac{a_{11}}{b_{11}} & \\frac{a_{12}}{b_{12}} & \\frac{a_{13}}{b_{13}} \\\\ \\frac{a_{21}}{b_{21}} & \\frac{a_{22}}{b_{22}} & \\frac{a_{23}}{b_{23}} \\\\ \\frac{a_{31}}{b_{31}} & \\frac{a_{32}}{b_{32}} & \\frac{a_{33}}{b_{33}} \\end{array}\\right]'

    >>> general_array_ab(A='M', B='N', rows=2, cols=2, operation='+', color=True)
    '\\left[\\begin{array}{cc} \\color{blue}M_{11}+N_{11} & \\color{blue}M_{12}+N_{12} \\\\ \\color{blue}M_{21}+N_{21} & \\color{blue}M_{22}+N_{22} \\end{array}\\right]'
    """
    from stemlab.core.validators.validate import ValidateArgs

    result = f"""
        \\left[
            \\begin{{array}}{{cccc}} 
                {A}_{{11}} {operation} {B}_{{11}} & {A}_{{12}} {operation} {B}_{{12}} & \\cdots & {A}_{{1{cols}}} {operation} {B}_{{1{cols}}} \\\\ 
                {A}_{{21}} {operation} {B}_{{21}} & {A}_{{22}} {operation} {B}_{{22}} & \\cdots & {A}_{{2{cols}}} {operation} {B}_{{2{cols}}} \\\\ 
                \\vdots & \\vdots & \\ddots & \\vdots \\\\ 
                {A}_{{{rows}1}} {operation} {B}_{{{rows}1}} & {A}_{{{rows}2}} {operation} {B}_{{{rows}2}} & \\cdots & {A}_{{{rows}{cols}}} {operation} {B}_{{{rows}{cols}}} 
            \\end{{array}}
        \\right]
        """
    from stemlab.core.validators.validate import ValidateArgs

    A = ValidateArgs.check_string(par_name='A', user_input=A)
    B = ValidateArgs.check_string(par_name='B', user_input=B)
    rows = ValidateArgs.check_string(par_name='rows', user_input=rows)
    cols = ValidateArgs.check_string(par_name='cols', user_input=cols)
    if rows is not None:
        operation = ValidateArgs.check_member(
            par_name='operation', 
            valid_items=VALID_OPERATIONS, 
            user_input=operation, 
            default='+'
        )
    constant = ValidateArgs.check_string(par_name='cols', user_input=constant)
    if color:
        pattern = re.compile(f'{re.escape(B)}_{{([a-zA-Z]+|\\d+)([a-zA-Z]+|\\d+)}}')
        result = re.sub(pattern, color_values_matched, result)

    return result


def color_values_matched(match: re.Match) -> str:
    """
    Color matched values in a string.

    Parameters
    ----------
    match : re.Match
        Match object containing the matched value.

    Returns
    -------
    str
        HTML representation of the colored value.
    """
    from stemlab.core.htmlatex import color_values

    return color_values(match.group())


def initialize_array(
    nrows: int, ncols: int, fill_value: str = '', dtype: str = 'U25000'
) -> ndarray:
    """
    Initialize a 2D array with the specified string fill value.

    Parameters
    ----------
    nrows : int
        Number of rows in the array.
    ncols : int
        Number of columns in the array.
    fill_value : str, optional
        Value to fill the array with. Defaults to an empty string.
    dtype : str, optional (default='U25000')
        Data type of the array. Defaults to "U25000", which supports 
        Unicode strings.

    Returns
    -------
    result : numpy.ndarray
        Initialized array with the specified dimensions and fill value.
    """
    result = full((nrows, ncols), fill_value, dtype=dtype)
    return result



def quote_rational(array_values: list) -> list:
    """
    Convert fractions to strings, enclosing them in quotation marks.

    Parameters
    ----------
    array_values : list
        The input array-like object containing fractions.

    Returns
    -------
    array_values : list
        The modified array with fractions converted to strings.

    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm
    >>> array_values = [['1/2', 4], ['5/6', 8]]
    >>> stm.quote_rational(array_values)
    [['1/2', '4'], ['5/6', '8']]
    """
    from stemlab.core.symbolic import sym_sympify
    array_values = sym_sympify(expr_array=array_values, is_expr=False)
    if not isinstance(array_values, (list, tuple)):
        try:
            array_values = array_values.tolist()
        except Exception as e:
            raise Exception(e)
    nrow, ncol = Matrix(array_values).shape
    for i in range(nrow):
        for j in range(ncol):
            if isinstance(array_values[i][j], Rational):  
                array_values[i][j] = str(array_values[i][j])

    return array_values


def arr_abrange(
    a: int | float, 
    b: int | float | None = None, 
    h: int | float = 1, 
    dtype=None
) -> NumpyArray[float64]:
    """
    Return evenly spaced values within the closed interval [a, b].  
    Unlike the `np.arange()` this function includes the last value `b`.

    Parameters
    ----------
    a : int
        Start of the sequence.
    b : int, optional (default=None)
        End of the sequence.
    h : int, optional (default=1)
        Stepsize or interval.
    dtype : dtype, optional (default=None)
        The type of the output array.  If `dtype` is not given, infer 
        the data type from the other input arguments.

    Returns
    -------
    result : NDarray
        A Numpy array of evenly spaced values between `a` and `b` 
        inclusive.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.arr_abrange(a=10)
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
    >>> stm.arr_abrange(a=10, b=20)
    array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    >>> stm.arr_abrange(a=10, b=50, h=3)
    array([10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 50])
    >>> stm.arr_abrange(a=50, b=20, h=-2)
    array([50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20])
    """
    if b is None:
        # if `b` is not given, then let `a = 0` and `b = a` so that it 
        # begins from `0` to `a`
        a, b = (0, a)
    result = arange(a, b, h, dtype)
    if result.size != 0:
        result = append(result, b) if result[-1] != b else result

    return result


def is_strictly_increasing(user_input: list, par_name: str='user_input') -> bool:
    """
    Checks if a sequence is strictly increasing.

    Parameters
    ----------
    user_input : Array-like
        A sequence of comparable elements 
    par_name : str
        Name to use in error messages to describe the parameter being 
        checked.
        
    Returns
    -------
    is_strict_increasing : bool
        `True` if the sequence is strictly increasing, 
        `False` otherwise.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.is_strictly_increasing([1, 2, 3, 4, 5])
    True

    >>> stm.is_strictly_increasing([1, 3, 2, 4, 5])
    False

    >>> stm.is_strictly_increasing([1])
    True

    >>> stm.is_strictly_increasing([])
    True

    >>> stm.is_strictly_increasing([1, 1, 2, 3])
    False
    """
    if not is_iterable:
        raise IterableError(par_name=par_name, user_input=user_input)
    
    x = conv_to_arraylike(
        array_values=user_input, n=2, label='at least', par_name=par_name
    )
    is_strict_increasing = all(diff(x) > 0)
    
    return is_strict_increasing


def is_diff_constant(
    user_input: Any = 'user_input',
    par_name: str = 'argument',
    decimal_points: int = 14
) -> bool:
    """
    Check if the difference between consecutive elements of a list is 
    constant.

    Parameters
    ----------
    user_input : list-like
        A list of numbers.
    par_name : str
        Name to use in error messages to describe the parameter being 
        checked.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    bool
        True if the difference between consecutive elements is 
        constant, False otherwise.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.is_diff_constant([1, 3, 5, 7, 9])
    True
    >>> stm.is_diff_constant([1, 3, 5, 8, 9])
    False
    >>> stm.is_diff_constant([0.1, 0.2, 0.3, 0.4, 0.5])
    True
    >>> stm.is_diff_constant([0.1, 0.3, 0.6, 1.0])
    False
    """
    if not is_iterable:
        raise IterableError(par_name=par_name, user_input=user_input)
    
    x = conv_to_arraylike(array_values=user_input, par_name=par_name)
    
    return all(round(diff(x, 2), decimal_points) == 0)


def list_join(
    lst: list, 
    delimiter: str = ", ",
    quoted: bool = True,
    use_and: bool = True, 
    html_tags: bool = True
) -> str:
    """
    Join elements of a list into a single string.

    Parameters
    ----------
    lst : list_like
        The list_like object to join.
    delimiter : str, optional (default=', ')
        The string to separate the elements in the joined string.
    use_and : bool, optional (default=True)
        If `True`, the preposition `and` will be included before the 
        last element.
    quoted : bool, optional (default=True)
        If `True` elements will be included within single quotation 
        marks.
    html_tags : bool, optional (default=True)
        If `True` elements will be styled (colored) using HTML.

    Returns
    -------
    lst : str
        The joined string.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.list_join(['apple', 'banana', 'orange'], use_and=True, html_tags=False)
    "'apple', 'banana' and 'orange'"
    >>> stm.list_join(['apple', 'banana', 'orange'], delimiter=' | ', use_and=False, quoted=False, html_tags=True)
    '<span style="color:#aaa;">apple | banana | orange</span>'
    """
    from stemlab.core.base.strings import str_replace_dot_zero
    from stemlab.core.validators.validate import ValidateArgs

    lst = conv_to_arraylike(array_values=lst, par_name='lst')
    quoted = ValidateArgs.check_boolean(user_input=quoted, default=True)
    quoted = "'" if quoted else ""
    delimiter = f'{quoted}{delimiter}{quoted}'
    html_tags = ValidateArgs.check_boolean(user_input=html_tags, default=True)
    if html_tags:
        html_left_aaa, html_left_000, html_right = (
            '<span style="color:#aaa;">', '<span style="color:#000;">', '</span>'
        )
    else:
        html_left_aaa, html_left_000, html_right = [''] * 3
    use_and = ValidateArgs.check_boolean(user_input=use_and, default=True)
    delimiter = ", " if not isinstance(delimiter, str) else delimiter

    if len(lst) == 1:
        joined_list = lst[0]
    else:
        if use_and:
            joined_list = f"{quoted}{delimiter.join(map(str, lst[:-1]))}{quoted}{html_right} and {html_left_000}'{lst[-1]}{quoted}"
        else:
            joined_list = f"{quoted}{delimiter.join(map(str, lst))}{quoted}"

    joined_list = f"{html_left_aaa}{joined_list}{html_right}"
    joined_list = str_replace_dot_zero(joined_list)

    return joined_list


def join_absolute(
    elements: list[any], decimal_points: int = -1
) -> str:
    """
    Join elements of a list or tuple as absolute values in LaTeX 
    format.

    Parameters
    ----------
    elements : list_like
        The elements to join as absolute values.
    decimal_points : int, optional (default=-1)
        The number of decimal points to round the elements. 
        Defaults to -1, which indicates no rounding.

    Returns
    -------
    result_string : str
        The joined string with absolute values in LaTeX format.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.join_absolute([1.234, -2.345, 3.456], decimal_points=2)
    '\\left|1.23\\right|+\\left|-2.34\\right|+\\left|3.46\\right|'
    """
    from stemlab.core.htmlatex import tex_to_latex
    from stemlab.core.base.strings import str_replace_dot_zero
    from stemlab.core.decimals import fround

    elements = flatten(elements)
    # attend to convert to float
    try:
        elements = fround(elements, decimal_points)
    except Exception:
        pass

    result_string = [f'\\left|{tex_to_latex(item)}\\right|' for item in elements]
    result_string = '+'.join(result_string)
    result_string = str_replace_dot_zero(result_string)

    return result_string


def join_squared(
    elements: list[any], decimal_points: int = -1
) -> str:
    """
    Join elements of a list or tuple as absolute values in LaTeX 
    format.

    Parameters
    ----------
    elements : list_like
        The elements to join as absolute values.
    decimal_points : int, optional (default=-1)
        The number of decimal points to round the elements. 
        Defaults to -1, which indicates no rounding.

    Returns
    -------
    result_string : str
        The joined string with absolute values in LaTeX format.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.join_squared([1.234, -2.345, 3.456], decimal_points=2)
    '\\left(1.23\\right)^{2}+\\left(-2.34\\right)^{2}+\\left(3.46\\right)^{2}'
    """
    from stemlab.core.htmlatex import tex_to_latex
    from stemlab.core.base.strings import str_replace_dot_zero
    from stemlab.core.decimals import fround
    
    elements = flatten(elements)
    # attend to convert to float
    try:
        elements = fround(elements, decimal_points)
    except Exception:
        pass

    result_string = [f'\\left({tex_to_latex(item)}\\right)^{{2}}' for item in elements]
    result_string = '+'.join(map(str, result_string))
    result_string = str_replace_dot_zero(result_string)

    return result_string


def join_elements(
    elements: list[any], 
    step_two: bool = True, 
    last_only: bool = False, 
    decimal_points: int = -1
) -> str:

    """
    Join elements of a list into a formatted expression.

    Parameters
    ----------
    elements : list, of float or int
        The elements to join and format.
    step_two : bool, optional (default=True)
        Whether to create a second line in the formatted expression.
    last_only : bool, optional (default=True)
        Whether to show only the second line of the formatted 
        expression.
    decimal_points : int, optional (default=-1)
        The number of decimal points to round the elements. 
        Defaults to -1, which indicates no rounding.

    Returns
    -------
    result_string : str
        The formatted expression.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.join_elements([1.234, -2.345, 3.456], step_two=True, last_only=False, decimal_points=2)
    '1.23+(-2.35)+3.46\n&= 1.23-2.35+3.46'
    """
    from stemlab.core.htmlatex import tex_to_latex
    from stemlab.core.datatypes import is_any_element_negative
    from stemlab.core.base.strings import str_replace_dot_zero
    from stemlab.core.decimals import fround
    from stemlab.core.validators.validate import ValidateArgs

    # attend to convert to float
    try:
        elements = flatten(fround(elements, decimal_points))
    except Exception:
        pass
    elements = conv_to_arraylike(array_values=elements)
    step_two = ValidateArgs.check_boolean(user_input=step_two, default=True)
    last_only = ValidateArgs.check_boolean(user_input=last_only, default=False)
    decimal_points = ValidateArgs.check_decimals(x=decimal_points)
    try:
        negative_found = False
        if is_any_element_negative(elements):
            negative_found = True
        result_string = [
            f'{tex_to_latex(item)}' if item > 0 else 
            f'\\left({tex_to_latex(item)}\\right)' for item in elements
        ]
        result_string = '+'.join(map(str, result_string))
        step_two = True if last_only else False
        if step_two:
            if negative_found:
                result_string_2 = [
                    tex_to_latex(item) if item < 0 else 
                    f'+{tex_to_latex(item)}' for item in elements
                ]
                result_string_2 = ''.join(map(str, result_string_2))
                if result_string_2.startswith('+'):
                    result_string_2 = result_string_2[1:]
                if last_only:
                    result_string = f'{result_string_2}'
                else:
                    result_string += f'\\\\[5pt] &= {result_string_2}'
    except Exception:
        result_string = [tex_to_latex(element) for element in elements]
        result_string = '+'.join(map(str, result_string))
        return result_string
    result_string = str_replace_dot_zero(result_string)

    return result_string


def case_list(
    lst: list[str], 
    case_: Literal['lower', 'upper', 'title', 'capitalize'] = 'lower'
) -> list[str]:
    """
    Change the case of list items.

    Parameters
    ----------
    lst : list, or tuple of str
        The list of strings to change the case.
    case_ : str, {'lower', 'upper', 'title', 'capitalize'}, optional (default='lower')
        The case to convert the list items to. Defaults to 'lower'.

    Returns
    -------
    lst : str
        The list with items converted to the specified case.

    Examples
    --------
    >>> import stemlab as stm
    >>> case_list(['hello', 'world'], case_='upper')
    ['HELLO', 'WORLD']
    >>> case_list(('hello', 'world'), case_='title')
    ['Hello', 'World']
    """
    if case_ in ['lower', 'upper', 'title', 'capitalize']:
        try:
            if case_ == 'lower':
                lst = [lst.lower() for lst in lst]
            elif case_ == 'upper':
                lst = [lst.upper() for lst in lst]
            elif case_ == 'title':
                lst = [lst.title() for lst in lst]
            else: # capitalize
                lst = [lst.capitalize() for lst in lst]
        except Exception as e:
            raise Exception(e)
        
    return lst


def is_len_equal(
    x: list, 
    y: list, 
    par_name: list[str] = ['x', 'y']
) -> bool:
    """
    Check if `x` and `y` have the same number of elements.

    Parameters
    ----------
    x: list_like
        An iterable object.
    y: list_like
        An iterable object.
    par_name : list-like, optional (default=['x', 'y'])
        Name to use in error messages to describe the parameter being 
        checked.

    Raises
    ------
    IterableError
        If `x` or `y` is not iterable.

    Returns
    -------
    bool
        True if `x` and `y` have the same number of elements, False 
        otherwise.

    Examples
    --------
    >>> import stemlab as stm

    >>> x1 = [1, 2, 3]
    >>> y1 = [4, 5, 6]
    >>> stm.is_len_equal(x1, y1)
    True

    >>> x2 = (1, 2, 3)
    >>> y2 = (4, 5, 6, 7)
    >>> stm.is_len_equal(x2, y2)
    False
    """
    try:
        par_name_x, par_name_y = par_name
    except Exception:
        par_name = ['x', 'y']
    if not is_iterable(x):
        raise IterableError(par_name=par_name_x, user_input=x)
    if not is_iterable(y):
        raise IterableError(par_name=par_name_y, user_input=y)

    return len(flatten(x)) == len(flatten(y))


def is_iterable(array_like, includes_str: bool = False):
    """
    Check if an object is iterable or a SymPy matrix.

    Parameters
    ----------
    array_like : array_like 
        The object to check for iterability.
    includes_str : bool, optional (default=False)
        If `True`, allows strings to be considered iterable.

    Returns
    -------
    bool
        True if the object is iterable or a SymPy matrix, False 
        otherwise.

    Examples
    --------
    >>> import stemlab as stm

    >>> is_iterable([1, 2, 3])
    True

    >>> is_iterable((1, 2, 3))
    True

    >>> is_iterable({1, 2, 3})
    True

    >>> is_iterable(pd.Series([1, 2, 3]))
    True

    >>> is_iterable(pd.DataFrame({'A': [1, 2, 3]}))
    True

    >>> is_iterable(np.array([1, 2, 3]))
    True

    >>> is_iterable(Matrix([[1, 2], [3, 4]]))
    True

    >>> is_iterable("hello")
    True

    >>> is_iterable(123)
    False

    >>> is_iterable("string", includes_str=True)
    True

    >>> is_iterable("string", includes_str=False)
    False
    """
    from sympy.matrices import MatrixBase
    from collections.abc import Iterable

    if includes_str and isinstance(array_like, str):
        return True
    elif isinstance(array_like, MatrixBase):
        return True
    else:
        return isinstance(array_like, Iterable)
    

def conv_list_to_dict(
    keys_list: list | None = None,
    values_list: list = []
) -> dict:
    """
    Convert two specified lists to a dictionary.

    Parameters
    ----------
    keys_list : {None, list, tuple}, optional (default=None)
        The keys of the dictionary to be created.
    values_list : list, or tuple, optional (default=[])
        The values of the dictionary to be created.

    Returns
    -------
    dict_result : dict
        A dictionary with the values of the first list as keys and 
        values of the second list as dictionary values.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> stm.conv_list_to_dict(['a', 'b', 'c'], [1, 2, 3])
    {'a': 1, 'b': 2, 'c': 3}

    >>> stm.conv_list_to_dict(['x', 'y', 'z'], [10, 20, 30])
    {'x': 10, 'y': 20, 'z': 30}

    >>> stm.conv_list_to_dict(['a', 'b', 'c'], ['x', 'y', 'z'])
    {'a': 'x', 'b': 'y', 'c': 'z'}

    >>> stm.conv_list_to_dict([], [])
    {}

    >>> stm.conv_list_to_dict(values_list=[1, 2, 3])
    {0: 1, 1: 2, 2: 3}

    >>> stm.conv_list_to_dict(values_list=['a', 'b', 'c'])
    {0: 'a', 1: 'b', 2: 'c'}

    >>> stm.conv_list_to_dict(None, [10, 20, 30])
    {0: 10, 1: 20, 2: 30}

    >>> stm.conv_list_to_dict(range(5), [10, 20, 30, 40, 50])
    {0: 10, 1: 20, 2: 30, 3: 40, 4: 50}
    """
    if keys_list is None:
        keys_list = list(range(len(values_list)))
    keys_list, values_list = list(keys_list), list(values_list)
    dict_result = dict(zip(keys_list, values_list))
    
    return dict_result


def dict_subset(
    dictionary: dict, 
    keys: list[str] | None = None,
    start_index: int = None, 
    end_index: int = None, 
    n: int | None = None
) -> dict:
    """
    Return a subset of a dictionary based on specified keys, 
    number of elements or index range.

    Parameters
    ----------
    dictionary : dict
        The dictionary to be subset.
    keys : list_like, optional (default=None)
        List of keys to include in the subset. If provided, other parameters 
        are ignored.
    start_index : int, optional (default=None)
        Starting index (inclusive) of the keys to include when keys parameter 
        is not provided.
    end_index : int, optional (default=None)
        Ending index (exclusive) of the keys to include when keys parameter is 
        not provided.
    n : int, optional (default=None)
        Number of first elements to include in the subset.

    Returns
    -------
    dictionary : dict
        The subset of the dictionary.

    Examples
    --------
    >>> import stemlab as stm
    >>> dictionary = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
    >>> stm.dict_subset(dictionary, keys=['a', 'c', 'e'])
    {'a': 1, 'c': 3, 'e': 5}
    >>> stm.dict_subset(dictionary, start_index=1, end_index=4)
    {'b': 2, 'c': 3, 'd': 4}
    >>> stm.dict_subset(dictionary, n=2)
    {'a': 1, 'b': 2}
    >>> stm.dict_subset(dictionary, n=-3)
    {'d': 4, 'e': 5, 'f': 6}
    """
    if keys is not None:
        dictionary = {k: dictionary[k] for k in keys if k in dictionary}
    else:
        if start_index or end_index: # at least one is not None
            if start_index is None and end_index is not None:
                start_index = 0
            else: # start_index is not None and end_index is None:
                end_index = 1000
            dictionary = dict(list(dictionary.items())[start_index:end_index])
    
    if n is not None:
        if n > 0:
            dictionary = dict(list(dictionary.items())[:n])
        else:
            lst = list(dictionary.items())
            m = len(lst)
            if m + n > 0:
                dictionary = dict(lst[m + n:])
            else:
                dictionary = dict(lst)
    
    return dictionary


def dict_sort(dictionary: dict, reverse: bool = False):
    """
    Sort a Python dictionary by its keys.

    Parameters
    ----------
    dictionary : dict
        The dictionary to be sorted.
    reverse : bool, optional (default=False)
        Whether to sort in descending order. Default is False, which 
        means ascending order.

    Returns
    -------
    dict
        The sorted dictionary.

    Examples
    --------
    >>> import stemlab as stm
    
    >>> dictionary = {'b': 2, 'a': 1, 'c': 3}
    >>> stm.dict_sort(dictionary)
    {'a': 1, 'b': 2, 'c': 3}

    >>> stm.dict_sort(dictionary, reverse=True)
    {'c': 3, 'b': 2, 'a': 1}
    """
    return dict(
        sorted(dictionary.items(), key=lambda item: item[0], reverse=reverse)
    )


def list_merge(*lists):
    """
    Merge multiple lists into a single list.

    This function takes multiple lists as arguments and concatenates 
    their elements into a single list. The elements in the resulting 
    list maintain the order in which they appear in the original lists.

    Parameters
    ----------
    *lists : list
        An arbitrary number of lists to concatenate. Each argument 
        should be a list, and they will be concatenated in the order 
        provided.

    Returns
    -------
    merged_list : list
        A new list containing all elements from the input lists.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.list_merge([1, 2, 3], [4, 5, 6], [7, 8, 9])
    [1, 2, 3, 4, 5, 6, 7, 8, 9]

    >>> stm.list_merge(['a', 'b'], ['c', 'd'], ['e'])
    ['a', 'b', 'c', 'd', 'e']

    >>> stm.list_merge([1, 2], [], [3], ['d'])
    [1, 2, 3, 'd']
    """
    return list(chain(*lists))


def tuple_merge(*tuples):
    """
    Merge multiple tuples into a single tuple.

    This function takes multiple tuples as arguments and concatenates 
    their elements into a single tuple. The elements in the resulting 
    tuple maintain the order in which they appear in the original 
    tuples.

    Parameters
    ----------
    *tuples : tuple
        An arbitrary number of tuples to merge. Each argument should be 
        a tuple, and they will be concatenated in the order provided.

    Returns
    -------
    merged_tuple : tuple
        A new tuple containing all elements from the input tuples.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.tuple_merge((1, 2, 3), (4, 5, 6), (7, 8, 9))
    (1, 2, 3, 4, 5, 6, 7, 8, 9)

    >>> stm.tuple_merge(('a', 'b'), ('c', 'd'), ('e',))
    ('a', 'b', 'c', 'd', 'e')

    >>> stm.tuple_merge((1, 2), (), (3,), ('d',))
    (1, 2, 3, 'd')
    """
    return tuple(chain(*tuples))

    
def dict_merge(*dicts) -> dict:
    """
    Merge multiple dictionaries into a single dictionary.

    This function takes multiple dictionaries as arguments and merges 
    them into one dictionary. If there are overlapping keys, the 
    values from the later dictionaries will overwrite the earlier ones.

    Parameters
    ----------
    *dicts : dict
        An arbitrary number of dictionaries to merge.

    Returns
    -------
    merged_dict : dict
        A new dictionary containing all the key-value pairs from the 
        input dictionaries. If the same key appears in multiple 
        dictionaries, the value from the last dictionary with that key 
        is used.

    Examples
    --------
    >>> stm.dict_merge({'a': 1, 'b': 2}, {'b': 3, 'c': 4})
    {'a': 1, 'b': 3, 'c': 4}

    >>> stm.dict_merge({'x': 1}, {'y': 2}, {'z': 3})
    {'x': 1, 'y': 2, 'z': 3}
    
    `value1` in the first dictionary will be replaced by `new_value1` 
    in the second dictionary
    
    >>> stm.dict_merge({'key1': 'value1'},
    ... {'key1': 'new_value1', 'key2': 'value2'})
    {'key1': 'new_value1', 'key2': 'value2'}
    """
    merged_dict = {}
    for dict in dicts:
        merged_dict.update(dict)

    return merged_dict


def arr_table_blank_row(
    data: ArrayMatrixLike,
    to_ndarray: bool = True,
    convert_pd: bool = True,
    col_names: list[str] | None = None,
    na_label: str = '',
    decimal_points: int = 14
):
    """
    Inserts a row with missing values in between two rows or an array.

    Parameters
    ----------
    data: {list, tuple, NDarray, Series}
        The values where we want to insert a row.
    to_ndarray: bool, optional (default=True)
        If `True`, result will be returned as a two dimensional list.
    convert_pd : bool, optional (default=True)
        If `True`, result will be converted to a Pandas DataFrame.
    col_names : {list, tuple, array, Series}, optional (default=None)
        Column names of the Pandas DataFrame. Only used when 
        `convert_pd=True`.
    na_label : str, optional (default='')
        The value to be used for the inserted blank rows. 
        Used when `convert_pd=True`.
    decimal_points : int, optional (default=14)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    result : {pandas.DataFrame, numpy.ndarray}
        A DataFrame or a 2D array with the blank rows inserted.

    Examples
    --------
    >>> import stemlab as stm
    >>> m = [22, 23, 14, 49, 43]
    >>> stm.arr_table_blank_row(data=m, to_ndarray=False)
             C1
        1  22.0
        2      
        3  23.0
        4      
        5  14.0
        6      
        7  49.0
        8      
        9  43.0
    >>> stm.arr_table_blank_row(data=m, to_ndarray=False, na_label='-')
             C1
        1  22.0
        2     -
        3  23.0
        4     -
        5  14.0
        6     -
        7  49.0
        8     -
        9  43.0
    >>> M = [[28, 17, 16, 37, 46], [21, 39, 19, 29, 30]]
    >>> stm.arr_table_blank_row(data=M, to_ndarray=True, na_label='-',
    ... col_names=['x', 'y'])
           x	   y
    1	28.0	21.0
    2	   -	   -
    3	17.0	39.0
    4	   -	   -
    5	16.0	19.0
    6	   -	   -
    7	37.0	29.0
    8	   -	   -
    9	46.0	30.0
    """
    L = []
    M = data
    remove_first_col = False
    # convert to numpy array just in case it is not
    if not isinstance(M, ndarray):
        try:
            if isinstance(M, (list, tuple, set, Series)):
                M = asarray(list(M)).T
                if M.shape[0] == 2 and M.shape[1] > 2:
                    M = M.T
            else: # pandas
                try:
                    M = asarray(M)
                except Exception:
                    raise NumpifyError(par_name='data', data_structure=type(data))
        except Exception:
            raise NumpifyError(par_name='data', data_structure=type(data))
    
    try:
        col_names_count = len(col_names)
    except Exception:
        col_names_count = 0
        col_names = ''
    
    col_names = list(map(str, col_names))
    if col_names_count == 2 and M.shape[1] > 2:
        N = M[:, :2]
    # do not add na row if bc and x1 is in columns
    elif '$b$$c$' in ''.join(col_names) or '_no_blank_rows' in ''.join(col_names):
        N = M
    else:
        # if the data is not 1D, set `to_ndarray` to `False` so that it uses 
        # a list
        if len(M.shape) == 1:
            to_ndarray = False

        if to_ndarray:
            row_count, col_count = M.shape
            if row_count == 2 and col_count > 2:
                M = M.reshape(col_count, row_count)
            row_count, col_count = M.shape
            # row 1
            na = repeat(nan, len(M[:, 0]))
            for i in range(len(M[:, 0])):
                L.append(M[i, 0])
                L.append(na[i])
            L = L[:len(L)-1]
            M1 = array([L]).T
            # from row 1 or row 2
            if isnan(M[0, 1]):
                M = M[:, :] # from row 1
                remove_first_col = True
            else:
                M = M[:, 1:] # from row 2
            row_count, col_count = M.shape
            for i in range(row_count):
                for j in range(col_count):
                    if i < j:
                        M[i, j] = nan
            N = zeros((row_count * 2 - 1, col_count))
            for j in range(col_count):
                Mi = M[:, j]
                na = repeat(nan, len(Mi))
                L = []
                for i in range(len(Mi)):
                    L.append(Mi[i])
                    L.append(na[i])
                L = L[:len(L)-1]
                LNew = L[j:] + [nan] * j # shift values up
                N[:, j] = LNew
            N = hstack([M1, N])
        else:
            M = M.flatten()
            na = repeat(nan, len(M))
            L = []
            for i in range(len(M)):
                L.append(M[i])
                L.append(na[i])
            N = array([L[:len(L)-1]]).T
    
    if remove_first_col:
        N = N[:, 1:]
    result = N

    if convert_pd:
        row_count, col_count = N.shape
        if col_names is None:
            col_names = [f'C{k + 1}' for k in range(col_count)]
        
        df = round(result, decimal_points)
        results_table = array(df, dtype = float64)
        row_names = list(range(row_count))
        row_names = list(map(lambda x: str(x).replace('nan', ''), row_names))
        # remove _no_blank_rows
        col_names = [col.replace('_no_blank_rows', '') for col in col_names]
        try:
            results_table = DataFrame(
                df, index = row_names, columns = col_names
            )
        except Exception:
            col_names = [f'C{n + 1}' for n in range(df.shape[1])]
            results_table = DataFrame(
                data=df, index = row_names, columns = col_names
            )
        result = results_table.fillna(na_label)

    return result


def conv_list_to_string(list_value: list, delimiter: str = ' ') -> str:
    """
    Convert a list of symbols to a string separated by a delimiter.

    Parameters
    ----------
    list_value : list_like
        The list of symbols to convert to a string.
    delimiter : str, optional (default=' ')
        The delimiter to use for joining the symbols.

    Returns
    -------
    list_str : str
        The string representation of the list.

    Examples
    --------
    >>> import stemlab as stm

    >>> stm.conv_list_to_string(list_value=['a', 'b', 'c'], delimiter=', ')
    'a, b, c'

    >>> stm.conv_list_to_string(list_value=['x', 'y', 'z'], delimiter='-')
    'x-y-z'
    """
    from stemlab.core.validators.validate import ValidateArgs

    list_value = conv_to_arraylike(
        array_values=list_value, par_name='list_value'
    )
    delimiter = ValidateArgs.check_string(
        par_name='delimiter', to_lower=True, user_input=delimiter
    )

    list_value = [latex(sympify(symbol)) for symbol in list_value]
    list_value.sort()
    list_str = delimiter.join(map(str, list_value))

    return list_str


def conv_to_arraylike(
    array_values: list | tuple,
    includes_str: bool = False,
    to_tuple: bool = False,
    flatten_list: bool = True,
    n: int | None = None,
    to_ndarray: bool = False,
    to_lower: bool = False,
    label: Literal['at least', 'exactly', 'at most'] = 'exactly',
    par_name: str = 'array_values'
) -> list | NumpyArray:
    """
    Convert an iterable to a list / tuple.

    Parameters
    ----------
    array_values : list_like
        The iterable to be converted.
    includes_str : bool, optional (default=False)
        Whether the iterable includes strings.
    to_tuple : bool, optional (default=False)
        Whether to convert the result to a tuple.
    flatten_list : bool, optional (default=True)
        Whether to flatten the list before conversion.
    n : int, optional (default=None)
        Number of elements the result should have.
    to_ndarray : bool, optional (default=False)
        Whether to convert the result to a numpy ndarray.
    to_lower : bool, optional (default=False)
        Whether to convert the result to a lowercase.
    label : {'at least', 'exactly', 'at most'}, optional (default='exactly')
        Label for the number of elements check.
    par_name : str, optional (default='array_values')
        Name to use in error messages to describe the parameter being 
        checked.

    Returns
    -------
    array_values : {list, tuple, NDarray}
        The converted result.

    Raises
    ------
    IterableError
        If the input is not iterable.
    ValueError
        If conversion to list or tuple fails.
    VectorLengthError
        If the number of elements doesn't meet the specified requirement.
    NumpifyError
        If conversion to ndarray fails.
        
    Examples
    --------
    >>> import stemlab as stm
    
    >>> stm.conv_to_arraylike([1, 2, 3])
    [1, 2, 3]

    >>> stm.conv_to_arraylike([1, 2, 3], to_tuple=True)
    (1, 2, 3)

    >>> stm.conv_to_arraylike([[1, 2], [3, 4]], flatten_list=True)
    [1, 2, 3, 4]

    >>> stm.conv_to_arraylike([1, 2, 3, 4, 5], n=3, label='at most')
    ...
    VectorLengthError: Expected 'array_values' to have 'exactly 3' elements but got: [1, 2, 3, 4, 5]

    >>> stm.conv_to_arraylike([1, 2, 3], to_ndarray=True)
    array([1, 2, 3])

    >>> stm.conv_to_arraylike(['Hello', 'World'], to_lower=True)
    ['hello', 'world']    
    """
    from stemlab.core.validators.validate import ValidateArgs
    # array_values
    if not is_iterable(array_values, includes_str=includes_str):
        raise IterableError(
            par_name=par_name, 
            includes_str=includes_str, 
            user_input=array_values
        )
    
    array_values = [array_values] if isinstance(array_values, str) else array_values
    
    # includes_str, ..., flatten_list
    includes_str = ValidateArgs.check_boolean(user_input=includes_str, default=False)
    to_tuple = ValidateArgs.check_boolean(user_input=to_tuple, default=True)
    flatten_list = ValidateArgs.check_boolean(user_input=flatten_list, default=True)
    to_ndarray = ValidateArgs.check_boolean(user_input=to_ndarray, default=False)
    to_lower = ValidateArgs.check_boolean(user_input=to_ndarray, default=False)
    label = ValidateArgs.check_member(
        par_name='label', 
        valid_items=['at least', 'exactly', 'at most'],
        user_input=label,
        default='exactly' # must NOT be `None`, otherwise it will run infinitely
    )

    # convert to list/tuple
    try:
        array_values = list(array_values) 
        array_values = flatten(array_values) if flatten_list else array_values
        array_values = tuple(array_values) if to_tuple else array_values
    except Exception:
        listuple = 'tuple' if to_tuple else 'list'
        raise ValueError(f"Unable to convert '{par_name}' to a {listuple}")
    
    # check number of elements
    if n is not None:
        n = ValidateArgs.check_numeric(par_name='n', limits=[1, 1000], user_input=n)
        m = len(array_values)
        if (
            (label == 'at least' and m < n) 
            or (label == 'exactly' and m != n) 
            or (label == 'at most' and m > n)
        ):
            length_error = True
        else:
            length_error = False
            
        if length_error:
            raise VectorLengthError(
                par_name=par_name, 
                n=n, 
                label=label, 
                user_input=array_values
            )
    
    if to_ndarray:
        is_ndarray = False
        try:
            is_ndarray = True
            array_values = asarray(array_values)
        except Exception:
            raise NumpifyError(par_name=par_name)
    
    if to_lower and not is_ndarray:
        try:
            array_values = [element.lower() for element in array_values]
        except:
            pass

    return array_values
