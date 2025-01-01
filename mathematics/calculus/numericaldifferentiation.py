from typing import Callable, Literal

from pandas import DataFrame
from sympy import Float, sympify, Expr
from numpy import nan, float64, ndarray, ones, repeat, matrix, hstack, where

from stemlab.core.validators.errors import RequiredError
from stemlab.core.arraylike import conv_to_arraylike, arr_table_blank_row
from stemlab.core.display import Result, display_results
from stemlab.core.decimals import float15
from stemlab.core.htmlatex import sta_dframe_color
from stemlab.core.symbolic import sym_lambdify_expr
from stemlab.core.validators.validate import ValidateArgs


# https://neuron.eng.wayne.edu/auth/ece3040/lectures/lecture21.pdf


class FiniteDifferences:
    """
    Methods
    -------
    first_derivative()
        Computes the first derivative of the function using the
        specified method.
        
    second_derivative()
        Computes the second derivative of the function using the
        specified method.
        
    third_derivative()
        Computes the third derivative of the function using the
        specified method.
        
    fourth_derivative()
        Computes the fourth derivative of the function using the
        specified method.
    """
    
    
    def __init__(self, x, y, x0, h=None) -> None:
        
        self.x = x
        self.y = y
        self.x0 = x0
        self.h = h
        
        self.x = conv_to_arraylike(
            array_values=self.x,
            flatten_list=True,
            to_ndarray=True,
            par_name='x'
        )
        
        self.x = ValidateArgs.check_constant(par_name='x', user_input=self.x)
        self.x = ValidateArgs.check_diff_constant(user_input=self.x, par_name='x')
        
        if self.h is None:
            self.h = self.x[1] - self.x[0]
        
        if isinstance(self.y, (list, tuple, ndarray)):
            self.y = conv_to_arraylike(
                array_values=self.y,
                flatten_list=True,
                to_ndarray=True,
                par_name='y'
            )
            self.y = ValidateArgs.check_constant(par_name='y', user_input=self.y)
        else:
            self.y = ValidateArgs.check_function(f=self.y, is_univariate=True, par_name='y')
            self.y = self.y(self.x)
            
        _ = ValidateArgs.check_len_equal(x=self.x, y=self.y, par_name=['x', 'y'])
        
        self.dframe = DataFrame([self.x, self.y]).T
        self.dframe.columns = ['x', 'y']
        
        if self.x0 is None:
            raise RequiredError(par_name='x0')
        
        self.x0 = ValidateArgs.check_numeric(
            par_name='x0', to_float=True, user_input=self.x0
        )
        
        self.idx = where(self.x == self.x0)[0]
        if self.idx.size == 0:
            raise ValueError(
                f"The value {self.x0} was not found in the array 'x'"
            )

        # `self.x` must be converted to list because of the use of 
        # `index` method in the class methods below.
        self.x = self.x.tolist()
    
    def first_derivative(self, method='2-point-centered') -> None:
        
        x, y, h, x0 = self.x, self.y, self.h, self.x0
        valid_methods = [
            '2-point-backward',
            '2-point-forward',
            '2-point-centered',
            '3-point-backward',
            '3-point-forward',
            '4-point-centered',
            '4-point-endpoint'
        ]
        method = ValidateArgs.check_member(
            par_name='method', valid_items=valid_methods, user_input=method
        )
        
        if method == '2-point-backward':
            result = (
                y[x.index(float15(x0))] 
                - y[x.index(float15(x0 - h))]) / h
        elif method == '2-point-forward':
            result = (
                y[x.index(float15(x0 + h))] 
                - y[x.index(float15(x0))]) / h
        elif method == '2-point-centered':
            result = (
                y[x.index(float15(x0 + h))] 
                - y[x.index(float15(x0 - h))]) / (2 * h)
        elif method == '3-point-backward':
            result = (
                y[x.index(float15(x0 - 2 * h))] 
                - 4 * y[x.index(float15(x0 - h))] 
                + 3 * y[x.index(float15(x0))]) / (2 * h)
        elif method == '3-point-forward':
            result = (
                -3 * y[x.index(float15(x0))] 
                + 4 * y[x.index(float15(x0 + h))] 
                - y[x.index(float15(x0 + 2 * h))]) / (2 * h)
        elif method == '4-point-centered':
            result = (
                y[x.index(float15(x0 - 2 * h))] 
                - 8 * y[x.index(float15(x0 - h))] 
                + 8 * y[x.index(float15(x0 + h))] 
                - y[x.index(float15(x0 + 2 * h))]) / (12 * h)
        elif method == '4-point-endpoint':
            result = (
                -25 * y[x.index(float15(x0))] 
                + 48 * y[x.index(float15(x0 + h))] 
                - 36 * y[x.index(float15(x0 + 2 * h))] 
                + 16 * y[x.index(float15(x0 + 3 * h))] 
                - 3 * y[x.index(float15(x0 + 4 * h))]) / (12 * h)

        return self.dframe, result
    
    def second_derivative(self, method='backward') -> None:
        
        x, y, h, x0 = self.x, self.y, self.h, self.x0
        valid_methods = [
            '3-point-backward',
            '3-point-forward',
            '3-point-centered',
            '4-point-backward',
            '4-point-forward',
            '5-point-centered'
        ]
        method = ValidateArgs.check_member(
            par_name='method', valid_items=valid_methods, user_input=method
        )
        
        if method == '3-point-backward':
            result = (
                y[x.index(float15(x0 - 2 * h))] 
                - 2 * y[x.index(float15(x0 - h))] 
                + y[x.index(float15(x0))]) / h ** 2
        elif method == '3-point-forward':
            result = (
                y[x.index(float15(x0))] 
                - 2 * y[x.index(float15(x0 + h))] 
                + y[x.index(float15(x0 + 2 * h))]) / h ** 2
        elif method == '3-point-centered':
            result = (
                y[x.index(float15(x0 - h))] 
                - 2 * y[x.index(float15(x0))] 
                + y[x.index(float15(x0 + h))]) / h ** 2
        elif method == '4-point-backward':
            result = (
                -y[x.index(float15(x0 - 3 * h))] 
                + 4 * y[x.index(float15(x0 - 2 * h))] 
                - 5 * y[x.index(float15(x0 - h))] 
                + 2 * y[x.index(float15(x0))]) / h ** 2
        elif method == '4-point-forward':
            result = (
                2 * y[x.index(float15(x0))] 
                - 5 * y[x.index(float15(x0 + h))] 
                + 4 * y[x.index(float15(x0 + 2 * h))] 
                - y[x.index(float15(x0 + 3 * h))]) / h ** 2
        elif method == '5-point-centered':
            result = (
                -y[x.index(float15(x0 - 2 * h))] 
                + 16 * y[x.index(float15(x0 - h))] 
                - 30 * y[x.index(float15(x0))] 
                + 16 * y[x.index(float15(x0 + h))] 
                - y[x.index(float15(x0 + 2 * h))]) / (12 * h ** 2)
        
        return self.dframe, result

    
    def third_derivative(self, method='backward') -> None:
        
        x, y, h, x0 = self.x, self.y, self.h, self.x0
        valid_methods = [
            '4-point-backward',
            '4-point-forward',
            '4-point-centered',
            '5-point-backward',
            '5-point-forward',
            '6-point-centered'
        ]
        method = ValidateArgs.check_member(
            par_name='method', valid_items=valid_methods, user_input=method
        )
        
        if method == '4-point-backward':
            result = (
                -y[x.index(float15(x0 - 3 * h))] 
                + 3 * y[x.index(float15(x0 - 2 * h))] 
                - 3 * y[x.index(float15(x0 - h))] 
                + y[x.index(float15(x0))]) / h ** 3
        elif method == '4-point-forward':
            result = (
                -y[x.index(float15(x0))] 
                + 3 * y[x.index(float15(x0 + h))] 
                - 3 * y[x.index(float15(x0 + 2 * h))] 
                + y[x.index(float15(x0 + 3 * h))]) / h ** 3
        elif method == '4-point-centered':
            result = (
                -y[x.index(float15(x0 - 2 * h))] 
                + 2 * y[x.index(float15(x0 - h))] 
                - 2 * y[x.index(float15(x0 + h))] 
                + y[x.index(float15(x0 + 2 * h))]) / (2 * h ** 3)
        elif method == '5-point-backward':
            result = (
                3 * y[x.index(float15(x0 - 4 * h))] 
                - 14 * y[x.index(float15(x0 - 3 * h))] 
                + 24 * y[x.index(float15(x0 - 2 * h))] 
                - 18 * y[x.index(float15(x0 - h))] 
                + 5 * y[x.index(float15(x0))]) / (2 * h ** 3)
        elif method == '5-point-forward':
            result = (
                -5 * y[x.index(float15(x0))] 
                + 18 * y[x.index(float15(x0 + h))] 
                - 24 * y[x.index(float15(x0 + 2 * h))] 
                + 14 * y[x.index(float15(x0 + 3 * h))] 
                - 3 * y[x.index(float15(x0 + 4 * h))]) / (2 * h ** 3)
        elif method == '6-point-centered':
            result = (
                y[x.index(float15(x0 - 3 * h))] 
                - 8 * y[x.index(float15(x0 - 2 * h))] 
                + 13 * y[x.index(float15(x0 - h))] 
                - 13 * y[x.index(float15(x0 + h))] 
                + 8 * y[x.index(float15(x0 + 2 * h))] 
                - y[x.index(float15(x0 + 3 * h))]) / (8 * h ** 3)

        return self.dframe, result
    
    
    def fourth_derivative(self, method='backward') -> None:
        
        x, y, h, x0 = self.x, self.y, self.h, self.x0
        valid_methods = [
            '5-point-backward',
            '5-point-forward',
            '5-point-centered',
            '6-point-backward',
            '6-point-forward',
            '7-point-centered'
        ]
        method = ValidateArgs.check_member(
            par_name='method', valid_items=valid_methods, user_input=method
        )
        
        if method == '5-point-backward':
            result = (
                y[x.index(float15(x0 - 4 * h))] 
                - 4 * y[x.index(float15(x0 - 3 * h))] 
                + 6 * y[x.index(float15(x0 - 2 * h))] 
                - 4 * y[x.index(float15(x0 - h))] 
                + y[x.index(float15(x0))]) / h ** 4
        elif method == '5-point-forward':
            result = (
                y[x.index(float15(x0))] 
                - 4 * y[x.index(float15(x0 + h))] 
                + 6 * y[x.index(float15(x0 + 2 * h))] 
                - 4 * y[x.index(float15(x0 + 3 * h))] 
                + y[x.index(float15(x0 + 4 * h))]) / h ** 4
        elif method == '5-point-centered':
            result = (
                y[x.index(float15(x0 - 2 * h))] 
                - 4 * y[x.index(float15(x0 - h))] 
                + 6 * y[x.index(float15(x0))] 
                - 4 * y[x.index(float15(x0 + h))] 
                + y[x.index(float15(x0 + 2 * h))]) / h ** 4
        elif method == '6-point-backward':
            result = (
                -2 * y[x.index(float15(x0 - 5 * h))] 
                + 11 * y[x.index(float15(x0 - 4 * h))] 
                - 24 * y[x.index(float15(x0 - 3 * h))] 
                + 26 * y[x.index(float15(x0 - 2 * h))] 
                - 14 * y[x.index(float15(x0 - h))] 
                + 3 * y[x.index(float15(x0))]) / h ** 4
        elif method == '6-point-forward':
            result = (
                3 * y[x.index(float15(x0))] 
                - 14 * y[x.index(float15(x0 + h))] 
                + 26 * y[x.index(float15(x0 + 2 * h))] 
                - 24 * y[x.index(float15(x0 + 3 * h))] 
                + 11 * y[x.index(float15(x0 + 4 * h))] 
                - 2 * y[x.index(float15(x0 + 5 * h))]) / h ** 4
        elif method == '7-point-centered':
            result = (
                y[x.index(float15(x0 - 3 * h))] 
                + 12 * y[x.index(float15(x0 - 2 * h))] 
                - 39 * y[x.index(float15(x0 - h))] 
                + 56 * y[x.index(float15(x0))] 
                + 39 * y[x.index(float15(x0 + h))] 
                + 12 * y[x.index(float15(x0 + 2 * h))] 
                - y[x.index(float15(x0 + 3 * h))]) / (6 * h ** 4)

        return self.dframe, result
        
        
def diff_fd_first_derivative(
    x: list[float],
    y: list[float] | str | Expr | Callable,
    x0: float,
    h: float = None,
    method: Literal[
        '2-point-centered', '2-point-forward', '2-point-centered',
        '3-point-backward', '3-point-forward', '4-point-centered',
        '4-point-endpoint'
    ] = '2-point-centered',
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Computes the numerical first derivative using a specified method.
    
    Parameters
    ----------
    x : list
        The list of x-values for the data points.
    y : {list, str, Expr, Callable}
        The list of y-values for the data points or a function f(x).
    x0 : float
        The point at which to compute the derivative.
    h : float, optional (default=None)
        The interval. If `None`, then `h` will be calculated as the
        difference between any two consecutive values x[i+1] - x[i].
    method : str, optional (default='2-point-centered')
        The finite difference method to use. Valid methods include:
        ========================================================
        method                              Description    
        ========================================================
        2-point-backward .................. Two point backward
        2-point-forward ................... Two point forward
        2-point-centered .................. Two point centered
        3-point-backward .................. Three point backward
        3-point-forward ................... Three point forward
        4-point-centered .................. Four point centered
        4-point-endpoint .................. Four point endpoint
        ========================================================
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=14)
        Number of decimal points for the result.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the values of `x` and `y` as entered.
    answer : Float
        Solution of the numerical derivative.
    
    Examples
    -------- 
    >>> import stemlab as stm
    >>> x = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
    ... 2.5, 2.6]
    >>> y = [5.67727995, 6.72253361, 7.92485188, 9.30571057,
    ... 10.88936544, 12.70319944, 14.7781122, 17.14895682, 19.8550297,
    ... 22.94061965, 26.45562331, 30.4562349, 35.00571889]
    >>> result = stm.diff_fd_first_derivative(x, y, x0=2.0,
    ... method='2-point-centered', auto_display=True)
          x            y
    0   1.4   5.67727995
    1   1.5   6.72253361
    2   1.6   7.92485188
    3   1.7   9.30571057
    4   1.8  10.88936544
    5   1.9  12.70319944
    6   2.0  14.77811220
    7   2.1  17.14895682
    8   2.2  19.85502970
    9   2.3  22.94061965
    10  2.4  26.45562331
    11  2.5  30.45623490
    12  2.6  35.00571889
    
    Answer = 22.2287869
    
    >>> methods = [
    ... '2-point-backward', '2-point-forward', '2-point-centered',
    ... '3-point-backward', '3-point-forward', '4-point-centered',
    ... '4-point-endpoint']
    >>> results = []
    >>> for method in methods:
    ...     result = stm.diff_fd_first_derivative(x, y, x0=2.0,
    ...     method=method, auto_display=False)
    ...     results.append([method, result.answer])
    >>> dframe = pd.DataFrame(results, columns=['Method', 'Answer'])
    >>> dframe.round(8)
                 Method       Answer
    0  2-point-backward  20.74912760
    1   2-point-forward  23.70844620
    2  2-point-centered  22.22878690
    3  3-point-backward  22.05452140
    4   3-point-forward  22.03230490
    5  4-point-centered  22.16699565
    6  4-point-endpoint  22.16591469
    """
    diff = FiniteDifferences(x=x, y=y, x0=x0, h=h)
    dframe, answer = diff.first_derivative(method=method)
    if auto_display:
        display_results({
            'table': dframe,
            'Answer': answer,
            'decimal_points': decimal_points
        })
    
    result = Result(table=dframe, answer=answer)
    
    return result


def diff_fd_second_derivative(
    x: list[float],
    y: list[float] | str | Expr | Callable,
    x0: float,
    h: float = None,
    method: Literal[
        '3-point-backward', '3-point-forward', '3-point-centered',
        '4-point-backward', '4-point-forward', '5-point-centered'
    ] = '3-point-centered',
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Computes the numerical second derivative using a specified method.
    
    Parameters
    ----------
    x : list
        The list of x-values for the data points.
    y : {list, str, Expr, Callable}
        The array of y-values for the data points or a function f(x).
    x0 : float
        The point at which to compute the derivative.
    h : float, optional (default=None)
        The interval. If `None`, then `h` will be calculated as the
        difference between any two consecutive values x[i+1] - x[i].
    method : str, optional (default='3-point-centered')
        The finite difference method to use. Valid methods include:
        ========================================================
        method                              Description  
        ========================================================
        3-point-backward .................. Three point backward
        3-point-forward ................... Three point forward
        3-point-centered .................. Three point centered
        4-point-backward .................. Four point backward
        4-point-forward ................... Four point forward
        5-point-centered .................. Five point centered
        ========================================================
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=14)
        Number of decimal points for the result.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the values of `x` and `y` as entered.
    answer : Float
        Solution of the numerical derivative.
        
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
    ... 2.5, 2.6]
    >>> y = [5.67727995, 6.72253361, 7.92485188, 9.30571057,
    ... 10.88936544, 12.70319944, 14.7781122, 17.14895682, 19.8550297,
    ... 22.94061965, 26.45562331, 30.4562349, 35.00571889]
    >>> result = stm.diff_fd_second_derivative(x, y, x0=2.0,
    ... method='3-point-centered', auto_display=True)
          x            y
    0   1.4   5.67727995
    1   1.5   6.72253361
    2   1.6   7.92485188
    3   1.7   9.30571057
    4   1.8  10.88936544
    5   1.9  12.70319944
    6   2.0  14.77811220
    7   2.1  17.14895682
    8   2.2  19.85502970
    9   2.3  22.94061965
    10  2.4  26.45562331
    11  2.5  30.45623490
    12  2.6  35.00571889
    
    Answer = 29.593186
    
    >>> methods = [
    ... '3-point-backward', '3-point-forward', '3-point-centered',
    ... '4-point-backward', '4-point-forward', '5-point-centered']
    >>> results = []
    >>> for method in methods:
    ...     result = stm.diff_fd_second_derivative(x, y, x0=2.0,
    ...     method=method, auto_display=False)
    ...     results.append([method, result.answer])
    >>> dframe = pd.DataFrame(results, columns=['Method', 'Answer'])
    >>> dframe.round(8)
                 Method      Answer
    0  3-point-backward  26.1078760
    1   3-point-forward  33.5228260
    2  3-point-centered  29.5931860
    3  4-point-backward  29.1978390
    4   4-point-forward  29.0939450
    5  5-point-centered  29.5561585
    """
    diff = FiniteDifferences(x=x, y=y, x0=x0, h=h)
    dframe, answer = diff.second_derivative(method=method)
    if auto_display:
        display_results({
            'table': dframe,
            'Answer': answer,
            'decimal_points': decimal_points
        })
    
    result = Result(table=dframe, answer=answer)
    
    return result


def diff_fd_third_derivative(
    x: list[float],
    y: list[float] | str | Expr | Callable,
    x0: float,
    h: float = None,
    method: Literal[
        '4-point-backward', '4-point-forward', '4-point-centered',
        '5-point-backward', '5-point-forward', '6-point-centered'
    ] = '4-point-centered',
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Computes the numerical third derivative using a specified method.
    
    Parameters
    ----------
    x : list
        The list of x-values for the data points.
    y : {list, str, Expr, Callable}
        The list of y-values for the data points or a function f(x).
    x0 : float
        The point at which to compute the derivative.
    h : float, optional (default=None)
        The interval. If `None`, then `h` will be calculated as the
        difference between any two consecutive values x[i+1] - x[i]
    method : str, optional (default='4-point-centered')
        The finite difference method to use. Valid methods include:
        ========================================================
        method                              Description  
        ========================================================
        4-point-backward .................. Four point backward
        4-point-forward ................... Four point forward
        4-point-centered .................. Four point centered
        5-point-backward .................. Five point backward
        5-point-forward ................... Five point forward
        6-point-centered .................. Six point centered
        ========================================================
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=14)
        Number of decimal points for the result.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the values of `x` and `y` as entered.
    answer : Float
        Solution of the numerical derivative.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
    ... 2.5, 2.6]
    >>> y = [5.67727995, 6.72253361, 7.92485188, 9.30571057,
    ... 10.88936544, 12.70319944, 14.7781122, 17.14895682, 19.8550297,
    ... 22.94061965, 26.45562331, 30.4562349, 35.00571889]
    >>> result = stm.diff_fd_third_derivative(x, y, x0=2.0,
    ... method='4-point-centered', auto_display=True)
          x            y
    0   1.4   5.67727995
    1   1.5   6.72253361
    2   1.6   7.92485188
    3   1.7   9.30571057
    4   1.8  10.88936544
    5   1.9  12.70319944
    6   2.0  14.77811220
    7   2.1  17.14895682
    8   2.2  19.85502970
    9   2.3  22.94061965
    10  2.4  26.45562331
    11  2.5  30.45623490
    12  2.6  35.00571889
    
    Answer = 37.07475
    
    >>> methods = [
    ... '4-point-backward', '4-point-forward', '4-point-centered',
    ... '5-point-backward', '5-point-forward', '6-point-centered']
    >>> results = []
    >>> for method in methods:
    ...     result = stm.diff_fd_third_derivative(x, y, x0=2.0,
    ...     method=method, auto_display=False)
    ...     results.append([method, result.answer])
    >>> dframe = pd.DataFrame(results, columns=['Method', 'Answer'])
    >>> dframe.round(8)
                 Method      Answer
    0  4-point-backward  30.8996300
    1   4-point-forward  44.2888100
    2  4-point-centered  37.0747500
    3  5-point-backward  36.1746500
    4   5-point-forward  35.8770650
    5  6-point-centered  36.9448825
    """
    diff = FiniteDifferences(x=x, y=y, x0=x0, h=h)
    dframe, answer = diff.third_derivative(method=method)
    if auto_display:
        display_results({
            'table': dframe,
            'Answer': answer,
            'decimal_points': decimal_points
        })
    
    result = Result(table=dframe, answer=answer)
    
    return result


def diff_fd_fourth_derivative(
    x: list[float],
    y: list[float] | str | Expr | Callable,
    x0: float,
    h: float = None,
    method: Literal[
        '5-point-backward', '5-point-forward', '5-point-centered',
        '6-point-backward', '6-point-forward', '7-point-centered'
    ] = '5-point-centered',
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Computes the numerical fourth derivative using a specified method.
    
    Parameters
    ----------
    x : list
        The list of x-values for the data points.
    y : {list, str, Expr, Callable}
        The list of y-values for the data points or a function f(x).
    x0 : float
        The point at which to compute the derivative.
    h : float, optional (default=None)
        The interval. If `None`, then `h` will be calculated as the
        difference between any two consecutive values x[i+1] - x[i]
    method : str, optional (default='5-point-centered')
        The finite difference method to use. Valid methods include:
        ========================================================
        method                              Description  
        ========================================================
        5-point-backward .................. Five point backward
        5-point-forward ................... Five point forward
        5-point-centered .................. Five point centered
        6-point-backward .................. Six point backward
        6-point-forward ................... Six point forward
        7-point-centered .................. Seven point centered
        ========================================================
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=14)
        Number of decimal points for the result.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the values of `x` and `y` as entered.
    answer : Float
        Solution of the numerical derivative.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
    ... 2.5, 2.6]
    >>> y = [5.67727995, 6.72253361, 7.92485188, 9.30571057,
    ... 10.88936544, 12.70319944, 14.7781122, 17.14895682, 19.8550297,
    ... 22.94061965, 26.45562331, 30.4562349, 35.00571889]
    >>> result = stm.diff_fd_fourth_derivative(x, y, x0=2.0,
    ... method='5-point-centered', auto_display=True)
          x            y
    0   1.4   5.67727995
    1   1.5   6.72253361
    2   1.6   7.92485188
    3   1.7   9.30571057
    4   1.8  10.88936544
    5   1.9  12.70319944
    6   2.0  14.77811220
    7   2.1  17.14895682
    8   2.2  19.85502970
    9   2.3  22.94061965
    10  2.4  26.45562331
    11  2.5  30.45623490
    12  2.6  35.00571889
    
    Answer = 44.433000000019
    
    >>> methods = [
    ... '5-point-backward', '5-point-forward', '5-point-centered',
    ... '6-point-backward', '6-point-forward', '7-point-centered'
    ... ]
    >>> results = []
    >>> for method in methods:
    ...     result = stm.diff_fd_fourth_derivative(x, y, x0=2.0,
    ...     method=method, auto_display=False)
    ...     results.append([method, result.answer])
    >>> dframe = pd.DataFrame(results, columns=['Method', 'Answer'])
    >>> dframe.round(8)
                 Method                Answer
    0  5-point-backward  3.51668000000000e+01
    1   5-point-forward  5.60783000000000e+01
    2  5-point-centered  4.44330000000000e+01
    3  6-point-backward  4.29566000000000e+01
    4   6-point-forward  4.22833000000000e+01
    5  7-point-centered  2.26042775603333e+06
    """
    diff = FiniteDifferences(x=x, y=y, x0=x0, h=h)
    dframe, answer = diff.fourth_derivative(method=method)
    if auto_display:
        display_results({
            'table': dframe,
            'Answer': answer,
            'decimal_points': decimal_points
        })
    
    result = Result(table=dframe, answer=answer)
    
    return result


def diff_richardson(
    fexpr: str | Expr | Callable,
    x: int | float,
    n: int = 3,
    h: float = 0.01,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Approximates the first derivative f'(x) at a given point x using 
    Richardson extrapolation.

    Parameters
    ----------
    fexpr : {str, sympy.Expr, Callable}
        Expression representing the function whose derivative is 
        required.
    x : {int, float}
        The value of x at which to find the derivative.
    n : int, optional (default=3)
        Number of levels of extrapolation.
    h : {int, float}, optional (default=0.01)
        Initial step size.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    answer : Float
        The value of the numerical derivative of `f` at `x` i.e. f(x).

    Examples
    --------
    >>> f = '2 ** x * sin(x)'
    >>> x, n, h = (1.05, 4, 0.4)
    >>> result = stm.diff_richardson(f, x, n, h, decimal_points=12)
    Stepsize              R1              R2              R3              R4
         h/1  2.203165697392                                                
                              2.275261092087                                
         h/2  2.257237243413                  2.275145945261                
                              2.275153141937                  2.275145841731
         h/4  2.270674167306                  2.275145843349                
                              2.275146299511                                
         h/8   2.27402826646                                                

    Answer = 2.275145841731
    
    >>> f = '(3/7) * x + exp(x)'
    >>> x, n, h = (0.0, 4, 0.45)
    >>> result = stm.diff_richardson(f, x, n, h, decimal_points=12)
    Stepsize              R1              R2              R3             R4
         h/1  1.462664799536                                               
                              1.428485482494                               
         h/2  1.437030311755                   1.42857145441               
                              1.428566081165                  1.42857142857
         h/4  1.430682138813                  1.428571428974               
                              1.428571094736                               
         h/8  1.429098855755                                               
         
    Answer = 1.42857142857

    >>> f = 'x ** 3 * cos(x)'
    >>> x, n, h = (2.3, 4, 0.4)
    >>> result = stm.diff_richardson(f, x, n, h, decimal_points=12)
    Stepsize               R1               R2               R3               R4
         h/1 -19.471761040256                                                   
                              -19.651043713986                                  
         h/2 -19.606223045554                  -19.646799217856                 
                              -19.647064498864                  -19.646795774529
         h/4 -19.636854135536                  -19.646795828331                 
                               -19.64681262024                                  
         h/8 -19.644322999064
                                                       
    Answer = -19.646795774529
    """
    f = sym_lambdify_expr(fexpr=fexpr, is_univariate=True, par_name='fexpr')
    x = ValidateArgs.check_numeric(user_input=x, par_name='x')
    n = ValidateArgs.check_numeric(
        par_name='n',
        limits=[1, 25],
        is_integer=True,
        user_input=n, 
        to_float=False
    )
    h = ValidateArgs.check_numeric(user_input=h, par_name='h')
    auto_display = ValidateArgs.check_boolean(user_input=auto_display, default=True)
    decimal_points = ValidateArgs.check_decimals(x=decimal_points)

    # begin calculations
    N = nan * ones((n, n ), dtype=float64)
    for i in range(n):
        N[i, 0] = (1 / 2) * (f(x + h) - f(x - h)) / h
        p4powerj = 1
        for j in range(1, i + 1):
            p4powerj *= 4
            N[i, j] = (
                N[i, j - 1] + (N[i, j - 1] - N[i - 1, j - 1]) / (p4powerj - 1)
            )
        h *= 0.5
    ncols = N.shape[1]
    col_names = ['C%s' %(k + 1) for k in range(ncols)]
    row_names = ['h/%s' %(2**k) for k in range(ncols)]
    n = len(row_names)
    row_names = list(repeat(row_names, 2)[:-1])
    row_names[1::2] = [''] * (n - 1)
    answer = sympify(N[-1, -1]) # should be here (i.e. before converting df)
    N = arr_table_blank_row(data=N, to_ndarray=True, decimal_points=14)
    N = hstack([matrix(row_names).T, N])
    dframe = DataFrame(N)
    dframe.columns = ['Stepsize'] + col_names
    
    # css styled
    dframe_styled = sta_dframe_color(
        dframe=dframe,
        style_indices=[[n - 1, -1]],
        decimal_points=decimal_points,
    )
    
    if auto_display:
        display_results({
            'dframe': dframe_styled,
            'Answer':answer,
            'decimal_points': decimal_points
        })
    
    result = Result(table=dframe, table_styled=dframe_styled, answer=answer)
    
    return result