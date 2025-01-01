from typing import Literal, Callable

from numpy import (
    ones, nan, array, linspace, dot, vstack, arange, 
    column_stack, where, isin, ceil, repeat, hstack, matrix
)
from scipy.special import roots_legendre
from pandas import DataFrame
from sympy import sympify, Float, Expr
from stemlab.core.display import Result, display_results
from stemlab.core.validators.errors import LowerGteUpperError, IntervalError
from stemlab.core.htmlatex import sta_dframe_color
from stemlab.core.symbolic import sym_lambdify_expr
from stemlab.core.arraylike import (
    conv_to_arraylike, arr_abrange, arr_table_blank_row
)
from stemlab.statistical.wrangle import dframe_labels
from stemlab.core.base.constraints import max_rows
from stemlab.core.validators.validate import ValidateArgs


def int_cotes(
    fexpr: str | Expr | Callable,
    a: float | int, 
    b: float | int, 
    points: int = 5,
    closed_methods: bool = True,
    auto_display: bool = True,
    decimal_points: int = 12
) -> tuple[DataFrame, Float]:
    """
    Numerical integration using Newton-Cotes methods.

    Parameters
    ----------
    fexpr : {str, sympy.Expr, callable}
        The function to be integrated.
    a : {int, float}
        Lower limit of integration.
    b : {int, float}
        Upper limit of integration.
    points : int, optional (default=5)
        Number of points / nodes.
    closed_methods : bool, optional (default=True)
        If `True`, use closed methods; otherwise, use open methods.
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
        Solution of the numerical integral.

    References
    ----------
    https://archive.lib.msu.edu/crcmath/math/math/n/n080.htm  
    (individual equations)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.nquad.html#scipy.integrate.nquad
    https://docs.scipy.org/doc/scipy/reference/integrate.html

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import stemlab as stm

    Newton-Cotes closed methods

    >>> result = stm.int_cotes(fexpr='sin(x)', a=0, b=np.pi/4,
    ... points=5, closed_methods=True, decimal_points=12)
                    x     k            f(x)           k f(x)
    0  0.000000000000   7.0  0.000000000000   0.000000000000
    1  0.196349540849  32.0  0.195090322016   6.242890304516
    2  0.392699081699  12.0  0.382683432365   4.592201188381
    3  0.589048622548  32.0  0.555570233020  17.778247456627
    4  0.785398163397   7.0  0.707106781187   4.949747468306
    
    Answer = 0.292893182561
    
    >>> result_table = []
    >>> for n in range(2, 12):
    ...     result = stm.int_cotes(fexpr='sin(x)', a=0, b=np.pi/4,
    ...     points=n, closed_methods=True, decimal_points=12)
    ...     result_table.append([n, result.answer])
    >>> dframe = pd.DataFrame(result_table, columns=['n', 'result'])
    >>> dframe
        n             answer
    0   2  0.277680183635000
    1   3  0.292932637840000
    2   4  0.292910702549000
    3   5  0.292893182561000
    4   6  0.292893198409000
    5   7  0.292893218841000
    6   8  0.292893218830000
    7   9  0.292893218813000
    8  10  0.292960243992000
    9  11  0.292893218813000

    Newton-Cotes open methods

    >>> result = stm.int_cotes(fexpr='sin(x)', a=0, b=np.pi/4,
    ... points=5, closed_methods=False, decimal_points=12)
                    x     k            f(x)          k f(x)
    0  0.000000000000   NaN  0.000000000000             NaN
    1  0.130899693900  11.0  0.130526192220  1.435788114421
    2  0.261799387799 -14.0  0.258819045103 -3.623466631435
    3  0.392699081699  26.0  0.382683432365  9.949769241492
    4  0.523598775598 -14.0  0.500000000000 -7.000000000000
    5  0.654498469498  11.0  0.608761429009  6.696375719096
    6  0.785398163397   NaN  0.707106781187             NaN

    Answer = 0.292893292327

    >>> result_table = []
    >>> for n in range(1, 8):
    ...     result = stm.int_cotes(fexpr='sin(x)', a=0, b=np.pi/4,
    ...     points=n, closed_methods=False, decimal_points=12)
    ...     result_table.append([n, result.answer])
    >>> dframe = pd.DataFrame(result_table, columns=['n', 'answer'])
    >>> dframe
       n             answer
    0  1  0.300558864942000
    1  2  0.297987542187000
    2  3  0.292858659193000
    3  4  0.292869228136000
    4  5  0.292893292327000
    5  6  0.292893270705000
    6  7  0.292893218723000
    """
    f = sym_lambdify_expr(fexpr=fexpr, is_univariate=True, par_name='fexpr')
    a = ValidateArgs.check_numeric(user_input=a, par_name='a')
    b = ValidateArgs.check_numeric(user_input=b, par_name='b')
    if a >= b:
        raise LowerGteUpperError(
            par_name='Limits', 
            lower_par_name='a', 
            upper_par_name='b', 
            user_input=[a, b]
        )
    # get closed_methods to get (minn, maxn) to use in points
    closed_methods = ValidateArgs.check_boolean(
        user_input=closed_methods, default=True
    )
    minn, maxn = (2, 11) if closed_methods else (1, 7)
    points = ValidateArgs.check_numeric(
        user_input=points, 
        to_float=False, 
        limits=[minn, maxn], 
        par_name='points'
    )
    n = points - 1 if closed_methods else points + 1
    h = (b - a) / n
    x = arr_abrange(a, b, h)
    y = f(x)
    auto_display = ValidateArgs.check_boolean(
        user_input=auto_display, default=True
    )
    decimal_points = ValidateArgs.check_decimals(x=decimal_points)
    data = False
    dframe, dframe_styled, answer = _int_cotes_table(
        x, y, data, closed_methods, points, decimal_points
    )
    
    if auto_display:
        display_results({
            'dframe': dframe_styled,
            'Answer': answer,
            'decimal_points': decimal_points
        })

    result = Result(table=dframe, table_styled=dframe_styled, answer=answer)
    
    return result

    
def int_cotes_data(
    x: list[float],
    y: list[float],
    points: int = 5,
    closed_methods: bool = True,
    auto_display: bool = True,
    decimal_points: int = 12
) -> tuple[DataFrame, Float]:
    """
    Numerical integration using Newton-Cotes methods given sample data points.

    Parameters
    ----------
    x : array_like
        The points corresponding to the ``y`` values.
    y : array_like
        The points to be integrated, synonymous to f(x).
    points : int, optional (default=5)
        Number of points / nodes.
    closed_methods : bool, optional (default=True)
        If `True`, use closed methods; otherwise, use open methods.
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
        Solution of the numerical integral.

    References
    ----------
    https://archive.lib.msu.edu/crcmath/math/math/n/n080.htm  
    (individual equations)

    Examples
    --------
    >>> import stemlab as stm

    >>> df = stm.dataset_read(name='cotes_n8')
    >>> x = df['x'].values
    >>> y = df['y'].values
    >>> result = stm.int_cotes_data(x, y, points=8, closed_methods=True)
                x       k           y                   ky
    0  0.00000000   751.0  0.20000000   150.19999999999999
    1  0.11428571  3577.0  1.30674076  4674.21169852000003
    2  0.22857143  1323.0  1.31892807  1744.94183661000011
    3  0.34285714  2989.0  1.92454430  5752.46291269999983
    4  0.45714286  2989.0  2.99838448  8962.17121072000009
    5  0.57142857  1323.0  3.53789492  4680.63497915999960
    6  0.68571429  3577.0  2.59901184  9296.66535168000155
    7  0.80000000   751.0  0.23200000   174.23200000000000
    
    Answer = 1.640533271322
    """
    x = conv_to_arraylike(array_values=x, to_ndarray=True, par_name='x')
    y = conv_to_arraylike(array_values=y, to_ndarray=True, par_name='y')
    _ = ValidateArgs.check_len_equal(x=x, y=y, par_name=['x', 'y'])
    _ = ValidateArgs.check_diff_constant(user_input=x, par_name='x')
    closed_methods = ValidateArgs.check_boolean(user_input=closed_methods, default=True)
    minn, maxn = (2, 11) if closed_methods else (1, 7)
    points = ValidateArgs.check_numeric(
        user_input=points, 
        to_float=False, 
        limits=[minn, maxn], 
        par_name='points'
    )
    auto_display = ValidateArgs.check_boolean(user_input=auto_display, default=True)
    decimal_points = ValidateArgs.check_decimals(x=decimal_points)
    
    data = True
    dframe, dframe_styled, answer = _int_cotes_table(
        x, y, data, closed_methods, points, decimal_points
    )
    
    if auto_display:
        display_results({
            'dframe': dframe_styled,
            'Answer': answer,
            'decimal_points': decimal_points
        })
        
    result = Result(table=dframe, table_styled=dframe_styled, answer=answer)
    
    return result


def _int_cotes_table(
    x: list[int | float],
    y: list[int | float],
    data: bool, 
    closed_methods: bool, 
    points: int,
    decimal_points: int
) -> tuple[DataFrame, DataFrame, Expr]:
    """
    Generate integration table of results for the Cotes' formulas.
    """
    h = x[1] - x[0]
    coeffs_dict = _coefficients(closed_methods=closed_methods)
    h_coef = coeffs_dict[points][0]
    eqtn_coeffs = coeffs_dict[points][1]
    result = (h_coef * h) * dot(y if closed_methods else y[1:-1], eqtn_coeffs)
    if closed_methods:
        coeffs = eqtn_coeffs
    else:
        coeffs = [nan] + eqtn_coeffs + [nan]
    N = vstack([x, coeffs, y, coeffs * y]).T
    col_labels = ['x', 'k', 'y', 'ky'] if data else ['x', 'k', 'f(x)', 'k f(x)']
    index_labels = dframe_labels(dframe=N, index=True)
    dframe = DataFrame(N, index=index_labels, columns=col_labels)
    answer = sympify(result)

    dframe_styled = sta_dframe_color(
        dframe=dframe, cols=[-1], decimal_points=decimal_points
    )

    return dframe, dframe_styled, answer


def _coefficients(closed_methods: bool = True) -> dict[float, list]:
    """
    Returns the coefficients for Newton-Cotes integration methods.

    Parameters
    ----------
    closed_methods : bool, optional (default=True)
        If `True`, closed methods coefficients are returned; otherwise, 
        open methods coefficients.

    Returns
    -------
    coeffs_dict : dict
        Dictionary containing the coefficients for the specified 
        integration methods.
    """
    if closed_methods:
        coeffs_dict = {
            2: (1 / 2, [1, 1]), # Trapezoidal's rule
            3: (1 / 3, [1, 4, 1]), # Simpson's 1/8 rule,
            4: (3 / 8, [1, 3, 3, 1]), # Simpson's 3/8 rule,
            5: (2 / 45, [7, 32, 12, 32, 7]), # Boole's rule
            6: (5 / 288, [19, 75, 50, 50, 75, 19]),
            7: (1 / 140, [41, 216, 27, 272, 27, 216, 41]),
            8: (7 / 17280, [751, 3577, 1323, 2989, 2989, 1323, 3577, 751]),
            9: (4 / 14175, [989, 5888, -928, 10496, -4540, 10496, -928, 5888, 989]),
            10: (9 / 89600, [2857, 15741, 1080, 19344, 5788, 5788, 19344, 1080, 15741, 2857]),
            11: (5 / 299376, [16067, 106300, -48525, 272400, -260550, 427368, -260550, 272400, -48525, 106300, 16067])
        }
    else:
        coeffs_dict = {
            1: (2, [1]), # Midpoint rule
            2: (3 / 2, [1, 1]),
            3: (4 / 3, [2, -1, 2]),
            4: (5 / 24, [11, 1, 1, 11]),
            5: (6 / 20, [11, -14, 26, -14, 11]),
            6: (7 / 1440, [611, -453, 562, 562, -453, 611]),
            7: (8 / 945, [460, -954, 2196, -2459, 2196, -954, 460])
        }

    return coeffs_dict


def int_composite(
    fexpr: str | Expr | Callable, 
    a: int | float, 
    b: int | float, 
    n: int | None = None, 
    h: float | None = None, 
    method: Literal['trapezoidal', 'simpson13', 'simpson38', 'boole', 'weddle']='trapezoidal',
    auto_display:bool = True,
    decimal_points: int = 12
) -> tuple[DataFrame, Float]:
    """
    Composite integration given a function.
    
    Parameters
    ----------
    f : {str, Expr, Callable}
        The univariate function to be integrated.
    a : {int, float}
        Lower limit of integration.
    b : {int, float}
        Upper limit of integration.
    n : int, optional (default=None)
        Number of points.
    h : {int, float}, optional (default=None)
        Stepsize (interval) if ``n=None``. 
    method: str, optional (default='trapezoidal')
        The integration method to be applied.
        ==========================================
        method          Description  
        ==========================================
        trapezoidal ... Composite Trapezoidal rule
        simpson13 ..... Composite Simpson 1/3 rule
        simpson38 ..... Composite Simpson 3/8 rule
        boole ......... Boole's rule
        weddle ........ Weddle's rule
        ==========================================
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
        The solution to the numerical integral.
    
    Examples
    --------
    >>> import stemlab as stm

    >>> f = 'sqrt(x^2 + log(x))'
    >>> a, b, n = (1, 1.5, 6)
    >>> result = stm.int_composite(f, a, b, n, method='trapezoidal',
    ... decimal_points=12)
         x            f(x)    k          k f(x)
    0  1.0  1.000000000000  1.0  1.000000000000
    1  1.1  1.142501719826  2.0  2.285003439651
    2  1.2  1.273703873274  2.0  2.547407746549
    3  1.3  1.397270290412  2.0  2.794540580824
    4  1.4  1.515411573343  2.0  3.030823146686
    5  1.5  1.629559789670  1.0  1.629559789670

    Answer = 0.664366735169

    >>> f = '(2 * x) / (x^2 - 4)'
    >>> a, b, n = (1, 1.6, 8)
    >>> result = stm.int_composite(f, a, b, n, method='simpson13',
    ... decimal_points=12)
                    x            f(x)    k          k f(x)
    0  1.000000000000 -0.666666666667  1.0 -0.666666666667
    1  1.085714285714 -0.769675925926  4.0 -3.078703703704
    2  1.171428571429 -0.891581236409  2.0 -1.783162472818
    3  1.257142857143 -1.039136302294  4.0 -4.156545209177
    4  1.342857142857 -1.222593831289  2.0 -2.445187662579
    5  1.428571428571 -1.458333333333  4.0 -5.833333333333
    6  1.514285714286 -1.774270683883  2.0 -3.548541367767
    7  1.600000000000 -2.222222222222  1.0 -2.222222222222

    Answer = -0.678124646808

    >>> f = 'x^2 * exp(-x)'
    >>> a, b, n = (0, 1, 8)
    >>> result = stm.int_composite(f, a, b, n, method='simpson38',
    ... decimal_points=12)
                    x            f(x)    k          k f(x)
    0  0.000000000000  0.000000000000  1.0  0.000000000000
    1  0.142857142857  0.017691385709  3.0  0.053074157128
    2  0.285714285714  0.061345085149  3.0  0.184035255447
    3  0.428571428571  0.119652071791  2.0  0.239304143583
    4  0.571428571429  0.184397754125  3.0  0.553193262375
    5  0.714285714286  0.249766152835  3.0  0.749298458506
    6  0.857142857143  0.311784131518  2.0  0.623568263036
    7  1.000000000000  0.367879441171  1.0  0.367879441171

    Answer = 0.148411766852

    >>> f = 'exp(3 * x) * sin(2 * x)'
    >>> a, b, h = (0, 1.6, .25)
    >>> result = stm.int_composite(f, a, b, h=h, method='boole',
    ... decimal_points=12)
                    x             f(x)     k              k f(x)
    0  0.000000000000   0.000000000000   7.0    0.00000000000000
    1  0.228571428571   0.876235068555  32.0   28.03952219377000
    2  0.457142857143   3.121753295728  12.0   37.46103954873900
    3  0.685714285714   7.668615099314  32.0  245.39568317806300
    4  0.914285714286  15.018136073809  14.0  210.25390503332000
    5  1.142857142857  23.283119583428  32.0  745.05982666970203
    6  1.371428571429  23.764397334114  12.0  285.17276800937202
    7  1.600000000000  -7.093066540184   7.0  -49.65146578128600

    Answer = 16.685903098352

    >>> f = 'x * log(x + 1)'
    >>> a, b, n = (0, .35, 11)
    >>> result = stm.int_composite(f, a, b, n, method='weddle',
    ... decimal_points=12)
            x            f(x)    k          k f(x)
    0   0.000  0.000000000000  1.0  0.000000000000
    1   0.035  0.001204049935  5.0  0.006020249676
    2   0.070  0.004736105393  1.0  0.004736105393
    3   0.105  0.010483760172  6.0  0.062902561031
    4   0.140  0.018343956737  1.0  0.018343956737
    5   0.175  0.028221925829  5.0  0.141109629147
    6   0.210  0.040030275518  1.0  0.040030275518
    7   0.245  0.053688204830  5.0  0.268441024148
    8   0.280  0.069120821821  1.0  0.069120821821
    9   0.315  0.086258549673  6.0  0.517551298040
    10  0.350  0.105036607358  1.0  0.105036607358
    
    0.012949571553
    """
    f = sym_lambdify_expr(fexpr=fexpr, is_univariate=True, par_name='fexpr')
    a = ValidateArgs.check_numeric(user_input=a, to_float=True, par_name='a')
    b = ValidateArgs.check_numeric(user_input=b, to_float=True, par_name='b')
    if a >= b:
        raise LowerGteUpperError(
            par_name='Limits', 
            lower_par_name='a', 
            upper_par_name='b', 
            user_input=[a, b]
        )
    
    n = 8 if n is None and h is None else n
    if n is not None and h is not None:
        raise ValueError("Provide 'n' or 'h', not both")
    if n is None:
        h = ValidateArgs.check_numeric(user_input=h, to_float=True, par_name='h')
        if h > abs(b - a):
            raise IntervalError(par_name='h', gte=True)
        n = int(ceil((b - a) / h)) + 1
    else:
        n = ValidateArgs.check_numeric(user_input=n, is_integer=True, par_name='n')
        h = float((b - a) / (n - 1))
    methods = ['trapezoidal', 'simpson13', 'simpson38', 'boole', 'weddle']
    method = ValidateArgs.check_member(
        par_name='method', valid_items=methods, user_input=method
    )
    auto_display = ValidateArgs.check_boolean(user_input=auto_display, default=True)
    decimal_points = ValidateArgs.check_decimals(x=decimal_points)

    integration_functions = {
        'trapezoidal': _trapezoidal,
        'simpson13': _simpson13,
        'simpson38': _simpson38,
        'boole': _boole,
        'weddle': _weddle
    }
    N, result = integration_functions[method](
        f=f, a=a, b=b, n=n, h=h, x=None, y=None, data=False
    )

    col_labels = dframe_labels(
        dframe=N, df_labels=['x', 'f(x)', 'k', 'k f(x)'], prefix=None, index=False
    )
    index_labels = dframe_labels(dframe=N, index=True)
    dframe = DataFrame(N, index=index_labels, columns=col_labels)
    
    # css style
    dframe_styled = sta_dframe_color(
        dframe=dframe,
        cols=[-1],
        decimal_points=decimal_points,
    )
    answer = sympify(result)

    if auto_display:
        display_results({
            'dframe': dframe_styled,
            'Answer': answer,
            'decimal_points': decimal_points
        })

    result = Result(table=dframe, table_styled=dframe_styled, answer=answer)
    
    return result


def int_composite_data(
    x, 
    y, 
    method: Literal[
        'trapezoidal', 'simpson13', 'simpson38', 'boole', 'weddle'
    ]='trapezoidal', 
    auto_display: bool = True, 
    decimal_points: int = 12
) -> tuple[DataFrame, Float]:
    """
    Composite integration given data.
    
    Parameters
    ----------
    x : array_like
        The points corresponding to the ``y`` values.
    y : array_like
        The points to be integrated, synonymous to f(x).
    method: {trapezoidal, ..., weddle}, optional (default='trapezoidal')
        The integration method to be applied.
        ==========================================
        Argument        Description  
        ==========================================
        trapezoidal     Composite Trapezoidal rule
        simpson13       Composite Simpson 1/3 rule
        simpson38       Composite Simpson 3/8 rule
        boole           Boole's rule
        weddle          Weddle's rule
        ==========================================
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
        The value of the numerical integral.
        
    Examples
    --------
    >>> x = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    >>> y = [1.0, 1.14250172, 1.27370387, 1.39727029, 1.51541157,
    ... 1.62955979]
    >>> result = stm.int_composite_data(x, y, method='trapezoidal',
    ... decimal_points=12)
         x           y    k          ky
    0  1.0  1.00000000  1.0  1.00000000
    1  1.1  1.14250172  2.0  2.28500344
    2  1.2  1.27370387  2.0  2.54740774
    3  1.3  1.39727029  2.0  2.79454058
    4  1.4  1.51541157  2.0  3.03082314
    5  1.5  1.62955979  1.0  1.62955979
    
    Answer = 0.6643667345

    >>> x = [1.0, 1.075, 1.15, 1.225, 1.3, 1.375, 1.45, 1.525, 1.6]
    >>> y = [-0.66666667, -0.75587783, -0.85901027, -0.98024506,
    ... -1.12554113, -1.3037037, -1.52832675, -1.82157521, -2.22222222]
    >>> result = stm.int_composite_data(x, y, method='simpson13',
    ... decimal_points=12)
           x           y    k          ky
    0  1.000 -0.66666667  1.0 -0.66666667
    1  1.075 -0.75587783  4.0 -3.02351132
    2  1.150 -0.85901027  2.0 -1.71802054
    3  1.225 -0.98024506  4.0 -3.92098024
    4  1.300 -1.12554113  2.0 -2.25108226
    5  1.375 -1.30370370  4.0 -5.21481480
    6  1.450 -1.52832675  2.0 -3.05665350
    7  1.525 -1.82157521  4.0 -7.28630084
    8  1.600 -2.22222222  1.0 -2.22222222

    Answer = 0.73400630975

    >>> df = stm.dataset_read(name='simpson38')
    >>> x = df['x'].values
    >>> y = df['y'].values
    >>> result = stm.int_composite_data(x, y, method='simpson38',
    ... decimal_points=12)
           x           y    k          ky
    0  0.000  0.00000000  1.0  0.00000000
    1  0.125  0.01378901  3.0  0.04136703
    2  0.250  0.04867505  3.0  0.14602515
    3  0.375  0.09665005  2.0  0.19330010
    4  0.500  0.15163266  3.0  0.45489798
    5  0.625  0.20908650  3.0  0.62725950
    6  0.750  0.26570619  2.0  0.53141238
    7  0.875  0.31915998  3.0  0.95747994
    8  1.000  0.36787944  1.0  0.36787944

    Answer = 0.15560725875

    >>> df = stm.dataset_read(name='boole')
    >>> x = df['x'].values
    >>> y = df['y'].values
    >>> result = stm.int_composite_data(x, y, method='boole',
    ... decimal_points=8)
         x            y     k                  ky
    0  0.0   0.00000000   7.0    0.00000000000000
    1  0.2   0.70956648  32.0   22.70612736000000
    2  0.4   2.38170610  12.0   28.58047320000000
    3  0.6   5.63850789  32.0  180.43225247999999
    4  0.8  11.01847613  14.0  154.25866582000000
    5  1.0  18.26372704  32.0  584.43926527999997
    6  1.2  24.72075984  12.0  296.64911807999999
    7  1.4  22.33913068  32.0  714.85218176000001
    8  1.6  -7.09306654   7.0  -49.65146578000000
    
    Answer = 17.175703272889

    >>> df = stm.dataset_read(name='weddle')
    >>> x = df['x'].values
    >>> y = df['y'].values
    >>> result = stm.int_composite_data(x, y, method='weddle',
    ... decimal_points=8)
            x           y    k          ky
    0   0.000  0.00000000  1.0  0.00000000
    1   0.035  0.00120405  5.0  0.00602025
    2   0.070  0.00473611  1.0  0.00473611
    3   0.105  0.01048376  6.0  0.06290256
    4   0.140  0.01834396  1.0  0.01834396
    5   0.175  0.02822193  5.0  0.14110965
    6   0.210  0.04003028  1.0  0.04003028
    7   0.245  0.05368820  5.0  0.26844100
    8   0.280  0.06912082  1.0  0.06912082
    9   0.315  0.08625855  6.0  0.51755130
    10  0.350  0.10503661  1.0  0.10503661
    
    Answer = 0.01294957167
    """
    x = conv_to_arraylike(array_values=x, to_ndarray=True, par_name='x')
    y = conv_to_arraylike(array_values=y, to_ndarray=True, par_name='y')
    _ = ValidateArgs.check_len_equal(x=x, y=y, par_name=['x', 'y'])
    _ = ValidateArgs.check_diff_constant(user_input=x, par_name='x')
    methods = ['trapezoidal', 'simpson13', 'simpson38', 'boole', 'weddle']
    method = ValidateArgs.check_member(
        par_name='method', valid_items=methods, user_input=method
    )
    n, h = len(x), x[1] - x[0]
    auto_display = ValidateArgs.check_boolean(user_input=auto_display, default=True)
    decimal_points = ValidateArgs.check_decimals(x=decimal_points)

    integration_functions = {
        'trapezoidal': _trapezoidal,
        'simpson13': _simpson13,
        'simpson38': _simpson38,
        'boole': _boole,
        'weddle': _weddle
    }
    N, result = integration_functions[method](
        f=None, a=None, b=None, n=n, h=h, x=x, y=y, data=True
    )
    
    col_labels = dframe_labels(
        dframe=N, df_labels=['x', 'y', 'k', 'ky'], prefix=None, index=False
    )
    index_labels = dframe_labels(dframe=N, index=True)
    dframe = DataFrame(N, index=index_labels, columns=col_labels)
    
    # css style
    dframe_styled = sta_dframe_color(
        dframe=dframe,
        cols=[-1],
        decimal_points=decimal_points,
    )
    answer = sympify(result)
    
    if auto_display:
        display_results({
            'dframe': dframe_styled,
            'Answer': answer,
            'decimal_points': decimal_points
        })
    
    result = Result(table=dframe, table_styled=dframe_styled, answer=answer)
    
    return result


def _trapezoidal(f, a, b, n, h, x, y, data):
    """
    Trapezoidal's rule integration given a function and limits of 
    integration or sample data points.
    """
    if not data:
        _ = ValidateArgs.check_divisibility(n=n, divisor=2, method="Trapezoidal's")
        x = linspace(a, b, n)
        y = f(x)
    k_values = array([1] + [2] * (n - 2) + [1])
    k_values_y = k_values * y
    N = column_stack((x, y, k_values, k_values_y))
    result = (1 / 2) * h * sum(k_values_y)
    
    return N, result


def _simpson13(f, a, b, n, h, x=None, y=None, data=False):
    """
    Simpson's 1/3 rule integration given a function and limits of 
    integration or sample data points.
    """
    if not data:
        _ = ValidateArgs.check_divisibility(n=n, divisor=2, method="Simpson's 1/3")
        x = linspace(a, b, n)
        y = f(x)
    k_values = where(arange(n) % 2 == 0, 2, 4)
    k_values[[0, -1]] = 1
    k_values_y = k_values * y
    N = column_stack((x, y, k_values, k_values_y))
    result = (h / 3) * sum(k_values_y)

    return N, result


def _simpson38(f, a, b, n, h, x=None, y=None, data=False):
    """
    Simpson's 3/8 rule integration given a function and limits of 
    integration or sample data points.
    """
    if not data:
        _ = ValidateArgs.check_divisibility(n=n, divisor=2, method="Simpson's 3/8")
        x = linspace(a, b, n)
        y = f(x)
    k_values = where(arange(n) % 3 == 0, 2, 3)
    k_values[[0, -1]] = 1
    k_values_y = k_values * y
    N = column_stack((x, y, k_values, k_values_y))
    result = (3 / 8) * h * sum(k_values_y)

    return N, result


def _boole(f, a, b, n, h, x=None, y=None, data=False):
    """
    Boole's rule integration given a function and limits of 
    integration or sample data points.
    """
    if not data:
        _ = ValidateArgs.check_divisibility(n=n, divisor=4, method="Boole's")
        x = linspace(a, b, n)
        y = f(x)
    m_values = arange(n)
    k_values = where(m_values % 2 == 1, 32, where(m_values % 4 == 0, 14, 12))
    k_values[[0, -1]] = 7
    k_values_y = k_values * y
    N = column_stack((x, y, k_values, k_values_y))
    result = (2 / 45) * h * sum(k_values_y)

    return N, result


def _weddle(f, a, b, n, h, x=None, y=None, data=False):
    """
    Weddle's rule integration given a function and limits of 
    integration or sample data points.
    """
    if n > 13:
        raise ValueError(f"The 'Weddle' method is only available for n <= 13.")
    if not data:
        x = linspace(a, b, n)
        y = f(x)
    k_values = where(
        isin(arange(n), [2, 4, 6, 8, 10, 12]), 1,
        where(
            isin(arange(n), [1, 5, 7, 11]), 5,
            where(
                isin(arange(n), [3, 9]), 6, 0
            )
        )
    )
    k_values[[0, -1]] = 1
    k_values_y = k_values * y
    N = column_stack((x, y, k_values, k_values_y))
    result = (3 / 10 * h) * sum(k_values_y)

    return N, result


def int_romberg(
    fexpr: str | Expr | Callable,
    a: float | int, 
    b: float | int, 
    n: int = 5,
    auto_display: bool = True,
    decimal_points: int = 12
) -> tuple[DataFrame, Float]:
    """
    Romberg integration.

    Parameters
    ----------
    fexpr : {str, sympy.Expr, callable}
        The function to be integrated.
    a : {int, float}
        Lower limit of integration.
    b : {int, float}
        Upper limit of integration.
    n : int, optional (default=5)
        Number of points / nodes.
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
        Solution of the numerical integral.

    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm

    >>> f = '(x^2 + 2 * x + 1) / (1 + (x + 1)^4)'
    >>> a, b, n = (0, 2, 4)
    >>> result = stm.int_romberg(f, a, b, n, decimal_points=10)
       k            C1            C2            C3            C4
    0  0  0.6097560976                                          
    1                   0.5169775227                            
    2  1  0.5401721664                0.5324513432              
    3                   0.5314842294                0.5344938032
    4  2  0.5336562136                0.5344618898              
    5                    0.534275786                            
    6  3  0.5341208929                                          
    
    Answer = 0.5344938032

    >>> f = 'exp(-x) * cos(x)'
    >>> a, b, n = (0, np.pi/2, 6)
    >>> result = stm.int_romberg(f, a, b, n, decimal_points=8)
        k          C1          C2          C3          C4          C5          C6
    0   0  0.78539816                                                            
    1                  0.59941268                                                
    2   1  0.64590905              0.60390974                                    
    3                  0.60362868              0.60393985                        
    4   2  0.61419877              0.60393938              0.60393979            
    5                  0.60391996              0.60393979              0.60393979
    6   3  0.60648966              0.60393978              0.60393979            
    7                  0.60393854              0.60393979                        
    8   4  0.60457632              0.60393979                                    
    9                  0.60393971                                                
    10  5  0.60409886                                                            
    
    Answer = 0.60393979

    >>> f = 'exp(3 * x) * sin(2 * x)'
    >>> a, b, n = (1, 3, 5)
    >>> result = stm.int_romberg(f, a, b, n, decimal_points=7)
       k            C1            C2            C3            C4            C5
    0  0 -2245.8635055                                                        
    1                  -1155.7090586                                          
    2  1 -1428.2476704               -1654.6479314                            
    3                  -1623.4642518               -1724.4950439              
    4  2 -1574.6601065               -1723.4036827               -1724.9722973
    5                  -1717.1574683               -1724.9704331              
    6  3 -1681.5331278               -1724.9459526                            
    7                  -1724.4591723                                          
    8  4 -1713.7276612                                                        
                                                           
    Answer = -1724.9722973
    """
    f = sym_lambdify_expr(fexpr=fexpr, is_univariate=True, par_name='fexpr')
    a = ValidateArgs.check_numeric(user_input=a, par_name='a')
    b = ValidateArgs.check_numeric(user_input=b, par_name='b')
    if a >= b:
        raise LowerGteUpperError(
            par_name='Limits', 
            lower_par_name='a', 
            upper_par_name='b', 
            user_input=[a, b]
        )
    n = ValidateArgs.check_numeric(
        user_input=n, is_positive=True, is_integer=True, par_name='n'
    )
    auto_display = ValidateArgs.check_boolean(user_input=auto_display, default=True)
    decimal_points = ValidateArgs.check_decimals(x=decimal_points)
    # initialize results table
    N = nan * ones((n, n))
    steps = []
    for k in range(n):
        # composite trapezoidal rule for 2^k panels
        steps.append(2 ** k)
        N[k, 0] = _trapezoidal_romberg(f, a, b, 2 ** k)
        # romberg recursive formula
        for j in range(k):
            N[k, j + 1] = (4 ** (j + 1) * N[k, j] - N[k - 1, j]) / (4 ** (j + 1) - 1)
    nrows, ncols = N.shape
    col_names = ['C%s' %(k + 1) for k in range(ncols)]
    row_names = arange(nrows)
    n = len(row_names)
    row_names = list(repeat(row_names, 2)[:-1])
    row_names[1::2] = [''] * (n - 1)
    answer = sympify(N[-1, -1]) # should be here (i.e. before converting df)
    N = arr_table_blank_row(data=N, to_ndarray=True)
    N = hstack([matrix(row_names).T, N])
    dframe = DataFrame(N)
    dframe.columns = ['k'] + col_names
    
    # css style
    dframe_styled = sta_dframe_color(
        dframe=dframe,
        style_indices=[[n - 1, -1]],
        decimal_points=decimal_points,
    )
    
    if auto_display:
        display_results({
            'dframe': dframe_styled,
            'Answer': answer,
            'decimal_points': decimal_points
        })
    
    result = Result(table=dframe, table_styled=dframe_styled, answer=answer)
    
    return result


def _trapezoidal_romberg(f, a, b, n):
    """
    Trapezoidal integration, used in Romberg integration.
    """
    h = (b - a) / n
    x = linspace(a, b, num=n + 1, endpoint=True)
    result = 0.5 * h * (f(a) + 2 * sum(f(x)[1:-1]) + f(b))

    return result

        
def int_gauss_legendre(
    fexpr: str | Expr | Callable,
    a: float | int,
    b: float | int,
    n: int = 5,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result | tuple[DataFrame, Float]:
    """
    Gauss-Legendre integration.

    Parameters
    ----------
    fexpr : {str, sympy.Expr, callable}
        The function to be integrated.
    a : {int, float}
        Lower limit of integration.
    b : {int, float}
        Upper limit of integration.
    n : int, optional (default=5)
        Number of points / nodes.
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
        The result of the numerical integral.

    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm

    >>> f = '(x^2 + 2 * x + 1) / (1 + (x + 1)^4)'
    >>> a, b, n = (0, 2, 4)
    >>> result = stm.int_gauss_legendre(f, a, b, n, decimal_points=14)
              Node (x)       Weight (w)             f(t)        w * f(t)
    0 -0.86113631159405  0.34785484513745  0.48355565474039  0.16820717739506
    1 -0.33998104358486  0.65214515486255  0.32066163832635  0.20911793378481
    2  0.33998104358486  0.65214515486255  0.17673644253964  0.11525781468987
    3  0.86113631159405  0.34785484513745  0.12036215617885  0.04186855919800
    
    Answer = 0.53445148506774

    >>> f = 'exp(-x) * cos(x)'
    >>> a, b, n = (0, np.pi/2, 6)
    >>> result = stm.int_gauss_legendre(f, a, b, n, decimal_points=14)
               Node (x)        Weight (w)              f(t)          w * f(t)
    0 -0.93246951420315  0.17132449237917  0.94701010896676  0.16224602619667
    1 -0.66120938646626  0.36076157304814  0.73940307222081  0.26674821545101
    2 -0.23861918608320  0.46791393457269  0.45449056263767  0.21266246738995
    3  0.23861918608320  0.46791393457269  0.21281747671385  0.09958026287501
    4  0.66120938646626  0.36076157304814  0.07132727228190  0.02573213894965
    5  0.93246951420315  0.17132449237917  0.01162069740130  0.00199091008337
    
    Answer = 0.60393978817679

    >>> f = 'exp(3 * x) * sin(2 * x)'
    >>> a, b, n = (1, 3, 15)
    >>> result = stm.int_gauss_legendre(f, a, b, n, decimal_points=12)
              Node (x)      Weight (w)                 f(t)            w * f(t)
    0  -0.987992518020  0.030753241996    18.72009251746500    0.57570353537900
    1  -0.937273392401  0.070366047488    20.60958805522900    1.45021525180500
    2  -0.848206583410  0.107159220467    23.54089125568500    2.52262355606200
    3  -0.724417731360  0.139570677926    25.56049613570100    3.56749577378900
    4  -0.570972172609  0.166269205817    20.35315350236700    3.38410266871000
    5  -0.394151347078  0.186161000016    -8.66212949961100   -1.61255068991200
    6  -0.201194093997  0.198431485327   -97.15376009978399  -19.27836492171400
    7   0.000000000000  0.202578241926  -305.31591759436901  -61.85036181815700
    8   0.201194093997  0.198431485327  -702.56794201301295 -139.41160027685399
    9   0.394151347078  0.186161000016 -1312.34359732972507 -244.30719644292199
    10  0.570972172609  0.166269205817 -2033.82279747745997 -338.16210130907501
    11  0.724417731360  0.139570677926 -2626.24295254265417 -366.54650928516298
    12  0.848206583410  0.107159220467 -2845.35777376225315 -304.90632098656999
    13  0.937273392401  0.070366047488 -2667.53038244886602 -187.70356956736600
    14  0.987992518020  0.030753241996 -2363.60603888629794  -72.68854849735700
    
    Answer = -1724.96698300934
    """
    f = sym_lambdify_expr(fexpr=fexpr, is_univariate=True, par_name='fexpr')
    a = ValidateArgs.check_numeric(user_input=a, par_name='a')
    b = ValidateArgs.check_numeric(user_input=b, par_name='b')
    if a >= b:
        raise LowerGteUpperError(
            par_name="'a', 'b'", 
            lower_par_name='a', 
            upper_par_name='b', 
            user_input=[a, b]
        )
    n = ValidateArgs.check_numeric(
        user_input=n, limits=[1, max_rows()], is_integer=True, par_name='n'
    )
    auto_display = ValidateArgs.check_boolean(user_input=auto_display, default=True)
    x, w = roots_legendre(n)
    t = 1/2 * (a + b) + 1/2 * (b - a) * x # transformation
    dt = 1/2 * (b - a) # derivative of transformed variable
    answer = sympify(dt * sum(w * f(t)))
    # table of weights
    dframe = DataFrame([x, w, f(t), w * f(t)]).T
    dframe.columns = ['Node (x)', 'Weight (w)', 'f(t)', 'w * f(t)']

    # css style
    dframe_styled = sta_dframe_color(
        dframe=dframe, cols=[-1], decimal_points=decimal_points,
    )
        
    if auto_display:
        display_results({
            'dframe': dframe_styled,
            'Answer': answer,
            'decimal_points': decimal_points
        })

    result = Result(table=dframe, table_styled=dframe_styled, answer=answer)
    
    return result