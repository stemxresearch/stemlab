from typing import Callable, Literal

from numpy import sqrt, nan, isnan, zeros, dot, asfarray
from pandas import DataFrame
from sympy import sympify, Expr
from stemlab.core.symbolic import sym_lambdify_expr
from stemlab.core.arraylike import conv_to_arraylike, is_iterable
from stemlab.statistical.wrangle import dframe_labels
from stemlab.core.htmlatex import sta_dframe_color
from stemlab.core.display import Result, display_results
from stemlab.mathematics.linearalgebra.linearsystems import (
    la_gauss_pivot, la_jacobian
)
from stemlab.core.base.strings import str_info_messages
from stemlab.core.validators.errors import (
    NoRootInIntervalError, LowerGteUpperError, RequiredError
)
from stemlab.core.validators.validate import ValidateArgs


NONLINEAR_METHODS = [
    'bisection', 'regula-falsi', 'mregula-falsi', 'secant', 
    'fixed-point', 'newton-raphson', 'mnewton-raphson', 'aitken', 
    'steffensen', 'system'
]


def nle_roots(
    method: Literal[
        'bisection', 'regula-falsi', 'mregula-falsi', 'secant', 
        'fixed-point', 'newton-raphson', 'mnewton-raphson', 'aitken', 
        'steffensen', 'system'
    ],
    fexpr: str | Expr | Callable,
    initial_guess: int | float | list[float, int] | None = None,
    tolerance: float = 1e-6,
    maxit: int = 10,
    stop_crit: Literal['relative', 'absolute'] = 'relative',
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Finds roots of nonlinear equations using various numerical methods.

    Parameters
    ----------
    method : str
        The nonlinear method to be used to calculate the root of the 
        univariate function.
        =========================================================
        method                      Description  
        =========================================================
        bisection ................. Bisection
        regula-falsi .............. Regula-Falsi / False position
        mregula-falsi ............. Modified Regula-Falsi 
        secant .................... Secant
        fixed-point ............... Fixed point iteration
        newton-raphson ............ Newton-Raphson
        mnewton-raphson ........... Modified Newton-Raphson
        aitken .................... Aitken
        steffensen ................ Steffensen
        system .................... System of nonlinear equations 
                                    using Newton-Rapshson method
        =========================================================
    fexpr : {str, sympy.Expr, Callable}
        The univariate expression representing the function whose 
        roots are to be found.
    initial_guess : {int, float, list-like}
        The starting value as integer/float for open methods or 
        a list of two elements [a, b] for closed methods. For 'system' 
        of equations method, it should be a list of n initial guesses.
    tolerance : float, optional (default=1e-6)
        Tolerance for stopping criterion.
    maxit : int, optional (default=10)
        Maximum number of iterations.
    stop_crit : str, optional (default='relative')
        Stopping criterion.
        ========================================================
        stop_crit                   Description  
        ========================================================
        absolute .................. Calculate the absolute error
        relative .................. Calculate the relative error
        ========================================================
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with root(s) highlighted.
    answer : {float, array-like}
        The solution (root) of the nonlinear equation or system of equations.
    msg : str
        A string with information on convergence.

    Raises:
    -------
    ValueError:
        If the input method is not a string.
    NoRootInIntervalError:
        If the interval does not contain a root for methods like 
        'bisection', 'regula-falsi', and 'modified regula-falsi'

    Examples
    --------
    import numpy as np
    import stemlab as stm

    ### Bisection method
    
    >>> f = '3 ** (3*x + 1) - 7.5**(2*x)'
    >>> a, b = (1, 2)
    >>> result = stm.nle_roots(method='bisection', fexpr=f,
    ... initial_guess=[a, b], tolerance=1e-6, maxit=15,
    ... decimal_points=10)
             $a_n$         $b_n$         $p_n$       $f(p_n)$  Relative error
    0   1.00000000  2.0000000000  1.5000000000  -0.9866537608    0.3333333333
    1   1.00000000  1.5000000000  1.2500000000  30.5931023150    0.2000000000
    2   1.25000000  1.5000000000  1.3750000000  23.8417046249    0.0909090909
    3   1.37500000  1.5000000000  1.4375000000  14.5912606133    0.0434782609
    4   1.43750000  1.5000000000  1.4687500000   7.7399012107    0.0212765957
    5   1.46875000  1.5000000000  1.4843750000   3.6319491455    0.0105263158
    6   1.48437500  1.5000000000  1.4921875000   1.3892750639    0.0052356021
    7   1.49218750  1.5000000000  1.4960937500   0.2183288849    0.0026109661
    8   1.49609375  1.5000000000  1.4980468750  -0.3798619425    0.0013037810
    9   1.49609375  1.4980468750  1.4970703125  -0.0796971815    0.0006523157
    10  1.49609375  1.4970703125  1.4965820312   0.0695824695    0.0003262643
    
    Answer = 1.4965820313

    ### Newton-Raphson method
    
    >>> f = '3 ** (3*x + 1) - 7.5**(2*x)'
    >>> result = stm.nle_roots(method='newton-raphson', fexpr=f,
    ... initial_guess=1.25, tolerance=1e-12, maxit=20,
    ... decimal_points=14)
                   $p_n$    Relative error
    0   1.25000000000000  0.66669188030705
    1   3.75028367491173  0.06773235047431
    2   3.51238179984504  0.07186672439981
    3   3.27688295558555  0.07634631035037
    4   3.04445040046532  0.08113233632514
    5   2.81598311157139  0.08610803875448
    6   2.59272835766937  0.09100250152036
    7   2.37646417314010  0.09524395984151
    8   2.16980349609415  0.09768487161101
    9   1.97670893733795  0.09613684032277
    10  1.80334139372224  0.08685708486551
    11  1.65922587139908  0.06533623293141
    12  1.55746685422827  0.03291034460915
    13  1.50784321442497  0.00708286465792
    14  1.49723847693222  0.00028591328887
    15  1.49681051891394  0.00000044888384
    16  1.49680984702018  0.00000000000110
    17  1.49680984701853  0.00000000000000
    
    Answer = 1.49680984701853
    
    'The tolerance (1e-12) was achieved before reaching the maximum number of iterations (20).'

    ### Secant method

    >>> f = '3 ** (3*x + 1) - 7.5**(2*x)'
    >>> a, b = (1.25, 1.5)
    >>> result = stm.nle_roots(method='secant', fexpr=f,
    ... initial_guess=[a, b], tolerance=1e-12, maxit=10,
    ... decimal_points=14)
               $ p_{n}$    Relative error
    0  1.25000000000000               NaN
    1  1.50000000000000               NaN
    2  1.49218919108815  0.00523446286738
    3  1.49675571719038  0.00305094949683
    4  1.49681077100894  0.00003678074719
    5  1.49680984683531  0.00000061742888
    6  1.49680984701853  0.00000000012240
    7  1.49680984701853  0.00000000000000
    
    Answer = 1.49680984701853
    
    'The tolerance (1e-12) was achieved before reaching the maximum number of iterations (10).'
    
    ### Systems of non-linear equations
    
    >>> def f(x):
            f = np.zeros(len(x))
            f[0] = np.sin(x[0]) + x[1] ** 2 + np.log(x[2]) - 7.0
            f[1] = 3.0 * x[0] + 2.0 ** x[1] - x[2] ** 3 + 1.0
            f[2] = x[0] + x[1] + x[2] - 5.0
            return f
    >>> x0 = [1, 1, 1]
    >>> result = stm.nle_roots(method='system', fexpr=f, initial_guess=x0,
    ... tolerance=1e-12, maxit=10, decimal_points=14)
                  $x_1$             $x_2$             $x_3$
    0  1.00000000000000  1.00000000000000  1.00000000000000
    1 -0.60349871636056  3.42140435463437  2.18209437388113
    2  0.46943641867960  2.58963283147861  1.94093074984178
    3  0.59282350783269  2.40348170271811  2.00369478944920
    4  0.59904612622781  2.39594264538855  2.00501122838364
    5  0.59905375661788  2.39593140240478  2.00501484097734
    6  0.59905375664057  2.39593140237782  2.00501484098162
    
    Answer = array([0.59905375664057, 2.39593140237782, 2.00501484098162])
    
    'The tolerance (1e-12) was achieved before reaching the maximum number of iterations (10).'

    >>> def f(x):
            f = np.zeros(len(x))
            f[0] = 3 * x[0] - np.cos(x[1] * x[2]) - 3/2
            f[1] = 4 * x[0] ** 2 - 625 * x[1] ** 2 + 2 * x[2] - 1
            f[2] = 20 * x[2] + np.exp(-x[0] * x[1]) + 9
            return f
    >>> x0 = [1, 1, 1]
    >>> result = stm.nle_roots(method='system', fexpr=f, initial_guess=x0,
    ... tolerance=1e-12, maxit=10, decimal_points=14)
                  $x_1$             $x_2$             $x_3$
    0  1.00000000000000  1.00000000000000  1.00000000000000
    1  1.23270065281820  0.50313207294959 -0.47325306848465
    2  0.83259228137685  0.25180648799324 -0.49063625130746
    3  0.83323760604989  0.12840590942084 -0.49470224084709
    4  0.83327539235728  0.06908184260902 -0.49714716661081
    5  0.83328101185556  0.04358546684408 -0.49820589845546
    6  0.83328158226316  0.03611676322137 -0.49851671038037
    7  0.83328161354326  0.03534310097445 -0.49854892447751
    8  0.83328161381671  0.03533461716109 -0.49854927776844
    9  0.83328161381676  0.03533461613949 -0.49854927781104
    
    Answer = array([ 0.83328161381676,  0.03533461613949 , -0.49854927781104])
    
    'The maximum number of iterations (10) was reached before achieving the tolerance (1e-12).'
    
    >>> f = ['sin(x1) + x2 ** 2 + log(x3) - 7', '3*x1 + 2 ** x2 - x3 ** 3 + 1', 
    ... 'x1 + x2 + x3 - 5']
    >>> x0 = [1, 1, 1]
    >>> result = stm.nle_roots(method='system', fexpr=f, initial_guess=x0,
    ... tolerance=1e-12, maxit=10, decimal_points=14)
                  $x_1$             $x_2$             $x_3$
    0  1.00000000000000  1.00000000000000  1.00000000000000
    1  3.42140435463437  2.18209437388113 -0.60349871636056
    2  2.58963280685095  1.94093074849202  0.46943644465703
    3  2.40348170151075  2.00369479050629  0.59282350798296
    4  2.39594264524909  2.00501122837745  0.59904612637346
    5  2.39593140240470  2.00501484097743  0.59905375661787
    6  2.39593140237782  2.00501484098162  0.59905375664057
    
    Answer = array([39593140237782, 2.00501484098162, 0.59905375664057])
    
    'The tolerance (1e-12) was achieved before reaching the maximum number of iterations (10).'
    """
    msg, result_table = [[]] * 2
    
    # method -> will take care of `false-position` entry
    method = 'regula-falsi' if 'fals' in method else method

    # just incase there is a typo, correct it
    method = method\
        .replace('regular', 'regula')\
        .replace('false', 'falsi')\
        .replace('systems', 'system')
    
    method = ValidateArgs.check_member(
        par_name='method', valid_items=NONLINEAR_METHODS, user_input=method
    )

    closed_methods = NONLINEAR_METHODS[:4]
    open_methods = NONLINEAR_METHODS[4:-1]
    system = NONLINEAR_METHODS[-1]
    
    # function
    if method in system:
        f = sym_lambdify_expr(fexpr=fexpr)
    else:
        f = sym_lambdify_expr(fexpr=fexpr, is_univariate=True, par_name='fexpr')
        if 'newton' in method:
            f_sym = sympify(fexpr)
            d1f = sym_lambdify_expr(
                fexpr=f_sym.diff(), is_univariate=True, par_name='fexpr.diff()'
            )
        if 'mnewton' in method:
            d2f = sym_lambdify_expr(
                fexpr=f_sym.diff().diff(), 
                is_univariate=True, 
                par_name='fexpr.diff().diff()'
            )

    # initial_guess
    if method in open_methods:
        if is_iterable(array_like=initial_guess):
            try:
                if len(initial_guess) == 1:
                    initial_guess = initial_guess[0]
                else:
                    raise ValueError(
                        f"Expected 'initial guess' to be float or integer "
                        f"but got {initial_guess}"
                    )
            except Exception as e:
                raise e
                
        x0 = ValidateArgs.check_numeric(
            par_name='initial_guess', to_float=True, user_input=initial_guess
        )
    elif method in closed_methods:
        initial_guess = conv_to_arraylike(
            array_values=initial_guess, n=2, par_name='initial_guess'
        )
        a, b = initial_guess
        a = ValidateArgs.check_numeric(user_input=a, to_float=True, par_name='a')
        b = ValidateArgs.check_numeric(user_input=b, to_float=True, par_name='b')
        if a >= b:
            raise LowerGteUpperError(
                par_name='Limits', 
                lower_par_name='a', 
                upper_par_name='b', 
                user_input=[a, b]
            )
    else: # system of equations
        if initial_guess is None: # initial guess for system not given
            if isinstance(f, (list, tuple)):
                x0 = zeros(len(f)) # use zero vector of length n
            else:
                raise RequiredError(
                    par_name='initial_guess', required_when="`fexpr` is a function"
                )
        else:
            x0 = conv_to_arraylike(
                array_values=initial_guess,
                to_ndarray=True,
                par_name='initial_guess'
            )
            
    tolerance_str = tolerance # used in msg
    tolerance = ValidateArgs.check_numeric(
        user_input=tolerance, 
        limits=[0, 1], 
        boundary='exclusive', 
        par_name='tolerance'
    )
    maxit = ValidateArgs.check_numeric(
        user_input=maxit, 
        limits=[1, 100],
        is_positive=True, 
        is_integer=True, 
        par_name='maxit'
    ) + 1
    valid_stop_crit = ['absolute', 'relative']
    stop_crit = ValidateArgs.check_member(
        par_name='stop_crit', valid_items=valid_stop_crit, user_input=stop_crit
    )
    error_title = _stop_crit(stop_crit)
    auto_display = ValidateArgs.check_boolean(user_input=auto_display, default=True)
    decimal_points = ValidateArgs.check_decimals(decimal_points)

    # begin calculations
    # ------------------
    if method == 'fixed-point':
        k = 1
        while k <= maxit:
            p = f(x0)
            kth_error = _kth_error(stop_crit, b=x0, p=p)
            result_table.append([x0, kth_error])
            if kth_error < tolerance:
                break
            k += 1
            x0 = p 
        col_names = ['$p_n$', error_title]
    
    if method == 'newton-raphson':
        k = 1
        while k <= maxit:
            p = x0 - f(x0) / d1f(x0)
            kth_error = _kth_error(stop_crit, b=x0, p=p)
            result_table.append([x0, kth_error])
            if kth_error < tolerance:
                break
            k += 1
            x0 = p     
        col_names = ['$p_n$', error_title]

    if method == 'mnewton-raphson':
        k, p0 = (1, x0)
        result_table.append([p0, nan])
        while k <= maxit:
            denom = (d1f(p0) ** 2 - f(p0) * d2f(p0))
            numer = (f(p0) * d1f(p0))
            if isnan(denom) or denom == 0 or isnan(numer):
                break
            p = p0 - numer / denom
            kth_error = _kth_error(stop_crit, b=p0, p=p)
            result_table.append([p, kth_error])
            if kth_error < tolerance:
                break
            k += 1
            p0 = p
        col_names = ['$p_n$', error_title]

    if method == 'secant':
        k = 2
        p0, p1, q0, q1 = (a, b, f(a), f(b))
        result_table.append([a, nan])
        result_table.append([b, nan])
        while k <= maxit:
            p = p1 - q1 * (p1 - p0) / (q1 - q0)
            kth_error = _kth_error(stop_crit, b=p1, p=p)
            result_table.append([p, kth_error])
            if kth_error < tolerance:
                break
            k += 1
            p0, q0, p1, q1 = (p1, q1, p, f(p))
        col_names = ['$p_n$', error_title]
        
    if method == 'bisection':
        k, fa, fb = (1, f(a), f(b))
        if fa * fb > 0: 
            raise NoRootInIntervalError(expr=str(fexpr), user_input=[a, b])
        while k <= maxit:
            p = a + (b - a) / 2 # better than (a + b)/2
            fp = f(p)
            kth_error = _kth_error(stop_crit, b, p)
            result_table.append([a, b, p, fp, kth_error])
            if fp == 0 or kth_error < tolerance:
                break
            k += 1
            if fa * fp > 0:
                a = p
                fa = fp
            else:
                b = p
        col_names = [
            '$a_n$', '$b_n$', '$p_n$', '$f(p_n)$', error_title
        ]
        
    if method == 'regula-falsi':
        k, fa, fb = 1, f(a), f(b)
        # check if interval contains the root
        if fa * fb > 0: 
            raise NoRootInIntervalError(expr=str(fexpr), user_input=[a, b])
        while k <= maxit:
            p = b - fb * (b - a) / (fb - fa)
            fp = f(p)
            kth_error = _kth_error(stop_crit, b=b, p=p)
            # append 
            result_table.append([a, b, p, fp, kth_error])
            if kth_error < tolerance:
                break
            k += 1
            if fp * fb < 0:
                a = b
                fa = fb
            b = p
            fb = fp 
        col_names = [
            '$a_n$', '$b_n$', '$p_n$', '$f(p_n)$', error_title
        ]

    if method == 'mregula-falsi':
        k, fa, fb, result_table = (1, f(a), f(b)/2, [])
        # check if interval contains the root
        if fa * fb > 0: 
            raise NoRootInIntervalError(expr=str(fexpr), user_input=[a, b])
        while k <= maxit:
            p = b - fb * (b - a) / (fb - fa)
            fp = f(p)/2
            kth_error = _kth_error(stop_crit, b=b, p=p)
            # append 
            result_table.append([a, b, p, fp, kth_error])
            if kth_error < tolerance:
                break
            k += 1
            if fp * fb < 0:
                a = b
                fa = fb
            b = p
            fb = fp
        col_names = [
            '$a_n$', '$b_n$', '$p_n$', '$f(p_n)$', error_title
        ]
        
    if method == 'aitken':
        k = 1
        while k <= maxit:
            x1 = f(x0)
            x2 = f(x1)
            d = (x2 - x1) - (x1 - x0)
            if isnan(d) or d == 0:
                break
            p = x2 - pow((x2 - x1), 2) / d
            kth_error = _kth_error(stop_crit, b=x2, p=p)
            # append 
            result_table.append([x0, kth_error])
            if kth_error < tolerance:
                break
            k += 1
            x0 = p
        col_names = ['$p_{n}$', error_title]
        
    if method == 'steffensen':
        k, p0 = (1, x0)
        while k <= maxit:
            p1 = f(p0)
            p2 = f(p1)
            # avoid division by NaN and zero
            denom = (p2 - 2 * p1 + p0)
            if isnan(denom) or denom == 0 or isnan(p0 - (p1 - p0) ** 2):
                break
            p = p0 - (p1 - p0) ** 2 / denom
            kth_error = _kth_error(stop_crit, b=p, p=p0)
            # append
            result_table.append([p0, p1, p2, kth_error])
            if kth_error < tolerance:
                break
            k += 1
            p0 = p
        col_names = ['$p_0$', '$p_1$', '$p_2$', error_title]

    # systems of nonlinear equations with Newton-Raphson
    if method == 'system':
        k = 1
        result_table.append(x0)
        x = asfarray(x0.copy())
        while k <= maxit:
            jac_f0 = la_jacobian(f=f, x0=x, auto_display=False)
            jac = jac_f0.jacobian
            f0 = jac_f0.answer
            if sqrt(dot(f0, f0) / len(x)) < tolerance:
                break
            dx = la_gauss_pivot(jac, -f0)
            x += dx
            p = x # these are the roots
            result_table.append(x.tolist()) # should be here, not above
            if sqrt(dot(dx, dx)) < tolerance * max(max(abs(x)), 1.0):
                break
            k += 1
        col_names = [f'$x_{index + 1}$' for index in range(len(x))]
    
    if result_table:
        index_names = dframe_labels(
            dframe=result_table, index=True
        )
        # note, None is returned as zero in df_labels() function above
        col_labels = dframe_labels(
            dframe=result_table, df_labels=col_names, index=False
        )
        dframe = DataFrame(
            data=result_table, index=index_names, columns=col_labels
        )
    else:
        p = nan

    answer = sympify(p)
    # css style
    if method == 'system':
        dframe_styled = sta_dframe_color(
            dframe=dframe, rows=[-1], decimal_points=decimal_points
        )
    else:
        try:
            answer_float = float(answer) # convert to float for isnan() to work
        except Exception:
            pass
        if ~isnan(answer_float):
            if method in closed_methods and method != 'secant': 
                dframe_styled = sta_dframe_color(
                    dframe=dframe, 
                    values=[answer_float],
                    operator='==', 
                    decimal_points=decimal_points
                )
            else:
                dframe_styled = sta_dframe_color(
                    dframe=dframe, 
                    style_indices=[[-1, 0]],
                    decimal_points=decimal_points
                )
                    
    msg = str_info_messages(maxit=maxit - 1, tolerance=tolerance_str, k=k)
    if auto_display:
        display_results({
            'dframe': dframe_styled,
            'Answer': answer,
            'msg': msg,
            'decimal_points': decimal_points
        })
    result = Result(
        table=dframe, table_styled=dframe_styled, answer=asfarray(answer) * 1, msg=msg
    )
    
    return result


def _stop_crit(stop_crit: str = 'relative') -> str:
    """
    Returns the title for the stopping criterion in iterative procedures.

    Parameters
    ----------
    stop_crit : str, optional (default='relative')
        The stopping criterion.

    Returns
    -------
    title : str
        The title for the stopping criterion.

    Raises
    ------
    ValueError
        If the provided stopping criterion is not supported.
    """
    crit_titles = {
        'absolute': 'Absolute error',
        'relative': 'Relative error',
        'function': 'Function'
    }
    title = crit_titles.get(stop_crit.lower())
    
    return title


def _kth_error(
    stop_crit: Literal['relative', 'absolute', 'function'], 
    b: int | float, 
    p: int | float
) -> float:
    """
    Calculate the error at the kth iteration of non-linear methods.

    Parameters
    ----------
    stop_crit : str
        The stopping criterion.
    b : {int, float}
        Current approximated value.
    p : {int, float}
        Previous approximated value.

    Returns
    -------
    kth_error : int or float
        The kth approximated error.

    Raises
    ------
    ValueError
        If the provided stopping criterion is not supported.

    Notes
    -----
    - For 'absolute' stopping criterion, the error is calculated as 
      the absolute difference between the current and previous 
      approximated values.
    - For 'relative' stopping criterion, the error is calculated as 
      the absolute difference divided by the previous approximated 
      value, avoiding division by zero.
    """
    stop_crit_methods = ['relative', 'absolute', 'function']
    stop_crit = ValidateArgs.check_member(
        par_name='stop_crit', 
        valid_items=stop_crit_methods, 
        default='relative'
    )
    if stop_crit == 'absolute':
        return abs(p - b)
    elif stop_crit == 'relative':
        p = 1e-15 if p == 0 else p # Avoid division by zero
        return abs((p - b) / p)
    else: # function
        pass
    
    
def nle_roots_bisection( 
    fexpr: str | Expr | Callable,
    initial_guess: float | list[float],
    tolerance: float = 1e-6,
    maxit: int = 10,
    stop_crit: Literal['relative', 'absolute'] = 'relative',
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Finds roots of nonlinear equations using the bisection method.

    Parameters
    ----------
    fexpr : {str, sympy.Expr, Callable}
        The univariate expression representing the function whose 
        roots are to be found.
    initial_guess : list_like
        The starting values as a list of two elements [a, b].
    tolerance : float, optional (default=1e-6)
        Tolerance for stopping criterion.
    maxit : int, optional (default=10)
        Maximum number of iterations.
    stop_crit : str, optional (default='relative')
        Stopping criterion.
        ========================================================
        stop_crit                   Description  
        ========================================================
        absolute .................. Calculate the absolute error
        relative .................. Calculate the relative error
        ========================================================
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with root(s) highlighted.
    answer : float
        The solution (root) of the nonlinear equation.
    msg : str
        A string with information on convergence.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = '3 ** (3*x + 1) - 7.5**(2*x)'
    >>> a, b = (1, 2)
    >>> result = stm.nle_roots_bisection(fexpr=f, initial_guess=[a, b],
    ... tolerance=1e-6, maxit=10, decimal_points=10)
             $a_n$         $b_n$         $p_n$       $f(p_n)$  Relative error
    0   1.00000000  2.0000000000  1.5000000000  -0.9866537608    0.3333333333
    1   1.00000000  1.5000000000  1.2500000000  30.5931023150    0.2000000000
    2   1.25000000  1.5000000000  1.3750000000  23.8417046249    0.0909090909
    3   1.37500000  1.5000000000  1.4375000000  14.5912606133    0.0434782609
    4   1.43750000  1.5000000000  1.4687500000   7.7399012107    0.0212765957
    5   1.46875000  1.5000000000  1.4843750000   3.6319491455    0.0105263158
    6   1.48437500  1.5000000000  1.4921875000   1.3892750639    0.0052356021
    7   1.49218750  1.5000000000  1.4960937500   0.2183288849    0.0026109661
    8   1.49609375  1.5000000000  1.4980468750  -0.3798619425    0.0013037810
    9   1.49609375  1.4980468750  1.4970703125  -0.0796971815    0.0006523157
    10  1.49609375  1.4970703125  1.4965820312   0.0695824695    0.0003262643
    
    Answer = 1.4965820313
    
    'The tolerance (1e-06) was achieved before reaching the maximum number of iterations (10).'
    """
    result = nle_roots(
        method='bisection',
        fexpr=fexpr,
        initial_guess=initial_guess,
        tolerance=tolerance,
        maxit=maxit,
        stop_crit=stop_crit,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def nle_roots_regula_falsi( 
    fexpr: str | Expr | Callable,
    initial_guess: float | list[float],
    tolerance: float = 1e-6,
    maxit: int = 10,
    stop_crit: Literal['relative', 'absolute'] = 'relative',
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Finds roots of nonlinear equations using the regula-falsi method.

    Parameters
    ----------
    fexpr : {str, sympy.Expr, Callable}
        The univariate expression representing the function whose 
        roots are to be found.
    initial_guess : list_like
        The starting values as a list of two elements [a, b].
    tolerance : float, optional (default=1e-6)
        Tolerance for stopping criterion.
    maxit : int, optional (default=10)
        Maximum number of iterations.
    stop_crit : str, optional (default='relative')
        Stopping criterion.
        ========================================================
        stop_crit                   Description  
        ========================================================
        absolute .................. Calculate the absolute error
        relative .................. Calculate the relative error
        ========================================================
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with root(s) highlighted.
    answer : float
        The solution (root) of the nonlinear equation.
    msg : str
        A string with information on convergence.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = '3 ** (3*x + 1) - 7.5**(2*x)'
    >>> a, b = (1, 2)
    >>> result = stm.nle_roots_regula_falsi(fexpr=f, initial_guess=[a, b],
    ... tolerance=1e-6, maxit=10, decimal_points=10)
        $a_n$         $b_n$         $p_n$       $f(p_n)$  Relative error
    0     1.0  2.0000000000  1.0247052218  25.7329791870    0.9517808219
    1     2.0  1.0247052218  1.0497324988  26.6945508325    0.0238415759
    2     2.0  1.0497324988  1.0750045147  27.6146541317    0.0235087532
    3     2.0  1.0750045147  1.1004290307  28.4696131038    0.0231041851
    4     2.0  1.1004290307  1.1258985678  29.2323825870    0.0226215201
    5     2.0  1.1258985678  1.1512907942  29.8731864823    0.0220554412
    6     2.0  1.1512907942  1.1764698088  30.3606344821    0.0214021766
    7     2.0  1.1764698088  1.2012884756  30.6633615243    0.0206600391
    8     2.0  1.2012884756  1.2255918910  30.7521600852    0.0198299414
    9     2.0  1.2255918910  1.2492219523  30.6024725477    0.0189158229
    10    2.0  1.2492219523  1.2720228486  30.1969949064    0.0179249110
    
    Answer = 1.2720228486
    
    'The tolerance (1e-06) was achieved before reaching the maximum number of iterations (10).'
    """
    result = nle_roots(
        method='regula-falsi',
        fexpr=fexpr,
        initial_guess=initial_guess,
        tolerance=tolerance,
        maxit=maxit,
        stop_crit=stop_crit,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def nle_roots_regula_falsi_modified( 
    fexpr: str | Expr | Callable,
    initial_guess: float | list[float],
    tolerance: float = 1e-6,
    maxit: int = 10,
    stop_crit: Literal['relative', 'absolute'] = 'relative',
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Finds roots of nonlinear equations using the modified regula-falsi method.

    Parameters
    ----------
    fexpr : {str, sympy.Expr, Callable}
        The univariate expression representing the function whose 
        roots are to be found.
    initial_guess : list_like
        The starting values as a list of two elements [a, b].
    tolerance : float, optional (default=1e-6)
        Tolerance for stopping criterion.
    maxit : int, optional (default=10)
        Maximum number of iterations.
    stop_crit : str, optional (default='relative')
        Stopping criterion.
        ========================================================
        stop_crit                   Description  
        ========================================================
        absolute .................. Calculate the absolute error
        relative .................. Calculate the relative error
        ========================================================
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with root(s) highlighted.
    answer : float
        The solution (root) of the nonlinear equation.
    msg : str
        A string with information on convergence.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = '3 ** (3*x + 1) - 7.5**(2*x)'
    >>> a, b = (1, 2)
    >>> result = stm.nle_roots_regula_falsi_modified(fexpr=f,
    ... initial_guess=[a, b], tolerance=1e-6, maxit=10,
    ... decimal_points=10)
        $a_n$         $b_n$         $p_n$       $f(p_n)$  Relative error
    0     1.0  2.0000000000  1.0247052218  25.7329791870    0.9517808219
    1     2.0  1.0247052218  1.0497324988  26.6945508325    0.0238415759
    2     2.0  1.0497324988  1.0750045147  27.6146541317    0.0235087532
    3     2.0  1.0750045147  1.1004290307  28.4696131038    0.0231041851
    4     2.0  1.1004290307  1.1258985678  29.2323825870    0.0226215201
    5     2.0  1.1258985678  1.1512907942  29.8731864823    0.0220554412
    6     2.0  1.1512907942  1.1764698088  30.3606344821    0.0214021766
    7     2.0  1.1764698088  1.2012884756  30.6633615243    0.0206600391
    8     2.0  1.2012884756  1.2255918910  30.7521600852    0.0198299414
    9     2.0  1.2255918910  1.2492219523  30.6024725477    0.0189158229
    10    2.0  1.2492219523  1.2720228486  30.1969949064    0.0179249110
    
    Answer = 1.2720228486
    
    'The tolerance (1e-06) was achieved before reaching the maximum number of iterations (10).'
    """
    result = nle_roots(
        method='mregula-falsi',
        fexpr=fexpr,
        initial_guess=initial_guess,
        tolerance=tolerance,
        maxit=maxit,
        stop_crit=stop_crit,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def nle_roots_secant( 
    fexpr: str | Expr | Callable,
    initial_guess: float | list[float],
    tolerance: float = 1e-6,
    maxit: int = 10,
    stop_crit: Literal['relative', 'absolute'] = 'relative',
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Finds roots of nonlinear equations using the secant method.

    Parameters
    ----------
    fexpr : {str, sympy.Expr, Callable}
        The univariate expression representing the function whose 
        roots are to be found.
    initial_guess : list_like
        The starting values as a list of two elements [a, b].
    tolerance : float, optional (default=1e-6)
        Tolerance for stopping criterion.
    maxit : int, optional (default=10)
        Maximum number of iterations.
    stop_crit : str, optional (default='relative')
        Stopping criterion.
        ======================================================== 
        stop_crit                   Description  
        ======================================================== 
        absolute .................. Calculate the absolute error
        relative .................. Calculate the relative error
        ======================================================== 
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with root(s) highlighted.
    answer : float
        The solution (root) of the nonlinear equation.
    msg : str
        A string with information on convergence.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = '3 ** (3*x + 1) - 7.5**(2*x)'
    >>> a, b = (1, 2)
    >>> result = stm.nle_roots_secant(fexpr=f, initial_guess=[a, b],
    ... tolerance=1e-6, maxit=10, decimal_points=10)
               $p_n$  Relative error
    0   1.0000000000             NaN
    1   2.0000000000             NaN
    2   1.0247052218    0.9517808219
    3   1.0497324988    0.0238415759
    4   0.3549408822    1.9574854611
    5   0.1752948643    1.0248219122
    6  -0.1001574796    2.7501924472
    7  -0.3241440712    0.6910093737
    8  -0.5577131351    0.4187978536
    9  -0.7813211980    0.2861922388
    10 -1.0041719767    0.2219249131
    11 -1.2239239986    0.1795471142
    
    Answer = 1.2239239986
    
    'The tolerance (1e-06) was achieved before reaching the maximum number of iterations (10).'
    """
    result = nle_roots(
        method='secant',
        fexpr=fexpr,
        initial_guess=initial_guess,
        tolerance=tolerance,
        maxit=maxit,
        stop_crit=stop_crit,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def nle_roots_fixed_point( 
    fexpr: str | Expr | Callable,
    initial_guess: float | list[float],
    tolerance: float = 1e-6,
    maxit: int = 10,
    stop_crit: Literal['relative', 'absolute'] = 'relative',
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Finds roots of nonlinear equations using the fixed point method.

    Parameters
    ----------
    fexpr : {str, sympy.Expr, Callable}
        The univariate expression representing the function whose 
        roots are to be found.
    initial_guess : list_like
        The starting values as a list of two elements [a, b].
    tolerance : float, optional (default=1e-6)
        Tolerance for stopping criterion.
    maxit : int, optional (default=10)
        Maximum number of iterations.
    stop_crit : str, optional (default='relative')
        Stopping criterion.
        ========================================================
        stop_crit                   Description  
        ========================================================
        absolute .................. Calculate the absolute error
        relative .................. Calculate the relative error
        ========================================================
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with root(s) highlighted.
    answer : float
        The solution (root) of the nonlinear equation.
    msg : str
        A string with information on convergence.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'x ** 3 + 4 * x ** 2 - 10'
    
    The above should first be transformed as follows
    
    >>> g = 'x - (x ** 3 + 4 * x ** 2 - 10) / (3 * x ** 2 + 8 * x)'
    >>> x0 = 1.5
    >>> result = stm.nle_roots_fixed_point(fexpr=g, initial_guess=x0,
    ... tolerance=1e-8, maxit=10, decimal_points=10)
    
              $p_n$  Relative error
    0  1.5000000000    0.0922330097
    1  1.3733333333    0.0059119190
    2  1.3652620149    0.0000234400
    3  1.3652300139    0.0000000004

    Answer = 1.3652300134
    
    'The tolerance (1e-08) was achieved before reaching the maximum number of iterations (10).'
    """
    result = nle_roots(
        method='fixed-point',
        fexpr=fexpr,
        initial_guess=initial_guess,
        tolerance=tolerance,
        maxit=maxit,
        stop_crit=stop_crit,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def nle_roots_newton_raphson( 
    fexpr: str | Expr | Callable,
    initial_guess: float | list[float],
    tolerance: float = 1e-6,
    maxit: int = 10,
    stop_crit: Literal['relative', 'absolute'] = 'relative',
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Finds roots of nonlinear equations using the Newton-Raphson method.

    Parameters
    ----------
    fexpr : {str, sympy.Expr, Callable}
        The univariate expression representing the function whose 
        roots are to be found.
    initial_guess : list_like
        The starting values as a list of two elements [a, b].
    tolerance : float, optional (default=1e-6)
        Tolerance for stopping criterion.
    maxit : int, optional (default=10)
        Maximum number of iterations.
    stop_crit : str, optional (default='relative')
        Stopping criterion.
        ========================================================
        stop_crit                   Description  
        ========================================================
        absolute .................. Calculate the absolute error
        relative .................. Calculate the relative error
        ========================================================
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with root(s) highlighted.
    answer : float
        The solution (root) of the nonlinear equation.
    msg : str
        A string with information on convergence.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = '3 ** (3*x + 1) - 7.5**(2*x)'
    >>> result = stm.nle_roots_newton_raphson(fexpr=f,
    ... initial_guess=1.75, tolerance=1e-6, maxit=20, decimal_points=10)
              $p_n$  Relative error
    0  1.7500000000    0.0810266504
    1  1.6188315056    0.0549296240
    2  1.5345398108    0.0220774956
    3  1.5013928175    0.0030111213
    4  1.4968855136    0.0000505379
    5  1.4968098680    0.0000000140
    
    Answer = 1.496809847
    
    'The tolerance (1e-06) was achieved before reaching the maximum number of iterations (20).'
    """
    result = nle_roots(
        method='newton-raphson',
        fexpr=fexpr,
        initial_guess=initial_guess,
        tolerance=tolerance,
        maxit=maxit,
        stop_crit=stop_crit,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def nle_roots_newton_raphson_modified( 
    fexpr: str | Expr | Callable,
    initial_guess: float | list[float],
    tolerance: float = 1e-6,
    maxit: int = 10,
    stop_crit: Literal['relative', 'absolute'] = 'relative',
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Finds roots of nonlinear equations using the modified 
    Newton-Raphson method.

    Parameters
    ----------
    fexpr : {str, sympy.Expr, Callable}
        The univariate expression representing the function whose 
        roots are to be found.
    initial_guess : list_like
        The starting values as a list of two elements [a, b].
    tolerance : float, optional (default=1e-6)
        Tolerance for stopping criterion.
    maxit : int, optional (default=10)
        Maximum number of iterations.
    stop_crit : str, optional (default='relative')
        Stopping criterion.
        ========================================================
        stop_crit                   Description  
        ========================================================
        absolute .................. Calculate the absolute error
        relative .................. Calculate the relative error
        ========================================================
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with root(s) highlighted.
    answer : float
        The solution (root) of the nonlinear equation.
    msg : str
        A string with information on convergence.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = '3 ** (3*x + 1) - 7.5**(2*x)'
    >>> result = stm.nle_roots_newton_raphson_modified(fexpr=f,
    ... initial_guess=1.75, tolerance=1e-6, maxit=10, decimal_points=10)
              $p_n$  Relative error
    0  1.7500000000             NaN
    1  1.2598674296    0.3890350356
    2  1.2918501519    0.0247573004
    3  1.3434238300    0.0383897300
    4  1.4108668171    0.0478025185
    5  1.4698035334    0.0400983634
    6  1.4941400823    0.0162879968
    7  1.4967837414    0.0017662265
    8  1.4968098445    0.0000174391
    9  1.4968098470    0.0000000017
    
    Answer = 1.496809847
    
    'The tolerance (1e-06) was achieved before reaching the maximum number of iterations (10).'
    """
    result = nle_roots(
        method='mnewton-raphson',
        fexpr=fexpr,
        initial_guess=initial_guess,
        tolerance=tolerance,
        maxit=maxit,
        stop_crit=stop_crit,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def nle_roots_aitken( 
    fexpr: str | Expr | Callable,
    initial_guess: float | list[float],
    tolerance: float = 1e-6,
    maxit: int = 10,
    stop_crit: Literal['relative', 'absolute'] = 'relative',
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Finds roots of nonlinear equations using the Aitken's method.

    Parameters
    ----------
    fexpr : {str, sympy.Expr, Callable}
        The univariate expression representing the function whose 
        roots are to be found.
    initial_guess : list_like
        The starting values as a list of two elements [a, b].
    tolerance : float, optional (default=1e-6)
        Tolerance for stopping criterion.
    maxit : int, optional (default=10)
        Maximum number of iterations.
    stop_crit : str, optional (default='relative')
        Stopping criterion.
        ========================================================
        stop_crit                   Description  
        ========================================================
        absolute .................. Calculate the absolute error
        relative .................. Calculate the relative error
        ========================================================
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with root(s) highlighted.
    answer : float
        The solution (root) of the nonlinear equation.
    msg : str
        A string with information on convergence.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'x ** 3 + 4 * x ** 2 - 10'
    
    The above should first be transformed as follows
    
    >>> g = '(10 / (x + 4)) ** (1/2)'
    >>> result = stm.nle_roots_aitken(fexpr=f, initial_guess=1.5,
    ... tolerance=1e-8, maxit=10, decimal_points=10)
            $p_{n}$    Relative error
    0  1.5000000000  1.6703910099e+01
    1  1.4662853703  4.2811373916e+00
    2  1.4517327761  5.0558759949e-01
    3  1.4495335806  9.6929911276e-03
    4  1.4494897598  3.7548156487e-06
    5  1.4494897428  5.3477971793e-13
    
    Answer = 1.4494897428
    
    'The tolerance (1e-08) was achieved before reaching the maximum number of iterations (10).'
    """
    result = nle_roots(
        method='aitken',
        fexpr=fexpr,
        initial_guess=initial_guess,
        tolerance=tolerance,
        maxit=maxit,
        stop_crit=stop_crit,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def nle_roots_steffensen( 
    fexpr: str | Expr | Callable,
    initial_guess: float | list[float],
    tolerance: float = 1e-6,
    maxit: int = 10,
    stop_crit: Literal['relative', 'absolute'] = 'relative',
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Finds roots of nonlinear equations using the Steffensen's method.

    Parameters
    ----------
    fexpr : {str, sympy.Expr, Callable}
        The univariate expression representing the function whose 
        roots are to be found.
    initial_guess : list_like
        The starting values as a list of two elements [a, b].
    tolerance : float, optional (default=1e-6)
        Tolerance for stopping criterion.
    maxit : int, optional (default=10)
        Maximum number of iterations.
    stop_crit : str, optional (default='relative')
        Stopping criterion.
        ========================================================
        stop_crit                   Description  
        ========================================================
        absolute .................. Calculate the absolute error
        relative .................. Calculate the relative error
        ========================================================
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with root(s) highlighted.
    answer : float
        The solution (root) of the nonlinear equation.
    msg : str
        A string with information on convergence.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'x ** 3 + 4 * x ** 2 - 10'
    
    The above should first be transformed as follows
    
    >>> g = '(10 / (x + 4)) ** (1/2)'
    >>> result = stm.nle_roots_steffensen(fexpr=g, initial_guess=1.5,
    ... tolerance=1e-8, maxit=10, decimal_points=10)
              $p_0$         $p_1$         $p_2$    Relative error
    0  1.5000000000  1.3483997249  1.3673763720  8.9823184028e-02
    1  1.3652652240  1.3652255336  1.3652305834  2.5790256762e-05
    2  1.3652300134  1.3652300134  1.3652300134  1.8230612806e-12
    
    Answer = 1.3652300134
    
    'The tolerance (1e-06) was achieved before reaching the maximum number of iterations (10).'
    """
    result = nle_roots(
        method='steffensen',
        fexpr=fexpr,
        initial_guess=initial_guess,
        tolerance=tolerance,
        maxit=maxit,
        stop_crit=stop_crit,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def nle_roots_system( 
    fexpr: str | Expr | Callable,
    initial_guess: float | list[float],
    tolerance: float = 1e-6,
    maxit: int = 10,
    stop_crit: Literal['relative', 'absolute'] = 'relative',
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Finds roots of a system of nonlinear equations using Newton-Raphson method.

    Parameters
    ----------
    fexpr : {str, sympy.Expr, Callable}
        The univariate expression representing the function whose 
        roots are to be found.
    initial_guess : list_like
        The starting values as a list of two elements [a, b].
    tolerance : float, optional (default=1e-6)
        Tolerance for stopping criterion.
    maxit : int, optional (default=10)
        Maximum number of iterations.
    stop_crit : str, optional (default='relative')
        Stopping criterion.
        ========================================================
        stop_crit                   Description  
        ========================================================
        absolute .................. Calculate the absolute error
        relative .................. Calculate the relative error
        ========================================================
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with root(s) highlighted.
    answer : {float, array-like}
        The solution (root) of the nonlinear equation.
    msg : str
        A string with information on convergence.

    Examples
    --------
    >>> import stemlab as stm
    >>> def f(x):
    ...     f = np.zeros(len(x))
    ...     f[0] = np.sin(x[0]) + x[1] ** 2 + np.log(x[2]) - 7.0
    ...     f[1] = 3.0 * x[0] + 2.0 ** x[1] - x[2] ** 3 + 1.0
    ...     f[2] = x[0] + x[1] + x[2] - 5.0
    ...     return f
    
    >>> x0 = [1, 1, 1]
    >>> result = stm.nle_roots_system(fexpr=f, initial_guess=x0,
    ... tolerance=1e-6, maxit=10, decimal_points=10)
              $x_1$         $x_2$         $x_3$
    0  1.0000000000  1.0000000000  1.0000000000
    1 -0.6034987164  3.4214043546  2.1820943739
    2  0.4694364381  2.5896328072  1.9409307547
    3  0.5928235097  2.4034817014  2.0036947889
    4  0.5990461263  2.3959426454  2.0050112284
    5  0.5990537566  2.3959314024  2.0050148410
    
    Answer = array([0.5990537566, 2.3959314024, 2.005014841 ])
    
    'The tolerance (1e-06) was achieved before reaching the maximum number of iterations (10).'

    >>> def f(x):
    ...     f = np.zeros(len(x))
    ...     f[0] = 3 * x[0] - np.cos(x[1] * x[2]) - 1/2
    ...     f[1] = x[0] ** 2 - 81 * (x[1] + 0.1) ** 2 + np.sin(x[2]) + 1.06
    ...     f[2] = np.exp(-x[0] * x[1]) + 20 * x[2] + (10 * np.pi - 3) / 3
    ...     return f

    >>> x0 = [1, 1, 1]
    >>> result = stm.nle_roots_system(fexpr=f, initial_guess=x0,
    ... tolerance=1e-6, maxit=10, decimal_points=10)
              $x_1$         $x_2$         $x_3$
    0  1.0000000000  1.0000000000  1.0000000000
    1  0.9196872043  0.4608224544 -0.5033876118
    2  0.5010004864  0.1874334832 -0.5208692328
    3  0.5005429355  0.0611534576 -0.5220009642
    4  0.5001044363  0.0116171085 -0.5232951460
    5  0.5000055104  0.0006056165 -0.5235829362
    6  0.5000000167  0.0000018264 -0.5235987278
    7  0.5000000000  0.0000000000 -0.5235987756
    
    Answer = array([ 0.5         ,  0.          , -0.5235987756])
    
    'The tolerance (1e-06) was achieved before reaching the maximum number of iterations (10).'
    """
    result = nle_roots(
        method='system',
        fexpr=fexpr,
        initial_guess=initial_guess,
        tolerance=tolerance,
        maxit=maxit,
        stop_crit=stop_crit,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result