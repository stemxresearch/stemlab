from typing import Literal

from sympy import linear_eq_to_matrix, Matrix, flatten
from numpy import array, asfarray, zeros_like, identity, dot, inf, round
from numpy.linalg import norm
from pandas import DataFrame
from stemlab.core.validators.errors import (
    LengthDifferError, MatrixCompatibilityError
)
from stemlab.core.symbolic import sym_sympify, sym_get_expr_vars
from stemlab.core.arraylike import is_len_equal, conv_to_arraylike
from stemlab.core.htmlatex import tex_display_latex
from stemlab.statistical.wrangle import dframe_labels
from stemlab.core.htmlatex import sta_dframe_color
from stemlab.core.datatypes import ArrayMatrixLike, is_nested_list
from stemlab.mathematics.linearalgebra.linearsystems import (
    la_inverse, la_relax_parameter
)
from stemlab.core.display import Result, display_results
from stemlab.core.base.strings import str_info_messages
from stemlab.core.validators.validate import ValidateArgs


def la_solve(
    matrix_or_eqtns: ArrayMatrixLike,
    b_or_unknowns: ArrayMatrixLike,
    method: Literal['jacobi', 'gauss-seidel', 'sor', 'conjugate'] = 'gauss-seidel',
    variables: list | None = None,
    x0: list | None = None,
    C: ArrayMatrixLike | None = None,
    w: float = None,
    tol: float = 1e-16,
    maxit: int = 10,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Solve a system of linear equations using a specified iterative 
    method.
    
    Parameters
    ----------
    method : {jacobi, gauss-seidel, sor, conjugate}, optional (default='gauss-seidel')
        The iterative method to be used to solve the system.
        ============================================================
        method                  Description  
        ============================================================
        jacobi ................ Jacobi iteration
        gauss-seidel .......... Gauss-Seidel iteration
        sor ................... Successive over-relaxation iteration
        conjugate ............. Conjugate gradient method
        ============================================================
    matrix_or_eqtns : array_like
        A 2D square array with the coefficients of the unknowns in the 
        system of equations or a 1D array with the linear equations.
    b_or_unknowns : {array_like, None}
        A 1D array of constants (values to the right of equal sign) 
        or the unknowns in the system of equation. If `None`, the 
        unknowns will be gotten from the system of equations.
    variables : {None, array_like}, optinal (default=None)
        The unknown variables in the linear system of equations. These 
        wil form the columns of the results table.
    x0 : array_like, optional (default=None)
        A 1D array with the initial solution. If `None`, a zeros 
        matrix (column vector) whose length is equal to the number 
        of rows / columns of the coefficients matrix. 
    C : array_like, optional (default=None)
        The preconditioning matrix for the Congugate gradient method. 
        If `None`, an identity matrix of size equal to size of 
        matrix_or_eqtns will be used.
    w : {None, int, float}, optional (default=None)
        The relaxation parameter in SOR method. If `None`, the system 
        will calculate the optimal value. 
    tol : float, optional (default=1e-6)
        Tolerance level, algorithm will stop if the difference between 
        consecutive values falls below this value.
    maxit : int, optional (default=10)
        Maximum number of iterations to be performed.
    auto_display : bool, optional (default=True)
        If `True`, the results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.
    
    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the approximations at each iteration.
    dframe : pandas.Styler
        Above table with solution (last row) highlighted.
    answer : numpy.array
        The solution of the linear system.
    answer_latex : Ipython.Latex
        The solution of the linear system presented in Latex format.
    msg : str
        A string with convergence information.
        
    Notes
    -----
    If `b_or_unknowns=None`, the unknowns will be gotten from the system of 
    equations. Take caution because this assumes that the unknowns are in 
    alphabetic order. If this is not the case, then provide the variables in 
    the order they appear in the system.

    Examples
    --------
    >>> import stemlab as stm
    
    >>> A = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1],
    ... [0, 3, -1, 8]])
    >>> b = np.array([6, 25, -11, 15])
    >>> methods = ['jacobi', 'gauss-seidel', 'sor', 'conjugate']
    >>> result_table = []
    >>> for method in methods:
    ...     result = stm.la_solve(method=method, matrix_or_eqtns=A,
    ...     b_or_unknowns=b, tol=1e-6, maxit=10, decimal_points=12)
    ...     result_table.append(result.answer.tolist())
    >>> df = pd.DataFrame(result_table, index=methods, 
    ... columns=result.table.columns)
    >>> df
                              x1              x2              x3              x4
    jacobi        1.000118598691  1.999767947010 -0.999828142874  0.999785978460
    gauss-seidel  1.000000666348  2.000000024607 -1.000000209122  0.999999964632
    sor           1.000000919584  2.000001627830 -1.000000778677  0.999999454297
    conjugate     1.001524810022  1.983268765909 -1.009858497869  1.019695902153
    
    """    
    # method
    valid_methods = ['jacobi', 'gauss-seidel', 'sor', 'conjugate']
    method = ValidateArgs.check_member(
        par_name='method', valid_items=valid_methods, user_input=method
    )

    # A
    if not is_nested_list(matrix_or_eqtns):
        matrix_or_eqtns = [
            str(item).replace('=', '-') for item in flatten(matrix_or_eqtns)
        ]
        A = sym_sympify(expr_array=matrix_or_eqtns, is_expr=False)
        if b_or_unknowns is None:
            b = sym_get_expr_vars(expr_array=matrix_or_eqtns)
            # convert each unknown to string to enable sorting then convert
            # back to Symbolic expression 
            b = sym_sympify(sorted([str(item) for item in b]))
        else:
            b = sym_sympify(expr_array=b_or_unknowns, is_expr=False)
        A, b = linear_eq_to_matrix(A, b)
    else:
        A, b = (Matrix(matrix_or_eqtns), Matrix(b_or_unknowns))
    # A continue 
    A = asfarray(ValidateArgs.check_square_matrix(par_name='matrix_or_eqtns', A=A))
    nrows, ncols = A.shape

    # b
    b = asfarray(conv_to_arraylike(array_values=b, par_name='b'))

    # check compatibility
    A, b = ValidateArgs.check_coeff_const(A, b)

    # x0
    if x0 is None:
        x = zeros_like(b)
    else:
        x = asfarray(conv_to_arraylike(array_values=x0, par_name='x0'))
        if not is_len_equal(b, x):
            raise LengthDifferError(par_name="'b', 'x0'", user_input=[b, x])

    # C -> done within method == 'conjugate'
    
    # w -> done within method == 'sor'

    # variables
    if variables is None:
        variables = [f'x{k + 1}' for k in range(len(b))]
    else:
        variables = conv_to_arraylike(
            array_values=variables, par_name='variables'
        )
        if len(variables) > ncols:
            variables = variables[:ncols]
        if len(variables) < ncols:
            raise LengthDifferError(
                par_name="b, variables", user_input=[b, variables]
            )

    # tolerance
    tol_str = tol # `tol_str` will be used in the `msg`
    tol = ValidateArgs.check_numeric(
        par_name='tol', 
        user_input=tol, 
        limits=[0, 1], 
        boundary='exclusive', 
        to_float=True
    )

    maxit = ValidateArgs.check_numeric(
        user_input=maxit, 
        limits=[1, 100],
        is_positive=True, 
        is_integer=True, 
        par_name='maxit'
    ) + 1
    auto_display = ValidateArgs.check_boolean(user_input=auto_display, default=False)
    decimal_points = ValidateArgs.check_decimals(decimal_points)

    # begin computations
    # ------------------
    def kth_norm(x_new, x):    
        return norm(x=x_new - x, ord=inf) / (norm(x=x_new, ord=inf) + 1e-32)
    
    X = []
    if method == 'jacobi':
        for k in range(maxit):
            X.append(x.tolist())
            x_new = zeros_like(x)
            for i in range(nrows):
                s1 = dot(A[i, :i], x[:i])
                s2 = dot(A[i, i + 1:], x[i + 1:])
                x_new[i] = (b[i] - s1 - s2) / A[i, i]
            if kth_norm(x_new=x_new, x=x) < tol:
                break
            x = x_new.copy()
    if method == 'gauss-seidel':
        x_new = zeros_like(x)
        for k in range(1, maxit+1):
            X.append(x.tolist())
            for i in range(nrows):
                s1 = dot(A[i, :i], x_new[:i])
                s2 = dot(A[i, i + 1:], x[i + 1:])
                x_new[i] = (b[i] - s1 - s2) / A[i, i]
            if kth_norm(x_new=x_new, x=x) < tol:
                break
            x = x_new.copy()
    elif method == 'sor':
        if w:
            w = ValidateArgs.check_numeric(
                par_name='w', is_positive=True, to_float=True, user_input=w
            )
        else:
            w = la_relax_parameter(A) # calculate optimal relaxation parameter

        for k in range(maxit):
            X.append(x.tolist())
            x_new  = x.copy()
            for i in range(nrows):
                s1 = dot(A[i, :i], x[:i])
                s2 = dot(A[i, i + 1:], x_new[i + 1:])
                x[i] = x[i] * (1 - w) + (w / A[i, i]) * (b[i] - s1 - s2)
            if kth_norm(x_new=x_new, x=x) < tol:
                break

    elif method == 'conjugate':
        C = identity(nrows) if C is None else sym_sympify(expr_array=C, is_expr=False)
        C = ValidateArgs.check_square_matrix(par_name='C', A=C)
        if A.shape != C.shape:
            raise MatrixCompatibilityError(
                A='matrix_or_eqtns',
                B='C',
                multiplication=True,
                dims_A=A.shape,
                dims_B=C.shape
            )
        
        # begin computations
        C_inverse = la_inverse(M=C, par_name='C')
        r = b - dot(A, x)
        w = dot(C_inverse, r)
        v = dot(C_inverse.T, w)
        alpha = dot(w, w)
        # step 3
        k = 1
        X = []
        while k <= maxit:
            X.append(x.tolist())
            # step 4
            if norm(v, inf) < tol:
                break
            # step 5
            u = dot(A, v)
            t = alpha / dot(v, u)
            x = x + t * v
            r = r - t * u
            w = dot(C_inverse, r)
            beta = dot(w, w)
            # step 6
            if abs(beta) < tol:
                if norm(r, inf) < tol:
                    break
            # step 7
            s = beta/alpha
            v = dot(C_inverse.T, w) + s * v
            alpha = beta
            k = k + 1
    # convergence message
    msg = str_info_messages(maxit=maxit - 1, tolerance=tol_str, k=k)
    X = round(array(X), decimal_points)
    dframe = DataFrame(X)
    solution = X[-1, :]
    index_labels = dframe_labels(dframe, index=True)
    col_labels = dframe_labels(dframe=dframe, df_labels=variables, index=False)
    dframe.index = index_labels
    dframe.columns = col_labels
    solution_latex = tex_display_latex(
        lhs=variables, rhs=solution, auto_display=False
    )

    dframe_styled = sta_dframe_color(
        dframe=dframe,
        rows=[-1],
        decimal_points=decimal_points,
    )
    if auto_display:
        display_results({
            'dframe': dframe_styled,
            'answer': solution_latex,
            'msg': msg,
            'decimal_points': decimal_points
        })
    result = Result(
        table=dframe, 
        table_styled=dframe_styled, 
        answer=solution, 
        answer_latex=solution_latex, 
        msg=msg
    )
    
    return result


def la_solve_jacobi(
    matrix_or_eqtns: ArrayMatrixLike,
    b_or_unknowns: ArrayMatrixLike,
    variables: list | None = None,
    x0: list | None = None,
    tol: float = 1e-16,
    maxit: int = 10,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Solve a system of linear equations using Jacobi iterative method.
    
    Parameters
    ----------
    matrix_or_eqtns : array_like
        A 2D square array with the coefficients of the unknowns in the 
        system of equations or a 1D array with the linear equations.
    b_or_unknowns : {array_like, None}
        A 1D array of constants (values to the right of equal sign) 
        or the unknowns in the system of equation. If `None`, the 
        unknowns will be gotten from the system of equations.
    variables : {None, array_like}, optinal (default=None)
        The unknown variables in the linear system of equations. These 
        wil form the columns of the results table.
    x0 : array_like, optional (default=None)
        A 1D array with the initial solution. If `None`, a zeros 
        matrix (column vector) whose length is equal to the number 
        of rows / columns of the coefficients matrix. 
    tol : float, optional (default=1e-6)
        Tolerance level, algorithm will stop if the difference between 
        consecutive values falls below this value.
    maxit : int, optional (default=10)
        Maximum number of iterations to be performed.
    auto_display : bool, optional (default=True)
        If `True`, the results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.
    
    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the approximations at each iteration.
    dframe : pandas.Styler
        Above table with solution (last row) highlighted.
    answer : numpy.array
        The solution of the linear system.
    answer_latex : Ipython.Latex
        The solution of the linear system presented in Latex format.
    msg : str
        A string with convergence information.
        
    Notes
    -----
    If `b_or_unknowns=None`, the unknowns will be gotten from the system of 
    equations. Take caution because this assumes that the unknowns are in 
    alphabetic order. If this is not the case, then provide the variables in 
    the order they appear in the system.

    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm
    
    >>> A = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1],
    ... [0, 3, -1, 8]])
    >>> b = np.array([6, 25, -11, 15])
    >>> result = stm.la_solve_jacobi(matrix_or_eqtns=A, b_or_unknowns=b,
    ... tol=1e-6, maxit=20)
                    x1              x2              x3              x4
    0   0.000000000000  0.000000000000  0.000000000000  0.000000000000
    1   0.600000000000  2.272727272727 -1.100000000000  1.875000000000
    2   1.047272727273  1.715909090909 -0.805227272727  0.885227272727
    3   0.932636363636  2.053305785124 -1.049340909091  1.130880681818
    4   1.015198760331  1.953695764463 -0.968108626033  0.973842716942
    5   0.988991301653  2.011414725770 -1.010285903926  1.021350510072
    6   1.003198653362  1.992241260683 -0.994521736746  0.994433739846
    7   0.998128473418  2.002306881553 -1.001972230620  1.003594310151
    8   1.000625134279  1.998670301122 -0.999035575513  0.998888390590
    9   0.999674145215  2.000447671545 -1.000369157685  1.000619190140
    10  1.000118598691  1.999767947010 -0.999828142874  0.999785978460
    11  0.999942423276  2.000084774585 -1.000068327191  1.000108502012
    12  1.000022142897  1.999958962732 -0.999969156995  0.999959668632
    13  0.999989727672  2.000015816364 -1.000012565443  1.000019244351
    14  1.000004094725  1.999992675380 -0.999994439463  0.999992498183
    15  0.999998155431  2.000002923701 -1.000002301589  1.000003441800
    16  1.000000752688  1.999998684404 -0.999998994536  0.999998615913
    17  0.999999667348  2.000000537310 -1.000000420506  1.000000619032
    
    x1 = 0.999999667348 
    x2 = 2.00000053731 
    x3 = -1.000000420506 
    x4 = 1.000000619032 
    
    'The tolerance (1e-06) was achieved before reaching the maximum number of iterations (20).'

    """
    result = la_solve(
        matrix_or_eqtns=matrix_or_eqtns,
        b_or_unknowns=b_or_unknowns,
        method='jacobi',
        variables=variables,
        x0=x0,
        C=None,
        w=None,
        tol= tol,
        maxit=maxit,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def la_solve_gauss_seidel(
    matrix_or_eqtns: ArrayMatrixLike,
    b_or_unknowns: ArrayMatrixLike,
    variables: list | None = None,
    x0: list | None = None,
    tol: float = 1e-16,
    maxit: int = 10,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Solve a system of linear equations using Gauss-Seidel iterative 
    method.
    
    Parameters
    ----------
    matrix_or_eqtns : array_like
        A 2D square array with the coefficients of the unknowns in the 
        system of equations or a 1D array with the linear equations.
    b_or_unknowns : {array_like, None}
        A 1D array of constants (values to the right of equal sign) 
        or the unknowns in the system of equation. If `None`, the 
        unknowns will be gotten from the system of equations.
    variables : {None, array_like}, optinal (default=None)
        The unknown variables in the linear system of equations. These 
        wil form the columns of the results table.
    x0 : array_like, optional (default=None)
        A 1D array with the initial solution. If `None`, a zeros 
        matrix (column vector) whose length is equal to the number 
        of rows / columns of the coefficients matrix. 
    tol : float, optional (default=1e-6)
        Tolerance level, algorithm will stop if the difference between 
        consecutive values falls below this value.
    maxit : int, optional (default=10)
        Maximum number of iterations to be performed.
    auto_display : bool, optional (default=True)
        If `True`, the results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.
    
    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the approximations at each iteration.
    dframe : pandas.Styler
        Above table with solution (last row) highlighted.
    answer : numpy.array
        The solution of the linear system.
    answer_latex : Ipython.Latex
        The solution of the linear system presented in Latex format.
    msg : str
        A string with convergence information.
        
    Notes
    -----
    If `b_or_unknowns=None`, the unknowns will be gotten from the system of 
    equations. Take caution because this assumes that the unknowns are in 
    alphabetic order. If this is not the case, then provide the variables in 
    the order they appear in the system.

    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm
    
    >>> A = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1],
    ... [0, 3, -1, 8]])
    >>> b = np.array([6, 25, -11, 15])
    >>> result = stm.la_solve_gauss_seidel(matrix_or_eqtns=A,
    ... b_or_unknowns=b, tol=1e-6, maxit=10)
                   x1              x2              x3              x4
    0  0.000000000000  0.000000000000  0.000000000000  0.000000000000
    1  0.600000000000  2.327272727273 -0.987272727273  0.878863636364
    2  1.030181818182  2.036938016529 -1.014456198347  0.984341219008
    3  1.006585041322  2.003555016905 -1.002527384673  0.998350945577
    4  1.000860978625  2.000298250657 -1.000307276102  0.999849746491
    5  1.000091280286  2.000021342246 -1.000031147183  0.999988103260
    6  1.000008363661  2.000001173336 -1.000002745073  0.999999216865
    7  1.000000666348  2.000000024607 -1.000000209122  0.999999964632
    
    x1 = 1.000000666348 
    x2 = 2.000000024607 
    x3 = -1.000000209122 
    x4 = 0.999999964632 
    
    'The tolerance (1e-06) was achieved before reaching the maximum number of iterations (10).'
    
    """
    result = la_solve(
        matrix_or_eqtns=matrix_or_eqtns,
        b_or_unknowns=b_or_unknowns,
        method='gauss-seidel',
        variables=variables,
        x0=x0,
        C=None,
        w=None,
        tol= tol,
        maxit=maxit,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result

def la_solve_sor(
    matrix_or_eqtns: ArrayMatrixLike,
    b_or_unknowns: ArrayMatrixLike,
    variables: list | None = None,
    x0: list | None = None,
    w: float = None,
    tol: float = 1e-16,
    maxit: int = 10,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Solve a system of linear equations using the 
    Successive-Over-Relaxation (SOR) iterative method.
    
    Parameters
    ----------
    matrix_or_eqtns : array_like
        A 2D square array with the coefficients of the unknowns in the 
        system of equations or a 1D array with the linear equations.
    b_or_unknowns : {array_like, None}
        A 1D array of constants (values to the right of equal sign) 
        or the unknowns in the system of equation. If `None`, the 
        unknowns will be gotten from the system of equations.
    variables : {None, array_like}, optinal (default=None)
        The unknown variables in the linear system of equations. These 
        wil form the columns of the results table.
    x0 : array_like, optional (default=None)
        A 1D array with the initial solution. If `None`, a zeros 
        matrix (column vector) whose length is equal to the number 
        of rows / columns of the coefficients matrix.
    w : {None, int, float}, optional (default=None)
        The relaxation parameter in SOR method. If `None`, the system 
        will calculate the optimal value. 
    tol : float, optional (default=1e-6)
        Tolerance level, algorithm will stop if the difference between 
        consecutive values falls below this value.
    maxit : int, optional (default=10)
        Maximum number of iterations to be performed.
    auto_display : bool, optional (default=True)
        If `True`, the results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.
    
    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the approximations at each iteration.
    dframe : pandas.Styler
        Above table with solution (last row) highlighted.
    answer : numpy.array
        The solution of the linear system.
    answer_latex : Ipython.Latex
        The solution of the linear system presented in Latex format.
    msg : str
        A string with convergence information.
        
    Notes
    -----
    If `b_or_unknowns=None`, the unknowns will be gotten from the system of 
    equations. Take caution because this assumes that the unknowns are in 
    alphabetic order. If this is not the case, then provide the variables in 
    the order they appear in the system.

    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm
    
    >>> A = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1],
    ... [0, 3, -1, 8]])
    >>> b = np.array([6, 25, -11, 15])
    >>> result = stm.la_solve_sor(matrix_or_eqtns=A, b_or_unknowns=b,
    ... tol=1e-6, maxit=10)
                   x1              x2              x3              x4
    0  0.000000000000  0.000000000000  0.000000000000  0.000000000000
    1  0.630080863867  2.446821741167 -1.030532956027  0.870168165338
    2  1.071880860508  2.018729933839 -1.025233352347  0.995820928922
    3  1.003662848998  1.998198599738 -1.000132258717  1.000901547991
    4  0.999654970487  1.999786543916 -0.999848644817  1.000058728135
    5  0.999963093543  2.000004807865 -0.999993164671  0.999996059591
    6  1.000000919584  2.000001627830 -1.000000778677  0.999999454297
    
    x1 = 1.000000919584 
    x2 = 2.00000162783 
    x3 = -1.000000778677 
    x4 = 0.999999454297 
    
    'The tolerance (1e-06) was achieved before reaching the maximum number of iterations (10).'
    
    """
    result = la_solve(
        matrix_or_eqtns=matrix_or_eqtns,
        b_or_unknowns=b_or_unknowns,
        method='sor',
        variables=variables,
        x0=x0,
        C=None,
        w=w,
        tol= tol,
        maxit=maxit,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def la_solve_conjugate(
    matrix_or_eqtns: ArrayMatrixLike,
    b_or_unknowns: ArrayMatrixLike,
    variables: list | None = None,
    x0: list | None = None,
    C: ArrayMatrixLike | None = None,
    tol: float = 1e-16,
    maxit: int = 10,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Solve a system of linear equations using the conjugate gradient 
    iterative method.
    
    Parameters
    ----------
    matrix_or_eqtns : array_like
        A 2D square array with the coefficients of the unknowns in the 
        system of equations or a 1D array with the linear equations.
    b_or_unknowns : {array_like, None}
        A 1D array of constants (values to the right of equal sign) 
        or the unknowns in the system of equation. If `None`, the 
        unknowns will be gotten from the system of equations.
    variables : {None, array_like}, optinal (default=None)
        The unknown variables in the linear system of equations. These 
        wil form the columns of the results table.
    x0 : array_like, optional (default=None)
        A 1D array with the initial solution. If `None`, a zeros 
        matrix (column vector) whose length is equal to the number 
        of rows / columns of the coefficients matrix. 
    C : array_like, optional (default=None)
        The preconditioning matrix for the Congugate gradient method. 
        If `None`, an identity matrix of size equal to size of 
        matrix_or_eqtns will be used.
    tol : float, optional (default=1e-6)
        Tolerance level, algorithm will stop if the difference between 
        consecutive values falls below this value.
    maxit : int, optional (default=10)
        Maximum number of iterations to be performed.
    auto_display : bool, optional (default=True)
        If `True`, the results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.
    
    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the approximations at each iteration.
    dframe : pandas.Styler
        Above table with solution (last row) highlighted.
    answer : numpy.array
        The solution of the linear system.
    answer_latex : Ipython.Latex
        The solution of the linear system presented in Latex format.
    msg : str
        A string with convergence information.
        
    Notes
    -----
    If `b_or_unknowns=None`, the unknowns will be gotten from the system of 
    equations. Take caution because this assumes that the unknowns are in 
    alphabetic order. If this is not the case, then provide the variables in 
    the order they appear in the system.

    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm
    
    >>> A = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1],
    ... [0, 3, -1, 8]])
    >>> b = np.array([6, 25, -11, 15])
    >>> result = stm.la_solve_conjugate(matrix_or_eqtns=A,
    ... b_or_unknowns=b, tol=1e-16, maxit=10)
                   x1              x2              x3              x4
    0  0.000000000000  0.000000000000  0.000000000000  0.000000000000
    1  0.471625946452  1.965108110218 -0.864647568496  1.179064866131
    2  0.996432359996  1.976565314555 -0.909846944904  1.097591134432
    3  1.001524810022  1.983268765909 -1.009858497869  1.019695902153
    4  1.000000000000  2.000000000000 -1.000000000000  1.000000000000
    
    x1 = 1  
    x2 = 2  
    x3 = -1  
    x4 = 1  
    
    'The tolerance (1e-16) was achieved before reaching the maximum number of iterations (10).'  
    
    """
    result = la_solve(
        matrix_or_eqtns=matrix_or_eqtns,
        b_or_unknowns=b_or_unknowns,
        method='conjugate',
        variables=variables,
        x0=x0,
        C=C,
        w=None,
        tol= tol,
        maxit=maxit,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result