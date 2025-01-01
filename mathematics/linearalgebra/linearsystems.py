from typing import Callable

from numpy import (
    float64, diag, triu, tril, dot, zeros, asfarray, argmax, dot
)
from numpy.linalg import inv, eigvals, det

from stemlab.core.validators.errors import SingularMatrixError
from stemlab.core.arraylike import conv_to_arraylike
from stemlab.core.display import Result, display_results
from stemlab.core.base.arrays import arr_swap
from stemlab.core.validators.validate import ValidateArgs
from stemlab.core.datatypes import NumpyArray


def la_relax_parameter(A: NumpyArray) -> float:
    """
    Calculate the relaxation parameter for the Successive 
    Over-Relaxation (SOR) method.

    Parameters
    ----------
    A : numpy.ndarray
        The coefficient matrix A of the linear system of equations.

    Returns
    -------
    w : float
        The relaxation parameter (omega) for the SOR method.

    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm

    >>> A1 = np.array([[3, -1], [-1, 2]])
    >>> stm.la_relax_parameter(A1)
    1.0455488498966776

    >>> A2 = np.array([[3, -1, 0], [-1, 2, -1], [0, -1, 3]])
    >>> stm.la_relax_parameter(A2)
    1.1010205144336438

    >>> A3 = [[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1],
    ... [0, 0, -1, 4]]
    >>> stm.la_relax_parameter(A3)
    1.044640497129556

    >>> A4 = np.array([[10, -1, 2, 0], [-1, 11, -1, 3],
    ... [2, -1, 10, -1], [0, 3, -1, 8]])
    >>> stm.la_relax_parameter(A4)
    1.050134773112479
    """
    A = conv_to_arraylike(
        array_values=A, flatten_list=False, to_ndarray=True, par_name='A'
    )
    D = diag(diag(A))
    L = tril(A, k = -1)
    U = triu(A, k = 1)
    Tj = dot(inv(D), L + U)
    w = 2 / (1 + (1 - (max(abs(eigvals(Tj)))) ** 2) ** (1/2))

    return w


def la_inverse(M: list, par_name: str = 'par_name') -> NumpyArray[float64]:
    """
    Calculate the inverse of a square matrix.

    Parameters
    ----------
    M : array_like
        2D square array representing the matrix.
    par_name : str
        Name to use in error messages to describe the parameter being 
        checked.

    Returns
    -------
    m_inverse : numpy.ndarray
        The inverse of the input matrix.

    Raises
    ------
    ValueError
        If unable to calculate the determinant of the matrix.
    SingularMatrixError
        If the input matrix is singular (determinant is zero).

    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm

    >>> A = np.array([[1, 2], [3, 4]])
    >>> stm.la_inverse(A)
    array([[-2. ,  1. ],
           [ 1.5, -0.5]])

    >>> A = np.array([[5, 3, 7], [2, 4, 9], [3, 6, 4]])
    >>> stm.la_inverse(A)
    array([[ 2.85714286e-01, -2.25563910e-01,  7.51879699e-03],
           [-1.42857143e-01,  7.51879699e-03,  2.33082707e-01],
           [ 1.28552140e-17,  1.57894737e-01, -1.05263158e-01]])

    >>> A = np.array([[2, 3, -1], [3, 2, 1], [1, -5, 3]])
    >>> stm.la_inverse(A)
    array([[ 0.73333333, -0.26666667,  0.33333333],
           [-0.53333333,  0.46666667, -0.33333333],
           [-1.13333333,  0.86666667, -0.33333333]])

    >>> A = np.array([[5, -1, 1], [-2, 3, 4], [1, 1, 7]])
    >>> stm.la_inverse(A)
    array([[ 0.27419355,  0.12903226, -0.11290323],
           [ 0.29032258,  0.5483871 , -0.35483871],
           [-0.08064516, -0.09677419,  0.20967742]])

    >>> A = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1],
    ... [0, 3, -1, 8]])
    >>> stm.la_inverse(A)
    array([[ 0.10507099,  0.00933063, -0.02068966, -0.00608519],
           [ 0.00933063,  0.10250169,  0.0045977 , -0.03786342],
           [-0.02068966,  0.0045977 ,  0.10574713,  0.01149425],
           [-0.00608519, -0.03786342,  0.01149425,  0.14063556]])
    """
    try:
        m_det = det(M)
    except Exception as e:
        raise e
    if m_det == 0:
        raise SingularMatrixError(par_name=par_name, user_input=M.tolist())
    m_inverse = inv(M)

    return m_inverse


def la_jacobian(
    f: Callable,
    x0: list,
    tolerance: float = 1e-8,
    auto_display: bool = True,
    decimal_points: int = -1
):
    """
    Compute the Jacobian matrix of a vector-valued function at a given 
    point.

    Parameters
    ----------
    f : Callable
        A vector-valued function of n variables.
    x0 : array_like
        An initial guess for the variables.
    tolerance : float, optional (default=1e-8)
        A small perturbation value for finite difference approximation.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=-1)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    jacobian : numpy.ndarray
        The Jacobian matrix evaluated at x0.
    answer : numpy.ndarray
        The function value at x0.
        
    Examples
    --------
    >>> import stemlab as stm
    >>> def f(x):
            f = np.zeros(len(x))
            f[0] = np.sin(x[0]) + x[1] ** 2 + np.log(x[2]) - 7.0
            f[1] = 3.0 * x[0] + 2.0 ** x[1] - x[2] ** 3 + 1.0
            f[2] = x[0] + x[1] + x[2] - 5.0
            return f
    >>> x0 = [1, 1, 1]
    >>> result = stm.la_jacobian(f=f, x0=x0)
    
    Jacobian = Matrix([[0.540302291796024, 1.99999998784506, 0.999999993922529], [2.99999998176759, 1.38629436818860, -2.99999998176759], [0.999999993922529, 0.999999993922529, 0.999999993922529]])
    
    Answer = Matrix([[-5.15852901519210], [5.00000000000000], [-2.00000000000000]])
    """
    f = ValidateArgs.check_function(f)
    x0 = conv_to_arraylike(array_values=x0, to_ndarray=True, par_name='x0')
    tolerance = ValidateArgs.check_numeric(
        par_name='tolerance', 
        limits=[0, 1], 
        boundary='exclusive',
        user_input=tolerance
    )
    auto_display = ValidateArgs.check_boolean(user_input=auto_display, default=True)
    decimal_points = ValidateArgs.check_decimals(x=decimal_points)
    
    x = asfarray(x0.copy()) # conversion to numpy.asfarray is mandatory
    n = len(x)
    jac = zeros((n, n))
    f0 = asfarray(f(x))
    
    for i in range(n):
        temp = x[i]
        x[i] = temp + tolerance
        f1 = asfarray(f(x))
        x[i] = temp
        jac[:, i] = (f1 - f0) / tolerance
    
    if auto_display:
        display_results({
            'Jacobian': jac, 
            'Answer': f0.reshape(1, -1), 
            'decimal_points': decimal_points
        })
    result = Result(jacobian=jac, answer=f0)
    
    return result


def la_gauss_pivot(A: NumpyArray, b: NumpyArray) -> NumpyArray:
    """
    Solve a system of linear equations using Gaussian elimination with 
    partial pivoting.

    Parameters
    ----------
    A : numpy.ndarray
        A square array representing the coefficient matrix of the 
        system of equations.
    b : numpy.ndarray
        A 1D array representing the matrix of constants in the system.

    Returns
    -------
    b : numpy.ndarray
        The solution vector of the system of equations Ax = b.

    Notes
    -----
    This function solves the system of linear equations Ax = b using 
    Gaussian elimination with partial pivoting. It first performs 
    partial pivoting to avoid division by small pivot elements, then 
    performs forward elimination to obtain an upper triangular matrix, 
    and finally performs back substitution to solve for the solution 
    vector x.
    
    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[3, 2], [1, 2]])
    >>> b = np.array([18, 16])
    >>> stm.la_gauss_pivot(A, b)
    array([1. , 7.5])

    >>> A = np.array([[2, -1, 1], [6, 3, -4], [7, 2, 5]])
    >>> b = np.array([8, 4, -6])
    >>> stm.la_gauss_pivot(A, b)
    array([ 2.16842105, -5.64210526, -1.97894737])
    """
    error_singular = 'Matrix A is singular (i.e. has no determinant)'
    A = conv_to_arraylike(
        array_values=A, flatten_list=False, to_ndarray=True, par_name='A'
    )
    b = conv_to_arraylike(array_values=b, to_ndarray=True, par_name='b')
    A, b = asfarray(A), asfarray(b)
    n = A.shape[0]
    s = zeros(n)
    for i in range(n):
        s[i] = max(abs(A[i, :]))
        if s[i] == 0:
            raise ValueError(
                'Matrix A has a zero row, which makes it a singular matrix'
            )
    for k in range(n - 1):
        p = argmax(abs(A[k:n, k]) / s[k:n]) + k
        if A[p, k] == 0:
            raise ValueError(error_singular)
        if p != k:
            arr_swap(b, k, p, axis=0)
            arr_swap(s, k, p, axis=0)
            arr_swap(A, k, p, axis=0)
        for i in range(k + 1, n):
            if A[i, k] != 0.:
                lam = A[i, k] / A[k, k]
                A[i, (k + 1):n] -= lam * A[k, (k + 1):n]
                b[i] -= lam * b[k]
    if A[n - 1, n - 1] == 0:
        raise ValueError(error_singular)
    b[n - 1] /= A[n - 1, n - 1]
    for k in range(n - 2, -1, -1):
        b[k] = (b[k] - dot(A[k, (k + 1):n], b[(k + 1):n])) / A[k, k]
    
    return b