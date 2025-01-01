from numpy import asfarray
from sympy import Matrix, MatrixBase

from stemlab.core.datatypes import ListArrayLike, NumpyArraySympyMatrix
from stemlab.core.validators.validate import ValidateArgs


def _echelon(A: ListArrayLike, is_ref: bool = True) -> NumpyArraySympyMatrix:
    """
    Reduce a matrix to its row echelon or reduced echelon form.

    This function converts the input matrix into its row echelon or
    reduced echelon form. It handles both SymPy matrices and NumPy
    arrays by checking the type of the input and applying the
    appropriate operations.

    Parameters
    ----------
    A : ListArrayLike
        A matrix represented as a list, NumPy array, or SymPy matrix.
    is_ref : bool, optional (default=True)
        If `True`, array `A` will be reduced to row echelon form (ref),
        otherwise it will be converted to reduced row echelon form
        (rref)

    Returns
    -------
    A : {numpy.ndarray, sympy.Matrix}
        The row echelon form (ref) or reduced row echelon form (rref)
        of the input matrix.

    Notes
    -----
    The return type will match the type of the input: if the input is
    a SymPy matrix, the output will also be a SymPy matrix; otherwise,
    it will be a NumPy array.

    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm
    >>> A = np.array([[2, 1, -1, -4], [-3, -1, 2, -6], [-2, 1, 2, -3]])
    >>> stm.la_echelon_form(A)
    
    >>> B = sp.Matrix([[2, 1, -1, -4], [-3, -1, 2, -6], [-2, 1, 2, -3]])
    >>> stm.la_echelon_form(B)
    """
    A = ValidateArgs.check_array_matrix(A=A, par_name='A')
    is_ref = ValidateArgs.check_boolean(user_input=is_ref, default=True)
    is_sympy = True if isinstance(A, MatrixBase) else False
    A = Matrix(A) # should be here
    A = A.echelon_form() if is_ref else A.rref()[0]
    A = A if is_sympy else asfarray(A)
    
    return A


def la_ref(A: ListArrayLike) -> NumpyArraySympyMatrix:
    """
    Convert a matrix to its row echelon form. Same as 
    `la_echelon_form()`.

    Parameters
    ----------
    A : ListArrayLike
        A matrix represented as a list, NumPy array, or SymPy matrix.

    Returns
    -------
    {numpy.ndarray, sympy.Matrix}
        The row echelon form of the input matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy as sym
    >>> import stemlab as stm
    >>> A = np.array([[2, 1, -1, -4], [-3, -1, 2, -6], [-2, 1, 2, -3]])
    >>> stm.la_ref(A)

    >>> B = sym.Matrix([[2, 1, -1, -4], [-3, -1, 2, -6], [-2, 1, 2, -3]])
    >>> stm.la_ref(B)
    """
    return _echelon(A=A, is_ref=True)


def la_echelon_form(A):
    """
    Convert a matrix to its row echelon form (same as `la_ref()`).

    Parameters
    ----------
    A : ListArrayLike
        A matrix represented as a list, NumPy array, or SymPy matrix.

    Returns
    -------
    {numpy.ndarray, sympy.Matrix}
        The row echelon form of the input matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy as sym
    >>> import stemlab as stm
    >>> A = np.array([[2, 1, -1, -4], [-3, -1, 2, -6], [-2, 1, 2, -3]])
    >>> stm.la_echelon_form(A)

    >>> B = sym.Matrix([[2, 1, -1, -4], [-3, -1, 2, -6], [-2, 1, 2, -3]])
    >>> stm.la_echelon_form(B)
    """
    return _echelon(A=A, is_ref=True)


def la_rref(A):
    """
    Convert a matrix to its reduced row echelon form.

    Parameters
    ----------
    A : ListArrayLike
        A matrix represented as a list, NumPy array, or SymPy matrix.

    Returns
    -------
    {numpy.ndarray, sympy.Matrix}
        The reduced row echelon form of the input matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy as sym
    >>> import stemlab as stm
    >>> A = np.array([[2, 1, -1, -4], [-3, -1, 2, -6], [-2, 1, 2, -3]])
    >>> stm.la_rref(A)

    >>> B = sym.Matrix([[2, 1, -1, -4], [-3, -1, 2, -6], [-2, 1, 2, -3]])
    >>> stm.la_rref(B)
    """
    return _echelon(A=A, is_ref=False)