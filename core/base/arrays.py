from typing import Literal

from numpy import argmax, float64, array, ndarray
from sympy import Matrix, flatten

from stemlab.core.datatypes import ArrayMatrixLike, ListArrayLike, NumpyArraySympyMatrix


def list_elements_not_in(lst: list, lower: int, upper: int):
    """
    Extract elements from the list that are not between the specified 
    lower and upper bounds.

    Parameters
    ----------
    lst : list
        List of numerical elements to check.
    lower : int
        The lower bound of the range (inclusive).
    upper : int
        The upper bound of the range (inclusive).

    Returns
    -------
    list
        A list of elements that are outside the specified range.

    Examples
    --------
    >>> import stemlab as stm
    >>> x = [1, 2, 5, -3]
    >>> stm.list_elements_not_in(x, lower=-2, upper=4)
    [5, -3]

    >>> x = [0, 2, 3]
    >>> stm.list_elements_not_in(x, lower=0, upper=4)
    []

    >>> x = [0, 1, 2, 3, 4, 5]
    >>> stm.list_elements_not_in(x, lower=1, upper=3)
    [0, 4, 5]
    """
    return [x for x in lst if x < lower or x > upper]


def arr_get(
    A: ArrayMatrixLike,
    i: int,
    axis: Literal[0, 1, 'index', 'rows', 'columns'] = 0
):
    """
    Extracts specific rows from an array, matrix, or list.
    
    Parameters
    ----------
    A : ArrayMatrixLike
        An m by n array or a 1D list / tuple.
    i : int
        The indices of the rows or columns to be extracted.
    axis: {0, 1, 'index', 'rows', 'columns'}, optional(default=0)
        If `0` or `index` or `rows`, rows will be extracted.

    Returns
    -------
    A : ArrayMatrixLike
        An m by n array representing extracted the row (s) / column(s).
        
    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm
    
    >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> stm.arr_get(A, i=1, axis=0)
    array([4, 5, 6])

    >>> stm.arr_get(A, i=[0, 2], axis=0)
    array([[1, 2, 3],
           [7, 8, 9]])

    >>> stm.arr_get(A, i=0, axis=1)
    array([1, 4, 7])

    >>> stm.arr_get(A, i=[0, 2], axis=1)
    array([[1, 3],
           [4, 6],
           [7, 9]])

    >>> A = np.array([10, 20, 30, 40, 50])
    >>> stm.arr_get(A, i=[0, 4])
    array([10, 50])
    """
    from stemlab.core.validators.validate import ValidateArgs
    
    A = ValidateArgs.check_array_matrix(A=A, par_name='A')
    m = A.shape 
    if len(m) == 1:
        n = m[0]
    else:
        n = m[0] if axis == 0 else m[1]
    if isinstance(i, int):
        i = [i]
    else:
        try:
            i = flatten(list(i))
        except Exception as e:
            raise e
        
    lst_elements = list_elements_not_in(lst=i, lower=-n, upper=n - 1)
    if lst_elements:
        rows_cols = 'rows' if axis == 1 else 'cols'
        raise ValueError(
            f"Expected all elements in '{rows_cols}' to be between "
            f"{-n} and {n - 1}. The values {lst_elements} were out "
            f"of the specified range"
        )
    try:
        A = A[i, :] if axis == 0 else A[:, i]
    except:
        A = list(A[i])
    
    return A
 

def arr_get_rows(
    A: ArrayMatrixLike, rows: ListArrayLike
) -> NumpyArraySympyMatrix:
    """
    Extracts specific rows from an array, matrix, or list.

    Parameters
    ----------
    A : {NumpyArraySympyMatrix, list}
        An m by n array or matrix, or a 1D list.
    rows : {int, list of int}
        The indices of the rows to be extracted.

    Returns
    -------
    rows : {NumpyArraySympyMatrix, list}
        The extracted rows from the input array or list.

    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm
    
    >>> A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    >>> stm.arr_get_rows(A, rows=1)
    array([5, 6, 7, 8])
    
    >>> stm.arr_get_rows(A, rows=-1)
    array([ 9, 10, 11, 12])
    
    >>> stm.arr_get_rows(A, rows=[-1, 0])
    >>> array([[ 9, 10, 11, 12],
               [ 1,  2,  3,  4]])
    
    >>> A = [10, 20, 30, 40, 50, 60]
    >>> stm.arr_get_rows(A, rows=[0, 3, -1])
    [10, 40, 60]
    """
    rows = arr_get(A=A, i=rows, axis=0)

    return rows

    
def arr_get_cols(
    A: ArrayMatrixLike, cols: ListArrayLike
) -> NumpyArraySympyMatrix:
    """
    Extracts specific rows from an array, matrix, or list.

    Parameters
    ----------
    A : {NumpyArraySympyMatrix, list}
        An m by n array or matrix, or a 1D list.
    rows : {int, list of int}
        The indices of the rows to be extracted.

    Returns
    -------
    rows : {NumpyArraySympyMatrix, list}
        The extracted rows from the input array or list.

    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm
    
    >>> A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    >>> stm.arr_get_cols(A, cols=1)
    array([[ 2],
           [ 6],
           [10]])
    
    >>> stm.arr_get_cols(A, cols=-1)
    array([[ 4],
           [ 8],
           [12]])
    
    >>> stm.arr_get_cols(A, cols=[-1, 0])
    array([[ 4,  1],
           [ 8,  5],
           [12,  9]])
    
    >>> A = [10, 20, 30, 40, 50, 60]
    >>> stm.arr_get_cols(A, cols=[0, 3, -1])
    [10, 40, 60]
    """
    cols = arr_get(A=A, i=cols, axis=1)
    
    return cols


def arr_swap(
    A: ArrayMatrixLike,
    i: int,
    j: int,
    axis: Literal[0, 1, 'index', 'rows', 'columns'] = 0
) -> NumpyArraySympyMatrix:
    """
    Swap rows or columns of a matrix or vector elements.

    Parameters
    ----------
    A : ArrayMatrixLike
        An m by n array or a 1D list / tuple.
    i, j : int
        The indices of the rows or columns to be swapped.
    axis: {0, 1, 'index', 'rows', 'columns'}, optional(default=0)
        The axis (rows / columns) to be swapped.

    Returns
    -------
    A : ArrayMatrixLike
        An m by n array after swapping the rows or elements specified by `i` 
        and `j`.

    Notes
    -----
    This function swaps rows, columns or elements `i` and `j` of the matrix `A`. 
    If `A` is a 1D array, it simply swaps the elements at indices 
    `i` and `j`. If `A` is a 2D array, it swaps the entire rows or columns.
    
    Example
    -------
    >>> import numpy as np
    >>> import stemlab as stm
    
    >>> A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    >>> A
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])
            
    >>> stm.arr_swap(A=A, i=0, j=2)
    array([[ 9, 10, 11, 12],
           [ 5,  6,  7,  8],
           [ 1,  2,  3,  4]])
           
    >>> stm.arr_swap(A=A, i=2, j=3, axis=1)
    array([[ 1,  2,  4,  3],
           [ 5,  6,  8,  7],
           [ 9, 10, 12, 11]])
           
    >>> stm.arr_swap(A=A, i=1, j=0)
    array([[ 5,  6,  7,  8],
           [ 1,  2,  3,  4],
           [ 9, 10, 11, 12]])
    """
    from stemlab.core.validators.validate import ValidateArgs
    
    A = ValidateArgs.check_array_matrix(A=A, par_name='A')
    nrows = A.shape[0]
    i = ValidateArgs.check_numeric(
        par_name='i',
        limits=[-nrows + 1, nrows - 1],
        is_integer=True,
        user_input=i
    )
    if len(A.shape) == 1:
        j = ValidateArgs.check_numeric(
            par_name='j',
            limits=[-nrows + 1, nrows - 1],
            is_integer=True,
            user_input=j
        )
        A[i], A[j] = A[j], A[i]
    else:
        ncols = A.shape[1]
        j = ValidateArgs.check_numeric(
            par_name='j',
            limits=[-ncols + 1, ncols - 1],
            is_integer=True,
            user_input=j
        )
        if axis == 1:
            A[:, [i, j]] = A[:, [j, i]]
        else:
            A[[i, j], :] = A[[j, i], :]
        
    return A

    
def arr_swap_rows(
    A: ArrayMatrixLike, i: int, j: int
) -> NumpyArraySympyMatrix:
    """
    Swap rows of an array.

    Parameters
    ----------
    A : ArrayMatrixLike
        An m by n array or a 1D array.
    i, j : int
        The indices of the rows to be swapped.

    Returns
    -------
    A : ArrayMatrixLike
        An m by n array after swapping the rows specified by `i` 
        and `j`.
    
    Example
    -------
    >>> import numpy as np
    >>> import stemlab as stm
    
    >>> A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    >>> A
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])
           
    >>> stm.arr_swap_rows(A=A, i=1, j=0)
    array([[ 5,  6,  7,  8],
           [ 1,  2,  3,  4],
           [ 9, 10, 11, 12]])
    """
    A = arr_swap(A=A, i=i, j=j, axis=0)
    
    return A

    
def arr_swap_cols(
    A: ArrayMatrixLike, i: int, j: int
) -> NumpyArraySympyMatrix:
    """
    Swap columns of an array.

    Parameters
    ----------
    A : ArrayMatrixLike
        An m by n array.
    i, j : int
        The indices of the columns to be swapped.

    Returns
    -------
    A : ArrayMatrixLike
        An m by n array after swapping the columns specified by `i` 
        and `j`.
    
    Example
    -------
    >>> import numpy as np
    >>> import stemlab as stm
    
    >>> A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    >>> A
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])
           
    >>> stm.arr_swap_cols(A=A, i=2, j=3)
    array([[ 1,  2,  4,  3],
           [ 5,  6,  8,  7],
           [ 9, 10, 12, 11]])
    """
    A = arr_swap(A=A, i=i, j=j, axis=1)
    
    return A


def arr_contains_string(arr: ArrayMatrixLike) -> bool:
    """
    Check if a array-like contains at least one string.

    Parameters
    ----------
    arr : array-like
        The input array to be checked.

    Returns
    -------
    string_found : bool
        True if the array-like contains at least one string, False 
        otherwise.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.arr_contains_string(['apple', 3, 'banana', True])
    True
    >>> stm.arr_contains_string([1, 2, 3, 4])
    False
    """
    try:
        arr = flatten(arr)
    except Exception as e:
        raise e
    string_found = len(list(filter(lambda x: isinstance(x, str), arr))) > 0
    
    return string_found


def arr_max_zeros(
    A: ArrayMatrixLike,
    axis: Literal[0, 1, 'index', 'rows', 'columns'] = 0
) -> tuple[int, Matrix]:
    """
    Find the row or column with the maximum number of zeros in an array.

    Parameters
    ----------
    A : {array_like, sympy.Matrix}
        Input array or matrix.
    axis : {0, 1, 'index', 'rows', 'columns'}, optional (default=0)
        If `0` or `index` or `rows`, consider rows, otherwise consider 
        columns.

    Returns
    -------
    result : tuple
        - Index of the row or column with maximum zeros
        = The corresponding row / column values.

    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm
    >>> M1 = np.array([[1, 0, 1], [0, 0, 0], [1, 1, 1]])
    >>> stm.arr_max_zeros(M1)
    (1, Matrix([[0, 0, 0]]))

    >>> M2 = np.array([[0, 1, 0, 3], [1, 1, 1, 1]])
    >>> stm.arr_max_zeros(M2)
    (0, Matrix([[0, 1, 0, 3]]))

    >>> M3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> stm.arr_max_zeros(M3)
    (None, None)
    """
    from stemlab.core.validators.errors import SympifyError
    from stemlab.core.validators.validate import ValidateArgs
    
    try:
        is_array = isinstance(A, ndarray)
        A = Matrix(A)
    except:
        raise SympifyError(par_name='A', to='a matrix', user_input=A)
    
    axis = ValidateArgs.check_axis(user_input=axis, par_name='axis')
    
    if axis == 0:
        N = array(A, dtype=float64).T == 0 # we must transpose
    sum_rows = sum(N) # sum columns, (i.e. rows in original matrix)
    if sum(sum_rows) == 0:
        max_zeros_index, max_zeros_row = [None] * 2
    else:
        max_zeros_index = argmax(sum_rows)
        max_zeros_row = A.row(max_zeros_index)
        if is_array:
            max_zeros_row = max_zeros_row.tolist()
            
    result = max_zeros_index, max_zeros_row

    return result