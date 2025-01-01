from pandas import DataFrame, Series
from numpy import nan, triu_indices, asfarray, round
from sympy import Matrix, MatrixBase

from stemlab.core.validators.validate import ValidateArgs
from stemlab.core.datatypes import NumpyArray


def dm_dframe_lower(
    data: DataFrame, k: int = 1, decimal_points: int | None = None
) -> DataFrame:
    """
    Return the lower triangle of the input DataFrame.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame.
    k : int, optional (default=1)
        Number of diagonal elements to consider above which to zero 
        out values.
    decimal_points : int, optional (default=None)
        Number of decimal points or significant figures for symbolic
        expressions. 
        Default is None, indicating no rounding.

    Returns
    -------
    DataFrame
        DataFrame representing the lower triangle of the input data.

    Examples
    --------
    >>> import pandas as pd
    >>> import stemlab as stm
    >>> labels = ['A', 'B', 'C', 'D']
    >>> data = pd.DataFrame([[1, 0.1145, -0.4686, 0.4949],
    ... [0.1145, 1, -0.4138, 0.4745], [-0.4686, -0.4138, 1, -0.7056],
    ... [0.4949, 0.4745, -0.7056,  1]], index=labels, columns=labels)
    >>> sta.dm_dframe_lower(data)
            A       B       C    D
    A  1.0000                     
    B  0.1145     1.0             
    C -0.4686 -0.4138     1.0     
    D  0.4949  0.4745 -0.7056  1.0
    """
    data = ValidateArgs.check_dframe(par_name='data', user_input=data)
    M = asfarray(data.values)
    M[triu_indices(M.shape[0], k)] = nan
    if decimal_points is not None:
        decimal_points = ValidateArgs.check_decimals(x=decimal_points)
        try:
            M = round(M, decimal_points)
        except Exception:
            pass
    dframe = DataFrame(
        data=M, index=data.columns, columns=data.index
    ).fillna('')

    return dframe

def series_name(data: Series, n: int = 1) -> str:
    """
    Get the name of a Series.

    Parameters
    ----------
    data : Series
        Input Series.
    n : int, optional (default=1)
        Sample number to use if the Series does not have a name.

    Returns
    -------
    str
        The name of the Series if it exists, otherwise a default name 
        based on the sample number.
    """
    return (
        data.name.capitalize() if isinstance(data, Series) else f'Sample {n}'
    )


def dataframe_labels(
    object_values: NumpyArray, 
    row_names: list[str] = [], 
    column_names: list[str] = []
) -> list[str] | list[str]:

    """
    Create row and column names for a DataFrame from given inputs.

    Parameters
    ----------
    object_values : numpy.ndarray
        The data values to determine the dimensions of the DataFrame.
    row_names : list_like, optional (default=[])
        List of row names. If not provided, default row names 
        (1, 2, 3, ...) will be used.
    column_names : list_like, optional (default=[])
        List of column names. If not provided, default column names 
        (C1, C2, C3, ...) will be used.
    error_list : list_like, optional (default=[])
        List of error messages. Used internally for error handling.

    Returns
    -------
    row_names : list
        List of row names.
    column_names : list
        List of column names.
    """
    try:
        if not isinstance(object_values, MatrixBase):
            object_values = Matrix([object_values]) 
        nrows, ncols = object_values.shape
    except Exception as e:
        raise Exception(e)

    default_rownames = list(map(str, range(1, nrows + 1)))
    if row_names:
        row_names = row_names.replace(" ", "").split(",")
        if len(row_names) == len(set(row_names)): # no duplicates
            row_names = (
                default_rownames if len(row_names) != nrows else row_names
            )
        else: # duplicates found
            row_names = default_rownames
    else:
        row_names = default_rownames

    default_colnames = list(map(lambda x: f"C{x + 1}", range(ncols)))
    if column_names:
        column_names = column_names.replace(" ", "").split(",")
        if len(column_names) == len(set(column_names)):
            column_names = (
                default_colnames 
                if len(column_names) != ncols else column_names
            )
        else:
            column_names = default_colnames
    else:
        column_names = default_colnames

    return row_names, column_names