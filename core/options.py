from contextlib import contextmanager
from pandas import set_option, get_option
from numpy import set_printoptions, get_printoptions

@contextmanager
def temporary_display_options(
    decimal_points: int = 14,
    nrows: int = 50,
    ncols: int = 50,
    numpy_options: bool = False
):
    """
    Context manager for temporarily setting display options for 
    NumPy arrays and Pandas DataFrames.

    Parameters
    ----------
    decimal_points : int, optional (default=14)
        Number of decimal points or significant figures for symbolic 
        expressions.
    nrows : int, optional (default=50)
        Maximum number of DataFrame rows to display.
    ncols : int, optional (default=50)
        Maximum number of DataFrame columns to display.
    numpy_options : bool, optional (default=False)
        If `True`, option for setting numpy decimals is also set.
    """
    # Save the current options
    original_pandas_precision = get_option('display.precision')
    original_pandas_max_rows = get_option('display.max_rows')
    original_pandas_max_columns = get_option('display.max_columns')
    original_numpy_options = get_printoptions()
    
    if numpy_options:
        set_printoptions(precision=decimal_points)
    
    set_option('display.precision', decimal_points)
    set_option('display.max_rows', nrows)
    set_option('display.max_columns', ncols)
    
    try:
        yield
    finally:
        # Restore the original options
        set_option('display.precision', original_pandas_precision)
        set_option('display.max_rows', original_pandas_max_rows)
        set_option('display.max_columns', original_pandas_max_columns)
        if numpy_options:
            set_printoptions(**original_numpy_options)