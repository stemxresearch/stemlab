import os

from numpy import asarray, isnan
from sympy import latex
from pandas import DataFrame


def domain_name() -> str:
    """
    Return the domain name based on the environment.

    Returns
    -------
    str
        The domain name.
    """
    if 'DOMAIN_NAME' in os.environ:
        return os.environ['DOMAIN_NAME']
    else:
        return "http://127.0.0.1:3000"


def appname_title(appname: str) -> str:
    """
    Convert an application name to a formatted title.

    Parameters
    ----------
    appname : str
        The name of the application.

    Returns
    -------
    str
        The formatted application name as a title.

    Raises
    ------
    TypeError
        If `appname` is not a string.

    Example
    -------
    >>> appname_title("my-awesome-app")
    'My awesome app'
    """
    if not isinstance(appname, str):
        raise TypeError("appname must be a string")
    appname = appname.capitalize().replace('-', ' ')
    return f"<span style ='font-weight:600;'>{appname}</span>"


def dict_to_querystring(url_path: str, query_string: str) -> str:
    """
    Convert a dictionary to a query string format and append it to a 
    URL path.

    Parameters
    ----------
    url_path : str
        The base URL path.
    query_string : dict
        The dictionary representing the query parameters.

    Returns
    -------
    query_string : str
        The URL with the appended query string.

    Example
    -------
    >>> dict_to_querystring('/search', {'q': 'python', 'page': 1})
    '/search?q=python&page=1'
    """
    query_string = [f'{key}={value}' for key, value in query_string.items()]
    query_string = '&'.join(query_string)
    query_string = f'/{url_path}?{query_string}'\
        .replace('+', '%2B')\
        .replace('**', '^')\
        .replace('*', ' * ')\
        .replace('+', 'add')\
        .replace('/(', ' / (')\
        .replace(')/', ') / ')\
        .replace('  ', ' ')
    
    return query_string


def xy_coordinates(
    coordinates: list, 
    variable_name: str, 
    sub_scripts: str | None = None
) -> str:
    """
    Format xy-coordinates as LaTeX expression.

    Parameters
    ----------
    coordinates : list_like
        List of x and y coordinates.
    variable_name : str
        Name of the variable representing the coordinates.
    sub_scripts : {str, None}, optional (default=None)
        Subscripts for x and y variables.

    Returns
    -------
    latex_syntax : str
        LaTeX expression representing the coordinates.

    Example
    -------
    >>> xy_coordinates([3, 4], 'P', '1')
    '$ P_{1}(x, y) = \\displaystyle \\left(3, 4\\right) $'
    """
    if sub_scripts is not None:
        xy_coords = f"(x_{sub_scripts}, y_{sub_scripts})"
    else:
        xy_coords = "(x, y)"

    latex_syntax = (
        f"$ {variable_name}{xy_coords} = \\displaystyle "
        f"\\left({latex(coordinates[0])}, {latex(coordinates[1])}\\right) $"
    )

    return latex_syntax


def dict_df(
    user_inputs, 
    transpose: bool = True, 
    remove_trail_zeros: bool = False, 
    decimal_points: int = 2
) -> DataFrame:
    """
    Convert dictionary of user inputs to a DataFrame.

    Parameters
    ----------
    user_inputs : dict
        Dictionary containing user inputs.
    transpose : bool, optional (default=True)
        Whether to transpose the resulting DataFrame.
    remove_trail_zeros : bool, optional (default=False)
        Whether to remove trailing zeros from numerical values.
    decimal_points : int, optional (default=2)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    DataFrame
        DataFrame representing the user inputs.

    Example
    -------
    >>> df = {'A': [1.2345, 2.3456], 'B': [3.4567, 4.5678]}
    >>> dict_df(df)
         1      2
    A  1.23  2.35
    B  3.46  4.57
    """
    if not isinstance(user_inputs, dict):
        raise TypeError(
            f"Expected 'user_inputs' to be a dictionary but got: {user_inputs}"
        )
    decimal_points = (
        8 if decimal_points == None or isinstance(decimal_points, str) 
        else decimal_points
    )
    N = DataFrame(user_inputs).values
    try:
        dframe = DataFrame(
            asarray(N.T if transpose else N)
        ).round(decimal_points)
    except TypeError:
        dframe = df_str_to_numeric(N, decimal_points)
    dframe.index = list(user_inputs.keys())
    dframe.columns = range(1, dframe.shape[1] + 1)
    dframe = (
        dframe.astype(str).replace(r"\.0$", "", regex=True) 
        if remove_trail_zeros else dframe
    )

    return dframe


def df_str_to_numeric(dframe: DataFrame, decimal_points: int = 4) -> DataFrame:
    """
    Convert pandas DataFrame with string values to numeric values 
    and round them.

    Parameters
    ----------
    dframe : DataFrame
        Pandas DataFrame containing string values.
    decimal_points : int, optional (default=4)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    numeric_df : DataFrame
        DataFrame with numeric values rounded to specified decimal 
        points.

    Example
    -------
    >>> import stemlab as stm
    >>> df = DataFrame({'A': ['1.234', '2.345'], 'B': ['3.456', '4.567']})
    >>> stm.df_str_to_numeric(df)
           A      B
    0  1.234  3.456
    1  2.345  4.567
    """
    try:
        numeric_df = dframe.map(lambda x: round(float(x), decimal_points))
    except ValueError:  # handle non-numeric values
        numeric_df = dframe.map(
            lambda x: x if isnan(float(x)) else round(float(x), decimal_points)
        )

    return numeric_df


