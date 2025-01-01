from datetime import datetime


def datetime_now(
    value: str = 'both', date_format: str = '%Y-%m-%d %H:%M:%S'
) -> str:
    """
    Get the current date and time in the specified format.

    Parameters
    ----------
    value : str, optional (default='both')
        Determines what part of the datetime to return. Possible values are:
        - 'both' (default): Returns both date and time.
        - 'date': Returns only the date.
        - 'time': Returns only the time.
    date_format : str, optional (default='%Y-%m-%d %H:%M:%S')
        The format string to use for the date and time representation.

    Returns
    -------
    date_time : str
        The current date and time in the specified format.

    Raises
    ------
    ValueError
        If an invalid value for `value` is provided.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.datetime_now()
    '2024-04-03 12:34:56'
    >>> stm.datetime_now(value='date')
    '2024-04-03'
    >>> stm.datetime_now(value='time')
    '12:34:56'
    """
    date_time = datetime.now().strftime(date_format)
    
    if value == 'date':
        date_time = date_time.split(" ")[0]
    elif value == 'time':
        date_time = date_time.split(" ")[1]
    elif value != 'both':
        raise ValueError(
            "Invalid value for 'value'. Use 'date', 'time', or 'both'."
        )

    return date_time


def datetime_now_number() -> str:
    """
    Get the current date and time in a numeric format.

    Returns
    -------
    str
        The current date and time in a numeric format, with spaces, 
        colons, and hyphens removed.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.datetime_now_number()
    '20240403123456'
    """
    return datetime.now().strftime("%Y%m%d%H%M%S")


def extract_date(date_string: str) -> str:
    """
    Extract the date portion from a datetime string.

    Parameters
    ----------
    date_string : str
        The datetime string to extract the date from.

    Returns
    -------
    formatted_datetime : str
        The extracted date in the format "YYYY-MM-DD".

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.extract_date("2024-04-03 12:34:56.789+0000")
    '2024-04-03'
    """
    datetime_object = datetime.strptime(str(date_string), "%Y-%m-%d %H:%M:%S.%f%z")
    formatted_datetime = datetime_object.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime