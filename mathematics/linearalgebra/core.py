from numpy import inf


def la_norm_order(order: str = 'fro') -> str | int | None:
    """
    Convert a string representation of a norm order to its 
    corresponding numerical value.

    Parameters
    ----------
    order : str, optional (default='fro')
        String representation of the norm order.

    Returns
    -------
    {str, int, None}
        Corresponding numerical value of the norm order or `fro`.
        Returns None if the input order is invalid.

    Examples
    --------
    >>> import stemlab as stm

    >>> stm.la_norm_order('euclidean-norm')
    2

    >>> stm.la_norm_order('inf')
    inf

    >>> stm.la_norm_order('minimum-absolute-row-sum')
    -inf

    >>> stm.la_norm_order('-1')
    -1

    >>> stm.la_norm_order('largest-singular-value')
    2

    >>> stm.la_norm_order('10')
    (no output because it is None)
    """
    order = str(order).lower()

    vector_norms = {
        'none': None,
        'manhattan-norm': 1,
        'euclidean-norm': 2,
        'maximum-norm': inf
    }

    matrix_norms = {
        'fro': 'fro',
        'frobenius-norm': 'fro',
        '-inf': -inf,
        'minimum-absolute-row-sum': -inf,
        'inf': inf,
        'maximum-absolute-row-sum': inf,
        '-1': -1,
        'minimum-absolute-column-sum': -1,
        '1': 1,
        'maximum-absolute-column-sum': 1,
        '-2': -2,
        'smallest-singular-value': -2,
        '2': 2,
        'largest-singular-value': 2
    }
    
    norms = vector_norms | matrix_norms
    try:
        if order in norms.keys():
            return norms.get(order, None)
    except Exception:
        return None