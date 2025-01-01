from numpy.random import randint, seed
from numpy import repeat
from pandas import DataFrame, concat

from stemlab.statistical.wrangle import dframe_labels
from stemlab.core.arraylike import is_iterable
from stemlab.core.validators.errors import InvalidInputError

def colsnames(n, blank=False):
    """
    Generate column names for a DataFrame.

    Parameters
    ----------
    n : int
        The number of columns.
    blank : bool, optional (default=False)
        If `True`, generates blank column names.

    Returns
    -------
    list
        A list containing column names.
    """
    return [[''] * n ] if blank else list(range(1, n + 1))


def dm_data_random(
    min: int = 10, 
    max: int = 100, 
    nrows: int = 10, 
    ncols: int = 5, 
    col_names: str | list | None = None, 
    index_names: str | list | None = None, 
    rand_seed: int = None
):
    """
    Generate random DataFrame.

    Parameters
    ----------
    min_value : int, optional (default=10)
        Minimum value for random data.
    max_value : int, optional (default=100)
        Maximum value for random data.
    nrows : int, optional (default=10)
        Number of rows for DataFrame.
    ncols : int, optional (default=5)
        Number of columns for DataFrame.
    col_names : {str, list_like, None}, optional (default=None)
        List of column names. If None, default column names will be 
        used.
    index_names : {str, list_like, None}, optional (default=None)
        List of index names. If None, default index names will be used.
    rand_seed : {int, None}, optional (default=None)
        Random seed for reproducibility.

    Returns
    -------
    dframe : pandas.DataFrame
        Randomly generated DataFrame.

    Examples
    --------
    >>> import stemlab as stm
    
    Note that th values obtained with the syntax below will vary 
    because they are randomly generated.

    >>> sta.dm_data_random(nrows=8, ncols=4)
        Col1  Col2  Col3  Col4
    R1    28    86    16    43
    R2    97    37    50    54
    R3    91    83    31    38
    R4    39    60    22    38
    R5    37    27    38    75
    R6    34    96    31    14
    R7    45    46    22    94
    R8    95    10    94    55

    >>> sta.dm_data_random(col_names=['A', 'B', 'C', 'D', 'E'])
          A   B   C   D   E
    R1   56  66  23  54  29
    R2   97  48  53  60  71
    R3   93  70  56  14  95
    R4   48  70  71  18  86
    R5   72  70  73  70  33
    R6   58  77  53  54  97
    R7   40  45  29  52  16
    R8   42  59  78  55  30
    R9   93  72  65  92  94
    R10  50  76  45  71  81

    >>> sta.dm_data_random(min=100, max = 800, nrows=8,
    ... col_names='ABCDE')
          A    B    C    D    E
    R1  632  318  379  324  294
    R2  769  356  710  138  644
    R3  297  620  294  157  148
    R4  639  250  646  775  162
    R5  495  747  389  553  424
    R6  723  153  755  375  411
    R7  340  362  488  356  268
    R8  605  454  419  344  579
    """
    try:
        seed(rand_seed)
    except Exception:
        seed(1234)
    M = randint(low=min, high=max, size=(nrows, ncols))
    nrows, ncols = M.shape
    col_names = (
        [f'Col{col + 1}' for col in range(ncols)] 
        if col_names is None else col_names
    )
    col_names = list(col_names) if isinstance(col_names, str) else col_names
    dframe = DataFrame(M)
    # column names
    try:
        dframe.columns = col_names
    except Exception:
        dframe.columns = (
            dframe_labels(dframe=M, prefix='Col')
        )
    # row names
    try:
        dframe.index = index_names
    except Exception:
        dframe.index = [f'R{index + 1}' for index in range(nrows)]

    return dframe


def dm_dataset_random(
    nrows: int = 10, rand_seed: int | None = None, **kwargs
) -> DataFrame:
    """
    Generate a random dataset based on specified columns.

    Parameters
    ----------
    nrows : int, optional (default=10)
        Number of rows in the dataset.
    rand_seed : {int, None}, optional (default=None)
        Seed value for random number generation.
    **kwargs : dict
        Dictionary specifying columns and their properties.

    Returns
    -------
    dframe : DataFrame
        Randomly generated dataset.

    Examples
    --------
    >>> import stemlab as stm

    Note that th values obtained with the syntax below will vary 
    because they are randomly generated.
    
    >>> sta.dm_dataset_random(age=[20, 60], grade=13)
       age  grade
    0   55     13
    1   34     13
    2   36     13
    3   50     13
    4   52     13
    5   57     13
    6   50     13
    7   22     13
    8   41     13
    9   48     13

    >>> n = 15
    >>> sta.dm_dataset_random(nrows=n, rand_seed=1234, Age=[20, 60],
    ... Gender={0: 0.6, 1: 0.4}, Education=['Bachelors', 'MSc', 'PhD'] * 5,
    ... Income=[2000, 10000], Grade=[1, 8])
        Age  Gender  Education  Income  Grade
    0    25       1        PhD    4388      5
    1    32       0        MSc    5328      4
    2    43       0  Bachelors    6199      1
    3    35       0        MSc    4130      7
    4    44       1        PhD    8905      6
    5    46       1  Bachelors    3207      5
    6    50       0        PhD    4711      2
    7    58       0        MSc    5689      1
    8    46       1        MSc    7355      4
    9    32       0        PhD    4681      5
    10   39       0  Bachelors    4482      5
    11   48       1        PhD    8478      3
    12   29       1        MSc    4393      6
    13   36       1  Bachelors    2574      7
    14   50       1  Bachelors    4359      1
    """
    try:
        seed(rand_seed)
    except Exception:
        seed(8765)

    columns_ = kwargs
    if len(columns_) == 0:
        raise ValueError('Specify at least one column to be generated')
    dframe = DataFrame([])
    for k, v in columns_.items():
        if is_iterable(v):
            try:
                M = randint(low=min(v), high=max(v) + 1, size=(nrows, 1))
            except Exception:
                if isinstance(v, str) or len(v) == 1:
                    M = repeat(v, nrows)
                else:
                    if len(v) == nrows:
                        M = v
                    else:
                        raise Exception(
                            f'Expected {k} to have {nrows} elements, '
                            f'but got {len(v)}'
                        )
        elif isinstance(v, dict):
            M = repeat(list(v.keys()), list(v.values()))
        elif isinstance(v, (int, float)):
            M = repeat(v, nrows)
        else:
            raise InvalidInputError(par_name=k, user_input=v)
        M = DataFrame(M, columns=[k])
        # reshaffle the values, particularly userful for replicate
        M.index = randint(low=1, high=nrows * 10000, size=(1, nrows)).flatten()
        M = M.sort_index()
        M.index = range(nrows)
        dframe = concat([dframe, M], axis=1)

    return dframe