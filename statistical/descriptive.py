from typing import Literal

from scipy.stats import mode, norm
from pandas import Series, DataFrame, crosstab
from numpy import equal, round, array, where, zeros, nan, sqrt

from stemlab.core.arraylike import (
    is_iterable, conv_list_to_dict, conv_to_arraylike
)
from stemlab.core.validators.errors import (
    IterableError, PandifyError, SerifyError
)
from stemlab.statistical.wrangle import dframe_labels
from stemlab.core.decimals import fround
from stemlab.core.display import Result
from stemlab.core.validators.validate import ValidateArgs
from stemlab.core.datatypes import ListArrayLike


LOCATION_MEASURES = ['n', 'mean', 'median', 'mode', 'sum']
DISPERSION_MEASURES = [
    'min', 'max', 'range', 'var', 'std', 'cv', 'sem', 'p25', 'p75', 'iqr'
]
DISTRIBUTION_MEASURES = ['skewness', 'kurtosis']
DESCRIPTIVE_MEASURES = (
    LOCATION_MEASURES + DISPERSION_MEASURES + DISTRIBUTION_MEASURES
)

VALID_STATISTICS = [
    'location', 'dispersion', 'distribution',
    'n', 'count', 'mean', 'median', 'mode', 'sum',
    'min', 'max', 'range', 'var', 'std',
    'sem', 'cv', 'p25', 'q1', 'p75', 'q3', 'iqr',
    'skew','skewness', 'kurt', 'kurtosis'
]

   
def get_samples_stats(kwargs, is_stats: bool = False):
    """
    Validate sample1 and sample2.
    """
    valid_values = (
        ['n1', 'std1', 'n2', 'std2'] if is_stats else ['sample1', 'sample2']
    )
    ValidateArgs.check_member_of(
        par_name='kwargs',
        user_input=list(kwargs),
        valid_values=valid_values
    )
    if is_stats:
        n1 = ValidateArgs.check_numeric(
            par_name='n1',
            is_positive=True,
            is_integer=True,
            user_input=kwargs.get('n1')
        )
        std1 = ValidateArgs.check_numeric(
            par_name='std1',
            is_positive=True,
            is_integer=False,
            user_input=kwargs.get('std1')
        )
        n2 = ValidateArgs.check_numeric(
            par_name='n2',
            is_positive=True,
            is_integer=True,
            user_input=kwargs.get('n2')
        )
        std2 = ValidateArgs.check_numeric(
            par_name='std2',
            is_positive=True,
            is_integer=False,
            user_input=kwargs.get('std2')
        )
    else:
        sample1 = ValidateArgs.check_series(
            par_name='sample1',
            is_dropna=True,
            user_input=kwargs.get('sample1')
        )
        sample2 = ValidateArgs.check_series(
            par_name='sample2',
            is_dropna=True,
            user_input=kwargs.get('sample2')
        )
        n1, n2 = len(sample1), len(sample2)
        std1, std2 = Series(sample1).std(), Series(sample2).std()
    
    return n1, std1, n2, std2


def eda_pooled_variance_sample_stats(is_stats: bool = False, **kwargs) -> float:
    """
    Calculate the pooled variance for two samples.

    The pooled variance is a weighted average of the variances from
    two independent samples, assuming that the two samples have equal
    variances.

    Parameters
    ----------
    is_stats : bool, optional (default=False)
        If `True`, the function expects sample sizes and standard
        deviations as inputs.
    **kwargs
        Keyword arguments that include sample sizes and standard
        deviations. Expected keys are 'n1', 'std1', 'n2', 'std2' when
        `is_stats` is `True` and `sample1`, `sample2` otherwise.

    Returns
    -------
    pooled_var : float
        The calculated pooled variance.

    Notes
    -----
    The pooled variance is used in situations where the assumption of
    equal variances between the two samples is reasonable.

    Examples
    --------
    >>> import stemlab as stm
    >>> from stemlab.statistical.descriptive import eda_pooled_variance_sample_stats
    >>> df = stm.dataset_read(name='scores')
    >>> female = df.loc[df['gender'] == 'Female', 'score_after']
    >>> male = df.loc[df['gender'] == 'Male', 'score_after']
    
    Given samples
    
    >>> dfn = eda_pooled_variance_sample_stats(is_stats=False, 
    ... sample1=female, sample2=male)
    >>> print(dfn)
    
    Given statistics
    
    >>> n1, std1 = (13, 14.543921102152648)
    >>> n2, std2 = (17, 13.838883838025122)
    >>> dfn = eda_pooled_variance_sample_stats(is_stats=True,
    ... n1=n1, std1=std1, n2=n2, std2=std2)
    >>> print(dfn)
    """
    n1, std1, n2, std2 = get_samples_stats(is_stats=is_stats, kwargs=kwargs)
    pooled_var = ((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2)
    
    return pooled_var


def eda_pooled_variance(
    sample1: ListArrayLike, sample2: ListArrayLike
) -> float:
    """
    Calculate the pooled variance for two samples.

    The pooled variance is a weighted average of the variances from
    two independent samples, assuming that the two samples have equal
    variances.

    Parameters
    ----------
    sample1 : ListArrayLike
        The first sample, which can be a list or a NumPy array of
        floats.
    sample2 : ListArrayLike
        The second sample, which can be a list or a NumPy array of
        floats.

    Returns
    -------
    pooled_var : float
        The calculated pooled variance.

    Notes
    -----
    The pooled variance is used when the assumption of equal variances
    between the two samples is reasonable.

    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')
    >>> female = df.loc[df['gender'] == 'Female', 'score_after']
    >>> male = df.loc[df['gender'] == 'Male', 'score_after']
    
    >>> dfn = sta.eda_pooled_variance(sample1=female, sample2=male)
    >>> print(dfn)
    """
    pooled_var = eda_pooled_variance_sample_stats(
        is_stats=False, sample1=sample1, sample2=sample2
    )
    return pooled_var


def eda_pooled_variance_stats(
    n1: int, std1: float, n2: int, std2: float
) -> float:
    """
    Calculate the pooled variance using sample statistics.

    The pooled variance is a weighted average of the variances from
    two independent samples, assuming that the two samples have equal
    variances. This function uses sample sizes and standard deviations
    to calculate the pooled variance.

    Parameters
    ----------
    n1 : int
        The sample size of the first sample.
    std1 : float
        The standard deviation of the first sample.
    n2 : int
        The sample size of the second sample.
    std2 : float
        The standard deviation of the second sample.

    Returns
    -------
    pooled_var : float
        The calculated pooled variance.

    Notes
    -----
    The pooled variance is used when the assumption of equal variances
    between the two samples is reasonable.

    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> n1, std1 = (13, 14.543921102152648)
    >>> n2, std2 = (17, 13.838883838025122)
    >>> dfn = sta.eda_pooled_variance_stats(n1, std1, n2, std2)
    >>> print(dfn)
    """
    pooled_var = eda_pooled_variance_sample_stats(
        is_stats=True, n1=n1, std1=std1, n2=n2, std2=std2
    )
    return pooled_var


def eda_standard_error_sample_stats(
    is_stats: bool = False, is_pooled: bool = False, **kwargs
) -> float:
    """
    Calculate the standard error of the difference between two means.

    This function calculates the standard error either using pooled
    variance or separately calculated variances, depending on the
    `is_pooled` parameter.

    Parameters
    ----------
    is_stats : bool, optional
        If `True`, the function expects sample sizes and standard
        deviations as inputs. Default is False.
    is_pooled : bool, optional
        If `True`, the pooled variance is used to calculate the standard
        error. Default is False.
    **kwargs
        Keyword arguments that include sample sizes and standard
        deviations when `is_stats` is True, or sample data when 
        `is_stats` is False.

    Returns
    -------
    std_error : float
        The calculated standard error of the difference between the two
        means.

    Notes
    -----
    The standard error is a measure of how much the sample mean
    difference is expected to vary from the true population mean
    difference. The pooled variance should be used when the assumption
    of equal variances between the two samples is reasonable.

    Examples
    --------
    >>> import stemlab as stm
    >>> from stemlab.statistical.descriptive import eda_standard_error_sample_stats
    >>> df = stm.dataset_read(name='scores')
    >>> female = df.loc[df['gender'] == 'Female', 'score_after']
    >>> male = df.loc[df['gender'] == 'Male', 'score_after']
    
    Given samples
    
    >>> dfn = eda_standard_error_sample_stats(is_stats=False, 
    ... is_pooled=False, sample1=female, sample2=male)
    >>> print(dfn)
    
    Given statistics
    
    >>> n1, std1 = (13, 14.543921102152648)
    >>> n2, std2 = (17, 13.838883838025122)
    >>> dfn = eda_standard_error_sample_stats(is_stats=True,
    ... is_pooled=False, n1=n1, std1=std1, n2=n2, std2=std2)
    >>> print(dfn)
    """
    n1, std1, n2, std2 = get_samples_stats(is_stats=is_stats, kwargs=kwargs)
    if is_pooled:
        if is_stats:
            pooled_var = eda_pooled_variance_stats(
                n1=n1, std1=std1, n2=n2, std2=std2
            )
        else:
            pooled_var = eda_pooled_variance(
                sample1=kwargs.get('sample1'), sample2=kwargs.get('sample2')
            )
        std_error = sqrt(pooled_var * (1 / n1 + 1 / n2))
    else:
        std_error = sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)
    
    return std_error


def eda_standard_error(
    sample1: ListArrayLike, sample2: ListArrayLike, is_pooled=False
) -> float:
    """
    Calculate the standard error of the difference between two means.

    This function calculates the standard error using either pooled
    variance or separately calculated variances, depending on the
    `is_pooled` parameter.

    Parameters
    ----------
    sample1 : ListArrayLike
        The first sample, which can be a list or a NumPy array of
        numeric values.
    sample2 : ListArrayLike
        The second sample, which can be a list or a NumPy array of
        numeric values.
    is_pooled : bool, optional
        If `True`, the pooled variance is used to calculate the standard
        error. Default is False.

    Returns
    -------
    std_error : float
        The calculated standard error of the difference between the
        two means.

    Notes
    -----
    The standard error is a measure of how much the sample mean
    difference is expected to vary from the true population mean
    difference. The pooled variance should be used when the
    assumption of equal variances between the two samples is
    reasonable.

    Examples
    --------
    >>> import stemlab as stm
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')
    >>> female = df.loc[df['gender'] == 'Female', 'score_after']
    >>> male = df.loc[df['gender'] == 'Male', 'score_after']
    
    Given samples
    
    >>> dfn = sta.eda_standard_error(is_pooled=False,
    ... sample1=female, sample2=male)
    >>> print(dfn)
    
    Given statistics
    
    >>> n1, std1 = (13, 14.543921102152648)
    >>> n2, std2 = (17, 13.838883838025122)
    >>> dfn = eda_standard_error_stats(is_pooled=False,
    ... n1=n1, std1=std1, n2=n2, std2=std2)
    >>> print(dfn)
    """
    std_error = eda_standard_error_sample_stats(
        is_stats=False, is_pooled=is_pooled, sample1=sample1, sample2=sample2
    )
    return std_error


def eda_standard_error_stats(
    n1: int, std1: float, n2: int, std2: float, is_pooled=False
) -> float:
    """
    Calculate the standard error of the difference between two means
    using sample statistics.

    This function calculates the standard error using either pooled
    variance or separately calculated variances, depending on the
    `is_pooled` parameter. It takes the sample sizes and standard
    deviations as inputs.

    Parameters
    ----------
    n1 : int
        The sample size of the first sample.
    std1 : float
        The standard deviation of the first sample.
    n2 : int
        The sample size of the second sample.
    std2 : float
        The standard deviation of the second sample.
    is_pooled : bool, optional
        If `True`, the pooled variance is used to calculate the standard
        error. Default is False.

    Returns
    -------
    std_error : float
        The calculated standard error of the difference between the
        two means.

    Notes
    -----
    The standard error is a measure of how much the sample mean
    difference is expected to vary from the true population mean
    difference. The pooled variance should be used when the
    assumption of equal variances between the two samples is
    reasonable.

    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> n1, std1 = (13, 14.543921102152648)
    >>> n2, std2 = (17, 13.838883838025122)
    >>> dfn = sta.eda_standard_error_sample_stats(is_pooled=False,
    ... n1=n1, std1=std1, n2=n2, std2=std2)
    >>> print(dfn)
    """
    std_error = eda_standard_error_sample_stats(
        is_stats=True, is_pooled=is_pooled, n1=n1, std1=std1, n2=n2, std2=std2
    )
    return std_error


def degrees_of_freedom(
    is_stats: bool = False, is_welch: bool = True, **kwargs
) -> float:
    """
    Calculate the degrees of freedom for a two-sample t-test.

    The degrees of freedom can be calculated using either the 
    Satterthwaite or Welch approximation based on the provided sample 
    statistics.

    Parameters
    ----------
    is_stats : bool, optional (default=False)
        If `True`, the function expects sample sizes and standard 
        deviations as inputs.
    is_welch : bool, optional (default=True)
        If `True`, Welch's approximation is used; otherwise, the 
        Satterthwaite approximation is used.
    **kwargs
        Keyword arguments that include sample sizes and standard 
        deviations. Expected keys are 'n1', 'std1', 'n2', 'std2' when 
        `is_stats` is True, and `sample1` and `sample2` otherwise.

    Returns
    -------
    dfn : float
        The calculated degrees of freedom.

    Notes
    -----
    - If `is_welch` is True, Welch's approximation is used, which is 
      suitable for cases with unequal variances and sample sizes.
    - If `is_welch` is False, Satterthwaite's approximation is used, 
      which is suitable for cases where the variances are unequal but 
      sample sizes are approximately equal.

    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')
    >>> female = df.loc[df['gender'] == 'Female', 'score_after']
    >>> male = df.loc[df['gender'] == 'Male', 'score_after']
    
    Welch's degrees of freedom
    
    >>> dfn = sta.degrees_of_freedom(is_stats=False, is_welch=True,
    ... sample1=female, sample2=male)
    >>> print(dfn)
    27.207532573005
    
    Satterthwaite's degrees of freedom
    
    >>> dfn = sta.degrees_of_freedom(is_stats=False, is_welch=False,
    ... sample1=female, sample2=male)
    >>> print(dfn)
    25.28023085138802
    """
    n1, std1, n2, std2 = get_samples_stats(is_stats=is_stats, kwargs=kwargs)
    if is_welch:
        dfn = (
            (((std1 ** 2 / n1 + std2 ** 2 / n2) ** 2) /
             ((std1 ** 2 / n1) ** 2 / (n1 + 1) +
              (std2 ** 2 / n2) ** 2 / (n2 + 1))) - 2
        )
    else:
        dfn = (
            ((std1 ** 2 / n1 + std2 ** 2 / n2) ** 2) /
            (((std1 ** 2 / n1) ** 2) / (n1 - 1) +
             ((std2 ** 2 / n2) ** 2) / (n2 - 1))
        )
        
    return dfn


def df_welch(
    sample1: ListArrayLike,
    sample2: ListArrayLike
) -> float:
    """
    Calculate Welch's degrees of freedom.

    Welch's approximation is used to estimate the degrees of freedom 
    for a two-sample t-test when the variances of the two samples are 
    unequal and the sample sizes may also be different.

    Parameters
    ----------
    sample1 : ListArrayLike
        The first sample, which can be a list or a NumPy array of 
        floats.
    sample2 : ListArrayLike
        The second sample, which can be a list or a NumPy array of 
        floats.

    Returns
    -------
    dfn : float
        The estimated degrees of freedom using Welch's approximation.

    Notes
    -----
    This method assumes that the two samples are independent and that
    the variances are unequal.
    
    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')
    >>> female = df.loc[df['gender'] == 'Female', 'score_after']
    >>> male = df.loc[df['gender'] == 'Male', 'score_after']
    >>> dfn = sta.df_welch(sample1=female, sample2=male)
    >>> print(dfn)
    27.207532573005
    """
    dfn = degrees_of_freedom(
        is_stats=False, is_welch=True, sample1=sample1, sample2=sample2
    )

    return dfn


def df_welch_stats(n1: int, std1: float, n2: int, std2: float) -> float:
    """
    Calculate Welch's degrees of freedom using sample sizes and
    standard deviations.

    Welch's approximation is used to estimate the degrees of freedom
    for a two-sample t-test when the variances of the two samples are
    unequal and the sample sizes may also be different.

    Parameters
    ----------
    n1 : int
        The size of the first sample.
    std1 : float
        The standard deviation of the first sample.
    n2 : int
        The size of the second sample.
    std2 : float
        The standard deviation of the second sample.

    Returns
    -------
    dfn : float
        The estimated degrees of freedom using Welch's approximation.

    Notes
    -----
    This method assumes that the two samples are independent and that
    the variances are unequal.
    
    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> n1, std1 = (13, 14.543921102152648)
    >>> n2, std2 = (17, 13.838883838025122)
    >>> df = sta.df_welch_stats(n1, std1, n2, std2)
    >>> print(df)
    27.207532573005
    """
    dfn = degrees_of_freedom(
        is_stats=True, is_welch=True, n1=n1, std1=std1, n2=n2, std2=std2
    )
    
    return dfn


def df_satterthwaite(
    sample1: ListArrayLike,
    sample2: ListArrayLike
) -> float:
    """
    Calculate Satterthwaite's degrees of freedom.

    The Satterthwaite approximation is used to estimate the degrees
    of freedom for a two-sample t-test when the variances of the two
    samples are unequal.

    Parameters
    ----------
    sample1 : ListArrayLike
        The first sample, which can be a list or a NumPy array of
        floats.
    sample2 : ListArrayLike
        The second sample, which can be a list or a NumPy array of
        floats.

    Returns
    -------
    dfn : float
        The estimated degrees of freedom using the Satterthwaite
        approximation.

    Notes
    -----
    This method assumes that the two samples are independent and that
    the variances are unequal.

    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')
    >>> female = df.loc[df['gender'] == 'Female', 'score_after']
    >>> male = df.loc[df['gender'] == 'Male', 'score_after']
    >>> dfn = sta.df_satterthwaite(sample1=female, sample2=male)
    >>> print(dfn)
    25.28023085138802
    """
    dfn = degrees_of_freedom(
        is_stats=False, is_welch=False, sample1=sample1, sample2=sample2
    )

    return dfn


def df_satterthwaite_stats(
    n1: int, std1: float, n2: int, std2: float
) -> float:
    """
    Calculate Satterthwaite's degrees of freedom using sample sizes
    and standard deviations.

    The Satterthwaite approximation is used to estimate the degrees
    of freedom for a two-sample t-test when the variances of the two
    samples are unequal, based on their sample sizes and standard
    deviations.

    Parameters
    ----------
    n1 : int
        The size of the first sample.
    std1 : float
        The standard deviation of the first sample.
    n2 : int
        The size of the second sample.
    std2 : float
        The standard deviation of the second sample.

    Returns
    -------
    dfn : float
        The estimated degrees of freedom using the Satterthwaite
        approximation.

    Notes
    -----
    This method assumes that the two samples are independent and that
    the variances are unequal.

    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> n1, std1 = (13, 14.543921102152648)
    >>> n2, std2 = (17, 13.838883838025122)
    >>> df = sta.df_satterthwaite_stats(n1, std1, n2, std2)
    >>> print(df)
    25.28023085138802
    """
    dfn = degrees_of_freedom(
        is_stats=True, is_welch=False, n1=n1, std1=std1, n2=n2, std2=std2
    )

    return dfn


def dm_unique_cat(
    data, col_names: str | list = 'object', dict_cat: bool = False
) -> None:
    """
    Return unique categories for specified columns in a DataFrame.
    
    Parameter
    ---------
    data : ArrayLike
        Data whose with distinct categories.
    col_names : {str, list}, optional (default='object')
        Column(s) whose categories we need. If object, then all 
        columns with object data type are considered.
    dict_cat: bool, optional (default=False)
        If `True`, then a dictionary of the unique categories for 
        each of the specified columns will be shown. Otherwise, 
        results will be displayed on the screen.
    
    Returns
    -------
    dict or None
        If `dict_cat` is `True`, returns a dictionary where keys are
        column names and values are arrays of unique categories. If
        `dict_cat` is `False`, prints the unique categories for each
        specified column and returns `None`.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import stemlab.statistical as sta
    >>> data = {'col1': ['a', 'b', 'a', 'c'],
    ... 'col2': ['x', 'y', 'x', 'z']}
    >>> df = pd.DataFrame(data)
    >>> df
      col1 col2
    0    a    x
    1    b    y
    2    a    x
    3    c    z
    
    >>> sta.dm_unique_cat(df)
    col1
    ----
    ['a', 'b', 'c']
    
    col2
    ----
    ['x', 'y', 'z']
    
    >>> sta.dm_unique_cat(df, dict_cat=True)
    {'col1': array(['a', 'b', 'c'], dtype=object),
     'col2': array(['x', 'y', 'z'], dtype=object)}
    
    >>> sta.dm_unique_cat(df, col_names=['col1'], dict_cat=True)
    {'col1': array(['a', 'b', 'c'], dtype=object)}
    
    >>> sta.dm_unique_cat(df, col_names=['col1'])
    col1
    ----
    ['a', 'b', 'c']
    """
    if not isinstance(data, DataFrame):
        try:
            dframe = DataFrame(data).map(lambda col: str(col))
        except Exception:
            raise IterableError(par_name='data', strng=False, user_input=data)
    else:
        if col_names == 'object':
            dframe = data.select_dtypes(include = ['object'])
        elif is_iterable(array_like=col_names, includes_str=True):
            col_names = conv_to_arraylike(
                array_values=col_names, 
                includes_str=True,
                par_name='array_values'
            )
            dframe = data[col_names]
            
    dict_cat = ValidateArgs.check_boolean(user_input=dict_cat, default=False)
    
    if dict_cat:
        dict_values = {}
        for col in dframe.columns:
            dict_values.update({col: dframe[col].unique()})
        return dict_values
    else:
        dframe = DataFrame(dframe)
        for col in dframe.columns:
            print(f'{col}\n{"-" * len(col)}')
            print(f'{dframe[col].unique()[:50].tolist()}\n')
        return None
    
    
def _proportions(freq_table, dim):

    """
    Proportions test.
    """

    if dim.startswith('r'):
        freq_table = freq_table.div(freq_table["Total"], axis=0)
    elif dim.startswith('c'):
        freq_table = freq_table / freq_table.loc["Total"]
    elif dim.startswith('b'):
        freq_table = freq_table / freq_table.loc["Total", "Total"]
    else:
        raise ValueError(f"{dim} is an invalid option for 'dim'")

    return freq_table


def eda_freq_tables(
    data, 
    col_names, 
    cells: Literal[
        'counts', 'percent', 'proportions', 'counts(percent)'
    ] = 'counts(percent)', 
    dim: Literal['rows', 'columns', 'both'] = 'row', 
    columns: ListArrayLike | None = None, 
    index: ListArrayLike | None = None, 
    decimal_points: int = 0
) -> DataFrame:
    """
    Generate one-way and two-way frequency tables.

    Parameters
    ----------
    data : {DataFrame, ArrayLike}
        The data to be used for generating the frequency table. Must be 
        convertible to a pandas DataFrame.
    col_names : str or list of str
        Column name(s) to be used for generating the frequency table. If 
        a list is provided, it should contain at most two elements.
    cells : {'counts', 'percent', 'proportions', 'counts(percent)'}, optional (default='counts(percent)')
        Specifies the type of cells to be included in the table:
        - 'counts': only counts.
        - 'percent': only percentages.
        - 'proportions': only proportions.
        - 'counts(percent)': counts with percentages in parentheses.
    dim : {'rows', 'columns', 'both'}, optional (default='rows')
        Specifies how percentages or proportions should be calculated 
        in a two-way table:
        - 'rows': percentages/proportions by row.
        - 'columns': percentages/proportions by column.
        - 'both': percentages/proportions by both row and column.
    columns : {ListLike}, optional (default=None)
        The column names for the resulting table.
    index : {ListLike}, optional(default=None)
        The index names for the resulting table.
    decimal_points : int, optional (default=2)
        The number of decimal points to round the frequency values.
    
    Returns
    -------
    dframe : pandas.DataFrame
        A pandas DataFrame containing the frequency table with the specified 
        format.

    Raises
    ------
    PandifyError
        If 'data' cannot be converted to a pandas DataFrame.
    ValueError
        If a column name in 'col_names' is not present in the data.
        If 'col_names' contains more than two elements.
    TypeError
        If 'col_names' is not a string, list, or tuple.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='nhanes')
    >>> sta.eda_freq_tables(df, col_names='region')
                Frequency Cum. Frequency
    North East       2086           2086
    North West       2773           4859
    South            2853           7712
    West             2625          10337
    Total           10337               
    
    >>> sta.eda_freq_tables(df, col_names='region', cells='percent')
                Frequency  Percent  Cum. Percent
    North East       2086     20.0          20.0
    North West       2773     27.0          47.0
    South            2853     28.0          75.0
    West             2625     25.0         100.0
    Total           10337    100.0           NaN
    
    >>> sta.eda_freq_tables(df, col_names=['race', 'highbp'])
                   No         Yes         Total
    White  5307 (59%)  3744 (41%)   9051 (100%)
    Black   545 (50%)   541 (50%)   1086 (100%)
    Other   113 (56%)    87 (44%)    200 (100%)
    Total  5965 (58%)  4372 (42%)  10337 (100%)
    
    >>> sta.eda_freq_tables(df, col_names=['race', 'highbp'], dim='columns')
                    No          Yes         Total
    White   5307 (89%)   3744 (86%)    9051 (88%)
    Black     545 (9%)    541 (12%)    1086 (11%)
    Other     113 (2%)      87 (2%)      200 (2%)
    Total  5965 (100%)  4372 (100%)  10337 (100%)
    """
    decimal_points = (
        2 if decimal_points == 0 and 'pr' in cells else decimal_points
    )
    try:
        dframe = DataFrame(data)
    except Exception:
        raise PandifyError(par_name='data')

    if isinstance(col_names, str):
        col_names = [col_names]

    if not isinstance(col_names, (tuple, list)):
        raise TypeError(
            f"Expected '{col_names} to be a list with at most two elements "
            f"representing the variables to be tabulated but got {col_names}"
        )

    if len(col_names) > 2:
        raise ValueError(
            f"Expected 'col_names' to have atmost 2 elements but got "
            f"{len(col_names)}"
        )

    for col_name in col_names:
        if col_name not in dframe.columns:
            raise ValueError(
                f"'{col_name}' is not one of the columns of the dataset. "
                f"Valid columns are: {', '.join(map(str, data.columns))}"
            )

    if len(col_names) == 1: # one way frequency table
        if "co" in cells or "Fr" in cells:
            cells = "Frequency"
        freq_table = crosstab(
            dframe[col_names[0]], 
            columns="Frequency", 
            margins=True, 
            margins_name="Total"
        )
        row_names = list(freq_table.index)
        freq_values = freq_table.values
        result = DataFrame(
            freq_values[:, 0], index=row_names, columns=["Frequency"]
        )
        if "pr" in cells:
            cells = "proportions"
            percent_prop = freq_values[:, 0] / sum(freq_values[-1:, 0])
        elif "pe" in cells:
            cells = "percent"
            percent_prop = (
                freq_values[:, 0] / sum(freq_values[-1:, 0]) * 100
            )
        else:
            percent_prop = freq_values[:, 0]
        cum_percent = [percent_prop[0].tolist()]
        for k in range(1, len(percent_prop)):
            cum_percent.append(cum_percent[k - 1] + percent_prop[k])
        cum_percent[-1] = nan
        if "co" not in cells:
            result[f"{cells.capitalize()}"] = percent_prop
        result[f"Cum. {cells.capitalize()}"] = cum_percent
        try:
            result = result.round(decimal_points)
            if "Fr" in cells:
                result["Cum. Frequency"] = [
                    int(value) for value in result["Cum. Frequency"][:-1]
                ] + [""]
        except Exception:
            pass
        dframe = result
    else: # two-way frequency table
        freq_table = crosstab(
            index=dframe[col_names[0]],
            columns=dframe[col_names[1]],
            margins=True,
            margins_name="Total",
        )
        row_names = list(freq_table.index)
        col_names = list(freq_table.columns)
        if "pr" in cells:
            freq_table = _proportions(freq_table, dim)
        elif cells == "percent":
            freq_table = _proportions(freq_table, dim) * 100
        elif "(" in cells or ")" in cells or "count(percent)" in cells:
            freq_array = array(freq_table)
            freq_percent = round(
                array(_proportions(freq_table, dim) * 100), decimal_points
            )
            rows, cols = freq_table.shape
            freq_counts_percent = zeros((rows, cols), dtype=object)
            for row in range(rows):
                for col in range(cols):
                    rowcol_count = round(freq_array[row, col], decimal_points)
                    rowcol_percent = round(freq_percent[row, col], decimal_points)
                    freq_counts_percent[
                        row, col
                    ] = f"{rowcol_count} ({rowcol_percent}%)"
            freq_table = DataFrame(freq_counts_percent)
        else:
            pass

        try:
            freq_table = round(freq_table, decimal_points)
        except Exception:
            pass

        dframe = DataFrame(
            freq_table.values, index=row_names, columns=col_names
        )
    # row and column names
    try:
        dframe.index = index
        dframe.columns = columns
    except Exception:
        pass
    dframe = dframe.replace('.0%', '%', regex=True)
    dframe = dframe.round(decimal_points)

    return dframe



def eda_mode_series(data: list) -> tuple:
    """
    Returns the mode and the number of times it appears.

    Parameters
    ----------
    data : list-like
        An array_like with the values for which we need the mode.
    
    Returns
    -------
    mode_freq_tuple : tuple
        The mode and how many times it appears, or None if there is no 
        mode.

    Examples
    --------
    >>> import stemlab as stm
    >>> sta.eda_mode_series([2, 1, 2, 2, 3, 2, 4])
    (2, 4)

    The following is bimodal (i.e. 1 and 4), the function will return 
    the first mode - in this case 1
    
    >>> sta.eda_mode_series([4, 1, 4, 1, 1, 3, 2, 2, 4])
    (1, 3)

    >>> sta.eda_mode_series([1, 2, 3, 4, 5])
    (None, None)

    """
    try:
        data = Series(data)
    except Exception:
        raise SerifyError(par_name='data')
    mode_cat = data.mode().values[0]
    mode_table = crosstab(data, columns='Frequency')
    mode_freq = mode_table.sort_values(by='Frequency', ascending=False)
    mode_freq = mode_freq.values[0][0]
    if mode_freq == 1: # there is no mode
        mode_freq, mode_cat = None, None
    mode_freq_tuple = (mode_cat, mode_freq)

    return mode_freq_tuple


def eda_mode_freq(
    data: DataFrame, 
    columns: list[str] = [], 
    index_labels: str | int | None = None
) -> float:
    """
    Returns the mode and the number of times it appears for specified 
    columns in a DataFrame.
    
    Parameters
    ----------
    data : {ListLike, DataFrame}
        Array-like with values/columns for which we need the mode.
    columns : list, optional (default=[])
        The list of column names for which to calculate the mode. 
        If empty, all columns are considered.
    index_labels : list, optional (default=None)
        Custom labels representing the index (row names) of the 
        resulting DataFrame.
    
    Returns
    -------
    dframe_mode : pandas.DataFrame
        A DataFrame with the mode and how many times it appears, 
        or None if there is no mode.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='nhanes')
    >>> df_category = df.select_dtypes(['category'])
    >>> sta.eda_mode_freq(data=df_category, index_labels='title')
                 Mode Frequency
    Region      South      2853
    Highbp         No      5965
    Sex        Female      5428
    Race        White      9051
    Agegroup    60-69      2852
    Heartatk       No      9862
    Diabetes       No      9836
    Highlead       No      4649
    Health    Average      2938
    """
    dframe = ValidateArgs.check_dframe(par_name='data', user_input=data)
    columns = conv_to_arraylike(array_values=columns, includes_str=True)
    index_labels = ValidateArgs.check_dflabels(
        par_name='index_labels', user_input=index_labels
    )
    try:
        dframe = (dframe[columns] if columns else dframe)
    except Exception as e:
        raise e
    # apply the series function on the DataFrame columns
    dframe_mode = dframe.apply(lambda col: eda_mode_series(data=col)).T
    dframe_mode.columns = ['Mode', 'Frequency']
    dframe_mode.index = dframe_labels(
        dframe=dframe_mode, df_labels=index_labels
    )

    return dframe_mode


def eda_tabstat(
    dframe: DataFrame,
    columns: list[str] = [],
    statistics: Literal[
        'location', 'dispersion', 'distribution',
        'n', 'count', 'mean', 'median', 'mode', 'sum', 'min',
        'max', 'range', 'var', 'std', 'sem', 'cv', 'p25', 'q1',
        'p75', 'q3', 'iqr', 'skew', 'skewness', 'kurt', 'kurtosis'
    ] = 'location',
    decimal_points=4
) -> float:
    """
    Tabulate statistics for a DataFrame.

    Parameters
    ----------
    dframe : DataFrame
        Input DataFrame.
    statistics : str, optional (default='location')
        Type of statistics to compute. Possible values are:
        ==========================================================
        Statistic                 Description  
        ==========================================================
        location ................ Mean, median, mode
        dispersion .............. All dispersion measures
        distribution ............ Skewness, kurtosis
        n or count .............. Frequency
        mean .................... Mean
        cimean .................. Confidence interval for the mean
        median .................. Median
        mode .................... Mode
        sum ..................... Sum
        min ..................... Minimum
        max ..................... Maximum
        range ................... Range (Max - Min)
        var ..................... Variance
        std ..................... Standard deviation
        sem ..................... Standard error of the mean
        cv ...................... Coefficient of variation
        p25 or q1 ............... 25th percentile (First quartile)
        p75 or q3 ............... 75th percentile (Third quartile)
        iqr ..................... Interquartile range (Q3 - Q1)
        skew or skewness.........  Skewness
        kurt or kurtosis ........  Kurtosis
        ==========================================================
    decimal_points : int, optional (default=4)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    dframe_stats : pandas.DataFrame
        DataFrame containing computed statistics.

    Raises
    ------
    ValueError
        If an invalid value is provided for the 'statistic' parameter.
    Exception
        If an unexpected error occurs.

    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> data = stm.dataset_read(name='nhanes')
    >>> sta.eda_tabstat(data, columns=['cholesterol'],
    ... statistics=['location', 'dispersion', 'distribution'])
    
    >>> sta.eda_tabstat(data, columns=['cholesterol'],
    ... statistics=['n', 'mean', 'std', 'median', 'iqr'])
    """
    dframe = ValidateArgs.check_dframe(par_name='data', user_input=dframe)
    columns = conv_to_arraylike(array_values=columns, par_name='columns')
    if columns:
        columns = ValidateArgs.check_member_of(
            par_name='columns', user_input=columns, valid_values=dframe.columns
        )
        dframe = dframe[columns] 
    measures = statistics
    measures = ([measures] if isinstance(measures, str) else measures)
    measures = conv_to_arraylike(array_values=measures, to_lower=True)
    if measures == ['location']:
        measures = LOCATION_MEASURES
    elif measures == ['dispersion']:
        measures = DISPERSION_MEASURES
    elif measures == ['distribution']:
        measures = DISTRIBUTION_MEASURES
    elif measures == '...':
        measures = DESCRIPTIVE_MEASURES
    dframe_stats = dframe.apply(lambda col: eda_tabstat_series(
        data=col,
        statistics=measures,
        decimal_points=decimal_points).values()
    ).T
    dframe_col_names = dframe.apply(lambda col: eda_tabstat_series(
        data=col,
        statistics=measures,
        decimal_points=decimal_points).keys()
    ).T
    dframe_stats.index = dframe_labels(dframe=dframe_stats, df_labels=columns)
    
    try:
        dframe_stats.columns = dframe_col_names.iloc[0, :]
    except:
        pass
    
    col_names = dframe_stats.columns
    col_names.name = 'Variable'
    dframe_stats = DataFrame(
        data=dframe_stats.values,
        index=dframe_stats.index,
        columns = col_names
    )
    
    return dframe_stats


def eda_tabstat_series(
    data: Series,
    statistics: Literal[
        'location', 'dispersion', 'distribution', 'n', 'count', 'mean',
        'median', 'mode', 'sum', 'min', 'max', 'range', 'var', 'std',
        'sem', 'cv', 'p25', 'q1', 'p75', 'q3', 'iqr', 'skew',
        'skewness', 'kurt or kurtosis'] = 'location',
    is_sample: bool = True,
    kurt_add_3: bool = False,
    decimal_points: int = 4
) -> float:
    """
    Tabulate values of a Series.

    Parameters
    ----------
    data : Series or array-like
        Input data for which descriptive statistics are calculated.
    statistics : str, optional (default='location')
        Type of statistics to compute. Possible values are:
        ==========================================================
        Statistic                 Description  
        ==========================================================
        location ................ Mean, median, mode
        dispersion .............. All dispersion measures
        distribution ............ Skewness, kurtosis
        n or count .............. Frequency
        mean .................... Mean
        median .................. Median
        mode .................... Mode
        sum ..................... Sum
        min ..................... Minimum
        max ..................... Maximum
        range ................... Range (Max - Min)
        var ..................... Variance
        std ..................... Standard deviation
        sem ..................... Standard error of the mean
        cv ...................... Coefficient of variation
        p25 or q1 ............... 25th percentile (First quartile)
        p75 or q3 ............... 75th percentile (Third quartile)
        iqr ..................... Interquartile range (Q3 - Q1)
        skew or skewness.........  Skewness
        kurt or kurtosis ........  Kurtosis
        ==========================================================
    is_sample : bool, optional(default=True)
        If `True`, sample is assumed, otherwise population is assumed.
    kurt_add_3 : bool, optional (default=False)
        If `True`, the we add 3 (Stata-like result)
    conf_level : float, optional (default=0.95)
        Confidence level for computing confidence intervals.
    decimal_points : int, optional (default=4)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    statistics : numpy.ndarray
        Array containing computed statistical measures.

    Raises
    ------
    SerifyError
        If the input data cannot be converted to a Series.

    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> data = stm.dataset_read(name='nhanes')['age']
    >>> sta.eda_tabstat_series(
        data=data, statistics=['location', 'dispersion', 'distribution']
    )
    {'n': 10337.0,
    'mean': 47.5637,
    'median': 49.0,
    'mode': 60.0,
    'sum': 491666.0,
    'min': 20.0,
    'max': 74.0,
    'range': 54.0,
    'var': 296.4174,
    'std': 17.2168,
    'cv': 0.362,
    'sem': 0.1693,
    'p25': 31.0,
    'p75': 63.0,
    'iqr': 32.0,
    'skewness': -0.1213,
    'kurtosis': -1.4392}

    >>> sta.eda_tabstat_series(
        data=data, statistics=['location', 'std', 'sem', 'skewness']
    )
    {'std': 17.2168,
    'sem': 0.1693,
    'skewness': -0.1213,
    'n': 10337.0,
    'mean': 47.5637,
    'median': 49.0,
    'mode': 60.0,
    'sum': 491666.0}
    """
    data = ValidateArgs.check_series(
        par_name='data', is_dropna=True, user_input=data
    )
    
    statistics = ValidateArgs.check_member_of(
        par_name='statistics',
        user_input=statistics,
        valid_values=VALID_STATISTICS
    )
    statistics = conv_to_arraylike(
        array_values=statistics, includes_str=True, to_lower=True
    )
    
    if 'count' in statistics:
        if 'n' in statistics:
            statistics.remove('count')
        elif 'n' not in statistics:
            statistics[statistics.index('count')] = 'n'

    if 'skew' in statistics:
        if 'skewness' in statistics:
            statistics.remove('skew')
        elif 'skewness' not in statistics:
            statistics[statistics.index('skew')] = 'skewness'

    if 'kurt' in statistics:
        if 'kurtosis' in statistics:
            statistics.remove('kurt')
        elif 'kurtosis' not in statistics:
            statistics[statistics.index('kurt')] = 'kurtosis'
    
    if 'q1' in statistics:
        if 'p25' in statistics:
            statistics.remove('q1')
        elif 'p25' not in statistics:
            statistics[statistics.index('q1')] = 'p25'
    
    if 'q3' in statistics:
        if 'p75' in statistics:
            statistics.remove('q3')
        elif 'p75' not in statistics:
            statistics[statistics.index('q3')] = 'p75'
    
    stats_dict = {
        'n': data.count(),
        'mean': data.mean(skipna=True),
        'median': data.median(skipna=True),
        'mode': eda_descriptive(data, statistic='mode').mode,
        'sum': data.sum(skipna=True),
        'min': data.min(skipna=True),
        'max': data.max(skipna=True),
        'range': data.max(skipna=True) - data.min(skipna=True),
        'var': eda_descriptive(data, statistic='var', is_sample=is_sample),
        'std': eda_descriptive(data, statistic='std', is_sample=is_sample),
        'sem': eda_descriptive(data, statistic='sem', is_sample=is_sample),
        'cv': eda_descriptive(data, statistic='cv', is_sample=is_sample),
        'p25': data.quantile(0.25),
        'q1': data.quantile(0.25),
        'p75': data.quantile(0.75),
        'q3': data.quantile(0.75),
        'iqr': data.quantile(0.75) - data.quantile(0.25),
        'skewness': data.skew(skipna=True),
        'kurtosis': eda_descriptive(data, statistic='kurtosis', kurt_add_3=kurt_add_3)
    }
    
    if 'location' in statistics:
        statistics += LOCATION_MEASURES
        statistics.remove('location')
    if 'dispersion' in statistics:
        statistics += DISPERSION_MEASURES
        statistics.remove('dispersion')
    if 'distribution' in statistics:
        statistics += DISTRIBUTION_MEASURES
        statistics.remove('distribution')
    if '...' in statistics:
        statistics += DESCRIPTIVE_MEASURES
        statistics.remove('...')
    
    statistics_list = [stats_dict.get(statistic) for statistic in statistics]
    if len(statistics_list) > 0:
        statistics_list = where(
            equal(statistics_list, None), nan, statistics_list
        ).tolist()
        statistics_list = round(statistics_list, decimal_points)
        
    result_dict = conv_list_to_dict(keys_list=statistics, values_list=statistics_list)
        
    return result_dict


def eda_descriptive(
    data: Series, 
    statistic: str, 
    is_sample: bool = True, 
    kurt_add_3: bool = False, 
    conf_level: float = 0.95, 
    decimal_points: int = 4
):
    """
    Calculate descriptive statistics for the input data.

    Parameters
    ----------
    data : Series or array-like
        Input data for which descriptive statistics are calculated.
    measures : str, optional (default='location')
        Type of statistics to compute. Possible values are:
        ==========================================================
        Measure                   Description  
        ==========================================================
        location ................ Mean, median, mode
        dispersion .............. All dispersion measures
        distribution ............ Skewness, kurtosis
        n or count .............. Frequency
        mean .................... Mean
        cimean .................. Confidence interval for the mean
        median .................. Median
        mode .................... Mode
        sum ..................... Sum
        min ..................... Minimum
        max ..................... Maximum
        range ................... Range (Max - Min)
        var ..................... Variance
        std ..................... Standard deviation
        sem ..................... Standard error of the mean
        cv ...................... Coefficient of variation
        p25 or q1 ............... 25th percentile (First quartile)
        p75 or q3 ............... 75th percentile (Third quartile)
        iqr ..................... Interquartile range (Q3 - Q1)
        skew or skewness......... Skewness
        kurt or kurtosis ........ Kurtosis
        ==========================================================
    is_sample : bool, optional(default=True)
        If `True`, sample is assumed, otherwise population is assumed.
    kurt_add_3 : bool, optional (default=False)
        If `True`, the we add 3 (Stata-like result)
    conf_level : float, optional (default=0.95)
        Confidence level for computing confidence intervals.
    decimal_points : int, optional (default=4)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    float or tuple or None
        The computed descriptive statistic or tuple of statistics.
        Returns None if the statistic is not recognized.

    Examples
    --------
    >>> import stemlab as stm
    >>> import stemlab.statistical as sta
    >>> data = stm.dataset_read(name='nhanes')
    >>> percentiles = list(range(0, 101, 5))
    >>> stats = ['n', 'mean', 'median', 'mode', 'sum', 'min', 'max',
    ... 'range', 'p25', 'p75', 'iqr', 'var', 'std', 'sem', 'cv',
    ... 'skewness', 'kurtosis', *percentiles]
    >>> for stat in stats:
    ...     result = sta.eda_descriptive(data=data['age'], statistic=stat)
    ...     print(f'{stat} = {result}')
    
    n = 10337
    mean = 47.5637
    median = 49.0
    mode = Result(
        mode: 60.0
        count: 344
    )
    sum = 491666.0
    min = 20.0
    max = 74.0
    range = 54.0
    p25 = 31.0
    p75 = 63.0
    iqr = 32.0
    var = 296.4174
    std = 17.2168
    sem = 0.1693
    cv = 0.362
    skewness = -0.1213
    kurtosis = -1.4392
    0 = 20.0
    5 = 21.0
    10 = 24.0
    15 = 26.0
    20 = 28.0
    25 = 31.0
    30 = 34.0
    35 = 37.0
    40 = 41.0
    45 = 45.0
    50 = 49.0
    55 = 53.0
    60 = 57.0
    65 = 60.0
    70 = 62.0
    75 = 63.0
    80 = 65.0
    85 = 67.0
    90 = 69.0
    95 = 72.0
    100 = 74.0
    """
    data = ValidateArgs.check_series(par_name='data', is_dropna=True, user_input=data)
    if isinstance(statistic, str):
        statistic = ValidateArgs.check_member(
            par_name='statistic', 
            valid_items=DESCRIPTIVE_MEASURES, 
            user_input=statistic
        )
    else: # percentiles
        statistic = ValidateArgs.check_numeric(
            par_name='statistic',
            limits=[0, 100], 
            user_input=statistic
        )
        statistic = statistic / 100 if statistic > 1 else statistic
            
    is_sample = ValidateArgs.check_boolean(user_input=is_sample, default=True)
    kurt_add_3 = ValidateArgs.check_boolean(user_input=kurt_add_3, default=False)
    conf_level = ValidateArgs.check_conf_level(conf_level)
    decimal_points = ValidateArgs.check_decimals(decimal_points)

    n = len(data)
    dof = 1 if is_sample else 0
    if statistic in ['n', 'count']:
        return n
    elif statistic == 'sum':
        return data.sum()
    elif statistic == 'mean':
        result = data.mean()
    elif statistic == 'cimean':
        mean, sigma = data.mean(), data.std(ddof=0)
        z_score = norm.ppf((1 + conf_level) / 2)
        margin_of_error = z_score * (sigma / sqrt(n))
        confidence_interval = (mean - margin_of_error, mean + margin_of_error)
        return Result(
            LCI=round(confidence_interval[0], decimal_points), 
            UCI=round(confidence_interval[1], decimal_points)
        )
    elif statistic == 'median':
        result = data.median()
    elif statistic == 'mode':
        mode_result = mode(data)
        return Result(
            mode=round(mode_result.mode), count=mode_result.count
        )
    elif statistic == 'min':
        result = data.min()
    elif statistic == 'max':
        result = data.max()
    elif statistic == 'range':
        result = data.max() - data.min()
    elif statistic in ['p25', 'q1']:
        result = data.quantile(0.25)
    elif statistic in ['p75', 'q3']:
        result = data.quantile(0.75)
    elif statistic == 'iqr':
        result = data.quantile(0.75) - data.quantile(0.25)
    elif statistic == 'var':
        result = data.var(ddof=dof)
    elif statistic == 'std':
        result = data.std(ddof=dof)
    elif statistic == 'sem':
        result = data.sem(ddof=dof)
    elif statistic == 'cv':
        mean = data.mean()
        std_dev = data.std(ddof=dof)
        result = (std_dev / mean)
    elif 'skew' in str(statistic):
        result = data.skew()
    elif 'kurt' in str(statistic):
        result = data.kurt() + (3 if kurt_add_3 else 0)
    else: # percentiles
        try:
            result = data.quantile(statistic)
        except Exception as e:
            raise e

    return fround(result, decimal_points)
