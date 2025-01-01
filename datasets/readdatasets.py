import os.path as op

from pandas import DataFrame, read_csv, read_stata
from stemlab.core.validators.validate import ValidateArgs

DIR_DATASETS = op.dirname(op.realpath(__file__))
DATA_SETS = read_csv(op.join(DIR_DATASETS, 'csv/datasets.csv'), sep=',')


def datasets_path():

    """
    Get the absolute path to the directory containing sample datasets.

    Returns
    -------
    str
        The absolute path to the datasets directory.
    """
    return op.dirname(op.realpath(__file__))


def save_data(dframe: DataFrame, file_name: str) -> None:
    """
    Save datasets to a file in either CSV or Excel format.

    Parameters
    ----------
    dframe : DataFrame
        DataFrame to be saved.
    file_name : str
        Name of the file to save.

    Raises
    ------
    TypeError
        If the file format is not supported.
        
    Examples
    --------
    >>> import pandas as pd
    >>> data = {'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]}
    >>> df = pd.DataFrame(data)
    
    >>> stm.save_data(dframe=df, file_name='data.csv')
    (Saves the DataFrame 'df' to a CSV file named 'data.csv' in the 'csv' directory)

    >>> stm.save_data(dframe=df, file_name='data.xlsx')
    (Saves the DataFrame 'df' to an Excel file named 'data.xlsx' in the 'excel' directory)
    """
    dframe = ValidateArgs.check_dframe(par_name='dframe', user_input=dframe)
    file_name = ValidateArgs.check_string(par_name='file_name', user_input=file_name)
    if '.' not in file_name: # default is .csv
        file_name = f'{file_name}.csv'
    if file_name.endswith('.csv'):
        file_name = f'{datasets_path()}/csv/{file_name}'
        dframe.to_csv(file_name, index=False)
    elif file_name.endswith('.xlsx') or file_name.endswith('.xlsx'):
        file_name = f'{datasets_path()}/excel/{file_name}'
        dframe.to_excel(file_name, index=False)
    else:
        raise TypeError("'file_name' must be '.csv' or '.xlsx'")


def dataset_read(name: str) -> DataFrame:
    """
    Read example datasets.

    Parameters
    ----------
    name : string
        Name of dataset to read (without extension).
        Must be a valid dataset present in stemlab.statistical.datasets

    Returns
    -------
    data : DataFrame
        Requested dataset.

    Examples
    --------
    >>> import stemlab as stm
    
    >>> df = stm.dataset_read(name='cotes_n8')
    >>> df
              x         y
    0  0.000000  0.200000
    1  0.114286  1.306741
    2  0.228571  1.318928
    3  0.342857  1.924544
    4  0.457143  2.998384
    5  0.571429  3.537895
    6  0.685714  2.599012
    7  0.800000  0.232000
    
    >>> sales = stm.dataset_read(name='sales')
    >>> sales
        Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec
    0  3863  1723  2318  4276  1664  1279  2257  1030  2182  3490  3168  4397
    1  4408  1969  3431  1246  3558  4503  4824  5006  2530  3791  5841  4915
    2  3384  1003  2282  2299  3956  5467  4691  4086  4315  3503  4900  1490
    3  5014  4153  1855  2805  5212  5704  3956  5293  1625  4039  4743  1761
    4  4673  2108  3093  3576  5137  5024  1986  4027  3351  2621  2710  3530
    5  4877  5231  4044  2731  2774  3607  4062  3292  3422  1687  1911  3594
    6  1010  4356  3130  5199  3393  3359  3711  3482  1574  2207  3388  4689
    7  4328  3681  5450  5442  3465  4954  3023  3298  3484  3496  5565  5468
    """
    name = ValidateArgs.check_string(par_name='name', user_input=name)
    file_name, _ = op.splitext(name) # just incase someone added an extension
    # check that dataset exist
    if file_name not in DATA_SETS["dataset"].to_numpy():
        raise ValueError(
            f"Dataset '{name}' does not exist. Available datasets include: "
            f"{DATA_SETS['dataset'].tolist()}"
        )
    # read dataset
    try:
        dframe = read_csv(
            op.join(DIR_DATASETS, f'csv/{file_name}.csv'), sep=','
        )
    except:
        dframe = read_stata(op.join(DIR_DATASETS, f'stata/{file_name}.dta'))

    return dframe


def datasets_show(details=True):
    """
    List available example datasets.
    
    Parameters
    ----------
    details : bool, optional (default=True)
        If `True`, a DataFrame containing information about each dataset will 
        be displayed. If `False`, only a list of dataset names will be shown.

    Returns
    -------
    dframe : {DataFrame, list}
        A dataframe with the name and description or a list of all 
        the datasets included in stemlab library.

    Examples
    --------
    >>> import stemlab as stm
    >>> data = stm.datasets_show()
    >>> data
    (DataFrame output omitted)
    >>> data = stm.datasets_show(details=False)
    >>> data
    (list output omitted)
    """
    details = ValidateArgs.check_boolean(user_input=details, default=False)
    dframe = DATA_SETS.sort_values(by='dataset')
    dframe.index = range(1, dframe.shape[0]+1)
    data = dframe if details else dframe['dataset'].tolist()
    
    return data
