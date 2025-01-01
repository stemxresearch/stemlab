from typing import Literal

from pandas import Series, DataFrame, Categorical, concat
from numpy import array, ndarray, where, arange

from stemlab.core.validators.errors import (
    IterableError, RowColumnsLengthError, DataFrameError, VectorLengthError,
    StringError, NotInColumnsError, RequiredError, PandifyError
)
from stemlab.core.arraylike import case_list, list_join, is_iterable
from stemlab.core.base.strings import str_singular_plural
from stemlab.core.validators.validate import ValidateArgs
from stemlab.core.base.dictionaries import Dictionary

    
class RelocateInsert:
    """
    A class for relocating or inserting columns into a DataFrame at 
    specified locations.
    
    Attributes
    ----------
    dframe : DataFrame
        The DataFrame in which columns will be relocated or inserted.
    columns : {str, list_like}
        The column name(s) to be relocated or inserted.
    loc : {str, int}
        The location where columns will be placed.
    after : bool, optional (default=True)
        If `True`, columns are placed after the specified location.
        If False, they are placed before.
    
    Methods
    -------
    _dframe_out(column_names: list[str] | None = None) -> DataFrame:
        Prepare DataFrame with specified column names.
    _get_data() -> tuple:
        Return data attributes of the class.
    _dframe_new(column_names: list[str] | None = None, is_relocate: bool = True) -> DataFrame:
        Relocate or insert columns into the DataFrame.
    """
    def __init__(
            self,
            dframe: DataFrame,
            columns: str | list,
            loc: str | int,
            after: bool = True
        ):
        """
        Initialize the RelocateInsert object.
        
        Parameters
        ----------
        dframe : pandas.DataFrame
            The DataFrame in which columns will be relocated or 
            inserted.
        columns :{str, list_like}
            The column name(s) to be relocated or inserted.
        loc : {str, list_like}
            The location where columns will be placed.
        after : bool, optional (default=True)
            If `True`, columns are placed after the specified location.
            If False, they are placed before.
        """
        self.dframe = dframe
        self.columns = columns
        self.loc = loc
        self.after = after
    
        # dframe
        if not isinstance(self.dframe, DataFrame):
            raise TypeError(
                f"'dframe' must be {DataFrame}, got {type(self.dframe).__name__}"
            )
        if isinstance(self.columns, (float, int)):
            self.columns = [self.columns] * self.dframe.shape[0]
        # columns
        if (not is_iterable(self.columns, includes_str=True) and 
            not isinstance(self.columns, DataFrame)):
            raise IterableError(
                user_input=self.columns, par_name='columns', includes_str=True
            )
        else:
            # convert to list if a string is given
            if not isinstance(self.columns, DataFrame):
                self.columns = (
                    [self.columns] if isinstance(self.columns, str) else self.columns
                )
                self.columns = array(self.columns)

        # loc
        if not isinstance(self.loc, (str, int)):
            TypeError(
                f"'loc' must be a string or integer, "
                f"got {type(self.loc).__name__}"
            )

        # after
        self.after = ValidateArgs.check_boolean(user_input=self.after, default=True)


    def _dframe_out(self, column_names: str | list[str]):
        """
        Prepare DataFrame with specified column names.
        
        Parameters
        ----------
        column_names : {str, list_like}
            The list of column names to use.
        
        Returns
        -------
        pandas.DataFrame
            The DataFrame with specified column names.
        """
        dframe = DataFrame(self.columns) # just in case it is not a DataFrame
        # check column names
        if not column_names: # None or []
            if isinstance(self.columns, DataFrame):
                # get column names from DataFrame (self.columns is a df)
                column_names = self.columns.columns
            else:
                column_names = [
                    f'Col{index + 1}' for index in range(dframe.shape[1])
                ]
        else:
            column_names = (
                [column_names] if isinstance(column_names, str) else column_names
            )
            
            if not is_iterable:
                raise IterableError(
                    par_name='column_names', user_input=self.column_names
                )

            if len(column_names) != dframe.shape[1]:
                raise RowColumnsLengthError(
                    par_name='column_names', rows=False, dframe=dframe, 
                )
        # rename the columns
        try:
            dframe.columns = column_names
        except Exception:
            pass

        self.dframe = dframe
            
        return self.dframe
    

    def _get_data(self):
        """
        Return data attributes of the class.
        
        Returns
        -------
        tuple
            A tuple containing dframe, columns, loc, and after attributes.
        """
        return (self.dframe, self.columns, self.loc, self.after)


    def _dframe_new(
            self, 
            column_names: list[str] | None = None, 
            is_relocate: bool = True
        ) -> DataFrame:
        """
        Concatenate or insert columns into a DataFrame at specified 
        locations.

        Parameters
        ----------
        column_names : {list_like, None}, optional (default=None)
            List of column names to insert into the DataFrame.
        is_relocate : bool, optional (default=True)
            If `True`, existing columns are relocated.
            If False, new columns are inserted.

        Returns
        -------
        dframe : pandas.DataFrame
            The modified DataFrame.

        Raises
        ------
        KeyError
            If 'loc' column does not exist in the DataFrame.
        TypeError
            If 'loc' is not an integer or string, or if 'columns' is 
            a DataFrame.
        """
        # drop columns that are being relocated
        dframe_dropped = (
            self.dframe.drop(self.columns, axis=1) 
            if is_relocate else self.dframe
        )

        if isinstance(self.loc, int):
            col_index = self.loc
        elif is_iterable(self.loc, includes_str=True):
            if is_iterable(self.loc, includes_str=False):
                self.loc = array(self.loc).flatten()[0] # get first element
            try:
                col_index = dframe_dropped.columns.get_loc(self.loc)
                if not isinstance(col_index, int): # there are duplicate cols
                    # get location of first occurance of the duplicated columns
                    if isinstance(col_index, slice):
                        col_index = int(
                            str(col_index).split(',')[0].split('(')[1]
                        )
                    elif isinstance(col_index, ndarray):
                        # get first element of tuple, 
                        # then first element of array, i.e. the [0][0] part
                        col_index = where(col_index==True)[0][0]
            except Exception:
                if self.loc in self.columns:
                    if is_relocate: # only needed if relocating, not inserting
                        raise KeyError(
                            f"'loc' cannot be in the column(s) you "
                            "are relocating / inserting."
                        )
                else:
                    raise KeyError(
                        f"'{self.loc}' does not exist in the specified "
                        "DataFrame"
                    )
            col_index += (1 if self.after else 0) # i.e. come after column
        else:
            raise TypeError(
                f"'loc' must be integer or string, got {type(self.loc).__name__}"
            )
        
        # if 'col_index' exceeds the number of columns, 
        # then just put the specified column(s) at the end
        col_index = (
            dframe_dropped.shape[1] 
            if abs(col_index) >= dframe_dropped.shape[1] else col_index
        )
        
        # if negative indices are given, then start from right
        col_index = (
            dframe_dropped.shape[1] + col_index + 1 
            if col_index < 0 else col_index
        )
        
        if is_relocate:
            if isinstance(self.columns, DataFrame):
                raise TypeError(f"'columns' cannot be a DataFrame")
            dframe_cols = self.dframe[self.columns] # subset the columns
        else: 
            dframe_cols = self._dframe_out(column_names=column_names)
            dframe_cols.index = dframe_dropped.index
        # now put the DataFrames together (concatenate)
        dframe = concat(
            objs=[
                dframe_dropped.iloc[:, :col_index],
                dframe_cols,
                dframe_dropped.iloc[:, col_index:]
            ],
            axis=1
        )
        
        return dframe


def dm_relocate(
    dframe: DataFrame,
    columns_to_relocate: str | list,
    relocate_to: str | int = -1,
    after: bool = True
) -> DataFrame:
    """
    Relocate column(s) before or after a specified column or at 
    a specified positional index.
    
    Parameters
    ----------
    dframe : DataFrame
        A DataFrame which contains the column(s) to be relocated.
    columns_to_relocate : list_like
        A list, tuple or array with the column(s) in the DataFrame 
        that need be relocated.
    relocate_to : {str, int}, optional (default=-1)
        Column name or index where the specified column(s) should be 
        relocated to.
    after : bool, optional (default=True)
        If `True`, specified column(s) will be inserted after the 
        column specified in `relocate_to`. Only used when 
        `relocate_to` is a column name (not index).
    
    Returns
    -------
    dframe : pandas.DataFrame
        A DataFrame with columns reordered.

    Examples
    --------
    >>> import stemlab.statistical as sta

    >>> df = sta.dm_data_random(nrows=5, ncols=len('BCDDEFGHJD'),
    ... col_names='BCDDEFGHJD', rand_seed=1234)
    >>> df
         B   C   D   D   E   F   G   H   J   D
    R1  57  93  48  63  86  34  25  59  33  36
    R2  40  53  40  36  68  79  90  83  57  60
    R3  86  47  44  48  77  21  10  85  90  13
    R4  12  29  22  75  85  91  24  81  70  56
    R5  38  91  97  23  22  79  41  99  94  55

    Move column H to come immediately after the first column D
    
    >>> sta.dm_relocate(dframe=df, columns_to_relocate=['H'],
    ... relocate_to='D', after=True)
         B   C   D   H   D   E   F   G   J   D
    R1  57  93  48  59  63  86  34  25  33  36
    R2  40  53  40  83  36  68  79  90  57  60
    R3  86  47  44  85  48  77  21  10  90  13
    R4  12  29  22  81  75  85  91  24  70  56
    R5  38  91  97  99  23  22  79  41  94  55

    Move columns 'B', 'J', and 'E' to come immediately before the
    last column
    
    >>> sta.dm_relocate(dframe=df,
    ... columns_to_relocate=['B', 'J', 'E'], relocate_to=-2,
    ... after=True)
         C   D   D   F   G   H   B   J   E   D
    R1  93  48  63  34  25  59  57  33  86  36
    R2  53  40  36  79  90  83  40  57  68  60
    R3  47  44  48  21  10  85  86  90  77  13
    R4  29  22  75  91  24  81  12  70  85  56
    R5  91  97  23  79  41  99  38  94  22  55
    """
    get_data = RelocateInsert(
        dframe=dframe,
        columns=columns_to_relocate,
        loc=relocate_to,
        after=after
    )
    dframe = RelocateInsert(*get_data._get_data())._dframe_new(is_relocate=True)

    return dframe


def dm_insert(
    dframe: DataFrame, 
    data_to_insert: list,
    column_names: list[str] | None = None, 
    insert_at: str | int = -1, 
    after: bool = True
) -> DataFrame:
    """
    Insert column(s) before or after a specified column or at 
    a specified positional index.
    
    Parameters
    ----------
    dframe : DataFrame
        A DataFrame to which new data should be inserted.
    data_to_insert : array_like
        A list, tuple or array with the data that needs to be inserted.
    col_names : list
        A list of column names of the new columns to be inserted.
    insert_at : {str, int}, optional (default=-1)
        Column name or index where the data in `data_to_insert` 
        should be inserted.
    after : bool, optional (default=True)
        If `True`, specified data will be inserted after the column 
        specified in `insert_at`. Only used when `insert_at` is 
        a column name (not index).
    
    Returns
    -------
    dframe : pandas.DataFrame
        A DataFrame with columns reordered.

    Examples
    --------
    >>> import stemlab as stm
    >>> import stemlab.statistical as sta
    >>> df = sta.dm_data_random(nrows=5, ncols=len('BCDDEFGHJD'),
    ... col_names='BCDDEFGHJD', rand_seed=1234)
    >>> data = [20]
    >>> sta.dm_insert(dframe=df, data_to_insert=2000,
    ... column_names=['x'], insert_at=2, after=True)
         B   C     x   D   D   E   F   G   H   J   D
    R1  57  93  2000  48  63  86  34  25  59  33  36
    R2  40  53  2000  40  36  68  79  90  83  57  60
    R3  86  47  2000  44  48  77  21  10  85  90  13
    R4  12  29  2000  22  75  85  91  24  81  70  56
    R5  38  91  2000  97  23  22  79  41  99  94  55
    
    >>> data = [[5000, 7000]] * 5
    >>> sta.dm_insert(dframe=df, data_to_insert=data,
    ... column_names=['x', 'y'], insert_at=2, after=True)
         B   C     x     y   D   D   E   F   G   H   J   D
    R1  57  93  5000  7000  48  63  86  34  25  59  33  36
    R2  40  53  5000  7000  40  36  68  79  90  83  57  60
    R3  86  47  5000  7000  44  48  77  21  10  85  90  13
    R4  12  29  5000  7000  22  75  85  91  24  81  70  56
    R5  38  91  5000  7000  97  23  22  79  41  99  94  55
    """
    get_data = RelocateInsert(
        dframe=dframe,
        columns=data_to_insert,
        loc=insert_at,
        after=after
    )
    dframe = RelocateInsert(*get_data._get_data())._dframe_new(
        column_names, is_relocate=False
    )
    
    return dframe


def dframe_labels(
    dframe: DataFrame | list, 
    df_labels: int | str | list[str] = 0,
    prefix: str = None, 
    index: bool = True
) -> list[str]:
    """
    Assigns index or column labels (names) to a DataFrame.

    Parameters
    ----------
    dframe : pandas DataFrame or list
        The DataFrame or list to label.
    df_labels : {None, int, str, list of str}, optional (default=0)
        Labels to assign to the DataFrame. 
        - If None, default labels are generated.
        - If an integer, a range of labels starting from the specified 
        integer is used.
        - If a string, labels are transformed based on the specified 
        method ('capitalize', 'title', 'lower', 'upper').
        - If a list of strings, it should match the length of rows or 
        columns in the DataFrame.
    prefix : str, optional (default=None)
        Prefix to prepend to each label.
    index : bool, optional (default=True)
        If `True`, assigns labels to the index. If False, assigns labels 
        to columns.

    Returns
    -------
    df_labels : list, of str
        The assigned labels.
    """
    # dframe
    if not isinstance(dframe, DataFrame):
        try:
            dframe = DataFrame(dframe)
            dframe.index = dframe.index.astype(str) # incase they are numeric
        except Exception:
            raise PandifyError(par_name='dframe')
    nrows, ncols = dframe.shape
    
    # index
    index = ValidateArgs.check_boolean(user_input=index, default=True)
    # df_labels
    if df_labels is None:
        df_labels = [''] * (nrows if index else ncols)
        return df_labels
    elif isinstance(df_labels, int):
        df_labels = arange(
            start=df_labels, stop=(nrows if index else ncols) + df_labels
        )
    elif isinstance(df_labels, str):
        df_labels = df_labels.lower()
        label_cases = {
            'capitalize': 'capitalize', 
            'title': 'title', 
            'lower': 'lower', 
            'upper': 'upper'
        }
        df_labels = ValidateArgs.check_member(
            par_name=f'{("index" if index else "col")}_labels', 
            valid_items=list(label_cases.keys()),
            user_input=df_labels
        )
        transform_method = label_cases[df_labels]
        df_labels = (
            dframe.index.str.__getattribute__(transform_method)() if index else 
            dframe.columns.str.__getattribute__(transform_method)()
        )
    elif is_iterable(array_like=df_labels):
        if len(df_labels) != (nrows if index else ncols):
            raise RowColumnsLengthError(
                par_name=f'{("index" if index else "col")}_labels', 
                rows=(True if index else False), 
                dframe=dframe
            )
    else:
        df_labels = arange(stop=ncols)

    if prefix is not None:
        prefix = ValidateArgs.check_string(par_name='prefix', user_input=prefix)
        df_labels = [f'{prefix}{label}' for label in df_labels]
    
    return df_labels


def dm_dframe_split(dframe: DataFrame, group_vars: list[str]) -> dict:
    """
    Split DataFrame

    Parameters
    ----------
    dframe : DataFrame
        A DataFrame that is to be split by specified group variable(s).
    group_vars : list_like
        Categorical variable(s) by which the DataFrame should be split. 

    Returns
    -------
    dict_dframes : Dict
        A dictionary containing the DataFrames after splitting `dframe`.
    
    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')
    >>> df = sta.dm_dframe_split(dframe=df, group_vars=['gender'])
    >>> df
    {'Female':     score_before  score_after  gender
    0             56           42  Female
    5             35           17  Female
    8             30           43  Female
    9             33           20  Female
    10            41           38  Female
    12            29           17  Female
    13            18           58  Female
    14            32           37  Female
    15            44           27  Female
    16            21           56  Female
    17            37           31  Female
    18            34           58  Female
    20            19           40  Female,
    'Male':     score_before  score_after gender
    1             30           51   Male
    2             46           32   Male
    3             21           21   Male
    4             29           51   Male
    6             42           50   Male
    7             34           58   Male
    11            48           38   Male
    19            37           31   Male
    21            48           47   Male
    22            23           56   Male
    23            25           60   Male
    24            44           45   Male
    25            30           18   Male
    26            20           35   Male
    27            27           21   Male
    28            29           24   Male
    29            52           34   Male}
    >>> df['male']
        score_before  score_after gender
    1             30           51   Male
    2             46           32   Male
    3             21           21   Male
    4             29           51   Male
    6             42           50   Male
    7             34           58   Male
    11            48           38   Male
    19            37           31   Male
    21            48           47   Male
    22            23           56   Male
    23            25           60   Male
    24            44           45   Male
    25            30           18   Male
    26            20           35   Male
    27            27           21   Male
    28            29           24   Male
    29            52           34   Male
    """
    dict_dframes = {
        key: dframe.loc[value] for key, value in 
        dframe.groupby(group_vars).groups.items()
    }
    
    dframes = Dictionary(dictionary=dict_dframes)

    return dframes


def dm_dframe_order_by_list(
    dframe: DataFrame, column_to_sort: str, labels_list: list
):
    """
    Sorts a DataFrame column by the specified list order.

    Parameters
    ----------
    dframe : pandas DataFrame
        The DataFrame to be sorted.
    column_to_sort : str
        The name of the column to be sorted.
    labels_list : list
        The list specifying the desired order of values.

    Returns
    -------
    pandas DataFrame
        The sorted DataFrame.

    Raises
    ------
    TypeError
        If 'dframe' is not a DataFrame or 'column_to_sort' is not a string.
    ValueError
        If 'labels_list' is not an iterable.

    Examples
    --------
    >>> df = pd.DataFrame({'A': ['c', 'b', 'a'], 'B': [3, 2, 1]})
    >>> df
       A  B
    0  c  3
    1  b  2
    2  a  1
    >>> labels_list = ['b', 'a', 'c']
    >>> sta.dm_dframe_order_by_list(dframe=df, column_to_sort='A',
    ... labels_list=labels_list)
       A  B
    1  b  2
    2  a  1
    0  c  3
    """
    if not isinstance(dframe, DataFrame):
        raise DataFrameError(par_name='dframe', user_input=dframe)
    
    if not isinstance(column_to_sort, str):
        raise StringError(par_name='column_to_sort', user_input=column_to_sort)
    
    if not is_iterable(labels_list):
        raise IterableError(par_name='label_list', user_input=labels_list)

    dframe = dframe.iloc[
        Categorical(dframe[column_to_sort], labels_list).argsort()
    ]
    
    return dframe


def dm_stack_cols(
    dframe: DataFrame, 
    columns_to_stack: list[str] = [], 
    col_labels: list[str] | None = None, 
    order_columns: bool = True
) -> DataFrame:
    """
    Stack columns of a DataFrame.

    Parameters
    ----------
    dframe : DataFrame
        DataFrame with the columns to be stacked.
    columns_to_stack : list, of str, optional
        Columns whose values are to be stacked into a single column.
    col_labels : list, of str or None, optional
        Column names for the new DataFrame. If None, default column names 
        '_stack' and 'values' are used.
    order_columns : bool, optional
        Whether to order the stacked DataFrame columns based on the provided 
        'columns_to_stack' order.

    Returns
    -------
    dframe : pandas.DataFrame
        A DataFrame with two columns, where the first column contains 
        the categories (derived from column names) and the second 
        contains the values.

    Raises
    ------
    TypeError
        If `dframe` is not a DataFrame or `col_labels` is not iterable.
    ValueError
        If `columns_to_stack` contains columns that do not exist in the 
        DataFrame or if `col_labels` has a length different from 2.

    Examples
    --------
    >>> import pandas as pd
    >>> import stemlab.statistical as sta
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> df
       A  B
    0  1  3
    1  2  4
    >>> sta.dm_stack_cols(df, columns_to_stack=['A', 'B'],
    ... col_labels=['Category', 'Value'])
      Category  Value
    0        A      1
    1        A      2
    2        B      3
    3        B      4
    
    >>> df = stm.dataset_read(name='sales')
    >>> sta.dm_stack_cols(df, columns_to_stack=['Jan', 'Feb', 'Mar'],
    ... col_labels=['month', 'sales'])
       month  sales
    0    Jan   4408
    1    Jan   3863
    2    Jan   1010
    3    Jan   4673
    4    Jan   4328
    5    Jan   5014
    6    Jan   4877
    7    Jan   3384
    8    Feb   4356
    9    Feb   1969
    10   Feb   5231
    11   Feb   1003
    12   Feb   4153
    13   Feb   3681
    14   Feb   2108
    15   Feb   1723
    16   Mar   4044
    17   Mar   3130
    18   Mar   1855
    19   Mar   3093
    20   Mar   2282
    21   Mar   3431
    22   Mar   2318
    23   Mar   5450
    """
    dframe = ValidateArgs.check_dframe(par_name='dframe', user_input=dframe)
    df_columns_set = set(dframe.columns) # should be here
    if columns_to_stack:
        if not is_iterable(array_like=columns_to_stack):
            raise IterableError(
                par_name='columns_to_stack', user_input=columns_to_stack
            )
        columns_to_stack = array(columns_to_stack).tolist()
        # check that the columns are in the DataFrame
        columns_set = set(columns_to_stack)
        if not columns_set.issubset(df_columns_set):
            not_in_columns = columns_set.difference(df_columns_set)
            not_in_columns = ', '.join(map(str, not_in_columns))
            if len(not_in_columns) == 1:
                not_in_columns = f"'{not_in_columns[0]}' is"
            else:
                not_in_columns = f"'{not_in_columns}' are"
            raise ValueError(
                f"{not_in_columns} not among the DataFrame column names"
            )
    else:
        columns_to_stack = dframe.columns.tolist()
    # extract the column names
    dframe = dframe[columns_to_stack]
    dframe = DataFrame(dframe.stack())\
        .reset_index()\
        .drop('level_0', axis=1)\
        .sort_values(by='level_1')
    if col_labels is None:
        dframe.columns = ['_stack', 'values']
    else:
        if not is_iterable(array_like=col_labels):
            raise IterableError(par_name='col_labels', user_input=col_labels)
        if len(col_labels) != 2:
            raise VectorLengthError(
                par_name='col_labels', 
                n=2, 
                label='exactly', 
                user_input=col_labels
            )
        try:
            dframe.columns = col_labels
        except Exception:
            dframe.columns = ['_stack', 'values']

    order_columns = ValidateArgs.check_boolean(
        user_input=order_columns, default=True
    )
    if order_columns:
        # sort DataFrame in the way the columns are ordered in the list
        # by default, Pandas will sort in alphabetic order
        dframe = dm_dframe_order_by_list(
            dframe=dframe,
            column_to_sort=dframe.columns[0],
            labels_list=columns_to_stack
        )
    
    dframe.index = range(dframe.shape[0])

    return dframe


def dm_unstack_cols(
    dframe: DataFrame, 
    cat_column: str | list[str], 
    categories: list = [], 
    values_column: str | list[str] = '', 
    col_labels: list[str] | None = None, 
    order_columns: bool = True
) -> DataFrame:
    """
    Unstacks specified categories from a DataFrame and returns a new 
    DataFrame.

    Parameters
    ----------
    dframe : DataFrame
        The DataFrame to unstack categories from.
    cat_column : str or list of str
        The column(s) containing the categories to unstack.
    categories : list, optional (default=[])
        The specific categories to unstack. If empty, all unique 
        categories from `cat_column` are used.
    values_column : {str, list_like}, optional (default='')
        The column(s) containing the values associated with the 
        categories.
    col_labels : {str, None}, optional (default=None)
        The labels for the columns in the resulting DataFrame.
         If None, original category names are used.
    order_columns : bool, optional (default=True)
        Whether to order the resulting DataFrame columns based on the 
        specified `categories` order.

    Returns
    -------
    DataFrame
        A DataFrame with unstacked categories as columns.

    Raises
    ------
    TypeError
        If `dframe` is not a DataFrame, `cat_column` is not a string or list 
        of strings, or `col_labels` is not iterable.
    ValueError
        If `categories` contain values not present in `cat_column` or if 
        `values_column` is not provided.

    Examples
    --------
    >>> import pandas as pd
    >>> import stemlab as stm
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='sales')
    
    Create stacked column that we will unstack
    
    >>> data = sta.dm_stack_cols(df,
    ... columns_to_stack=['Jan', 'Feb', 'Mar'],
    ... col_labels=['month', 'sales'])
       month  sales
    0    Jan   4408
    1    Jan   3863
    2    Jan   1010
    3    Jan   4673
    4    Jan   4328
    5    Jan   5014
    6    Jan   4877
    7    Jan   3384
    8    Feb   4356
    9    Feb   1969
    10   Feb   5231
    11   Feb   1003
    12   Feb   4153
    13   Feb   3681
    14   Feb   2108
    15   Feb   1723
    16   Mar   4044
    17   Mar   3130
    18   Mar   1855
    19   Mar   3093
    20   Mar   2282
    21   Mar   3431
    22   Mar   2318
    23   Mar   5450
    
    Now unstack the columns Feb and Jan, in that order
    
    >>> sta.dm_unstack_cols(data, cat_column='month',
    ... categories=['Feb', 'Jan'], values_column='sales')
        Feb   Jan
    0  4356  4408
    1  1969  3863
    2  5231  1010
    3  1003  4673
    4  4153  4328
    5  3681  5014
    6  2108  4877
    7  1723  3384
    """
    dframe = ValidateArgs.check_dframe(par_name='dframe', user_input=dframe)
    # get DataFrame columns, we will need them
    dframe_columns = set(dframe.columns)
    if not is_iterable(cat_column, includes_str=True):
        raise IterableError(par_name='cat_column', user_input=cat_column)
    if not isinstance(cat_column, str):
        if len(cat_column) != 1:
            raise VectorLengthError(
                par_name='cat_column', n=1, user_input=cat_column
            )
        else:
            cat_column = cat_column[0]
    # check that the given column is one of the DataFrame columns
    if cat_column not in dframe_columns:
        raise NotInColumnsError(par_name='cat_column', user_input=cat_column)
    
    # categories
    if categories:
        if not is_iterable(categories):
            raise IterableError(par_name='categories', user_input=categories)
        # get only those categories that are in the `cat_column` column
        categories_set = set(dframe[cat_column]).intersection(categories)
        if len(categories_set) == 0:
            raise ValueError(
                f"'{categories}' does not contain any of the categories "
                f"specified in 'categories'"
            )
        # maintain order or categories, note that `set` did order the 
        # elements alphabetically
        not_found = set(categories).difference(set(dframe[cat_column]))
        if not_found: # if it is not empty
            list_join_ = list_join(
                lst=not_found, 
                delimiter=", ", 
                use_and=True, 
                html_tags=False
            )
            was_were = str_singular_plural(
                n=len(not_found), singular_form='was', plural_form='were'
            )
            s = str_singular_plural(n=len(not_found))
            raise Exception(
                f"The value{s} {list_join_} {was_were} not found in the "
                f"column '{cat_column}'"
            )
        categories = [categ for categ in categories if categ in categories_set]
    else:
        # if categories are not given, then use all the categories of the 
        # specified column
        categories = dframe[cat_column].unique()
    categories = array(categories).tolist() # convert to list for convenience
    # values_column
    if not values_column:
        raise RequiredError(par_name='values_column')
    else:
        if not is_iterable(values_column, includes_str=True):
            raise IterableError(
                par_name='values_column', user_input=values_column
            )
        if not isinstance(values_column, str):
            if len(values_column) != 1:
                raise VectorLengthError(
                    par_name='values_column', n=1, user_input=values_column
                )
            else:
                values_column = values_column[0]
    # filter only the two variables from the DataFrame 
    # [cat_column, values_column]
    dframe = dframe.loc[:, [cat_column, values_column]]
    # filter the categories to unstack (filtering is by observations)
    dframe = dframe.loc[dframe[cat_column].isin(categories)]
    # split DataFrame
    df_dict = dm_dframe_split(dframe, group_vars=[cat_column])
    # initialize and concatenate
    dframe = DataFrame()
    for name, df in df_dict.items():
        df = df.iloc[:, [1]]
        df.index = arange(df.shape[0])
        df.columns = [name]
        dframe = concat([dframe, df], axis=1)
    # assign column labels after concatenation
    if col_labels is None: # no columns given
        dframe.columns = df_dict.keys()
    else:
        try:
            dframe.columns = dframe_labels(
                dframe=dframe, df_labels=col_labels, index=False
            )
        except Exception: # if invalid columns, then user original names
            dframe.columns = df_dict.keys()
    
    if not isinstance(order_columns, bool):
        order_columns = True
    if order_columns:
        # sort DataFrame in the way the columns are ordered in the list
        # by default, Pandas will sort in alphabetic order
        if col_labels in ['lower', 'upper', 'title', 'capitalize']:
            categories = case_list(lst=categories, case_=col_labels)
        dframe = dframe[categories]

    return dframe


def dm_outliers(
    data: list,
    method: Literal['iqr', 'std'] = 'iqr', 
    std: Literal[1, 2, 3] = 1
) -> Series:
    """
    Detect outliers in the given data.

    Parameters
    ----------
    data : array_like
        Input data values.
    method : str, {'iqr', 'std'}, optional (default='iqr')
        The method used to detect outliers:
            - 'iqr': Interquartile range method (default).
            - 'std': Standard deviation method.
    std : int, {1, 2, 3} optional, (default=1)
        Standard deviations from the mean. Only used if `method='std'`.

    Returns
    -------
    outliers : numpy.ndarray
        A Numpy array containing the outliers.

    Raises
    ------
    ValueError
        If `method` or `std` have invalid values.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 100])
    >>> sta.dm_outliers(data)
    [100]
    """
    data = ValidateArgs.check_series(par_name='data', is_dropna=True, user_input=data)

    method = ValidateArgs.check_member(
        par_name='method', valid_items=['iqr', 'std'], user_input=method
    )

    std = ValidateArgs.check_member(
        par_name='std', is_string=False, valid_items=[1, 2, 3], user_input=std
    )
    
    if method == 'iqr':
        Q1, Q3 = data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_limit = Q1 - (1.5 * IQR)
        upper_limit = Q3 + (1.5 * IQR)
    else:
        lower_limit = data.mean() - std * data.std()
        upper_limit = data.mean() + std * data.std()
    outliers = data[(data < lower_limit) | (data > upper_limit)].values

    return outliers


def dm_outliers_replace(
    data: list,
    method: Literal['iqr', 'std'] = 'iqr', 
    std: Literal[1, 2, 3] = 1,
    replace_with: int | float | Literal['mean', 'median'] = 'median'
) -> Series:
    """
    Replace outliers in the given data.

    Parameters
    ----------
    data : Series, list, tuple, or ndarray
        Input data values.
    method : str, {'iqr', 'std'}, optional
        The method used to detect outliers:
            - 'iqr': Interquartile range method (default).
            - 'std': Standard deviation method.
    std : int, optional
        Standard deviations from the mean (default=1). 
        Only used if `method='std'`.
    replace_with : {'median', 'mean', int, float}, optional (default='median')
        The value to replace the outlier(s) with.

    Returns
    -------
    replaced_data : Series
        A Pandas Series with outliers replaced.

    Raises
    ------
    SerifyError
        If data cannot be converted to a pandas Series.

    Examples
    --------
    >>> import pandas as pd
    >>> import stemlab.statistical as sta
    >>> data = pd.Series([1, 2, 3, 100])
    >>> print(data)
    >>> sta.dm_outliers_replace(data, replace_with='median')
    0    1
    1    2
    2    3
    3    2.5
    dtype: int64
    
    >>> sta.dm_outliers_replace(data, replace_with=-99)
    0    1
    1    2
    2    3
    3    -99  # Outlier 100 replaced with the value -99
    dtype: int64
    """
    data = ValidateArgs.check_series(par_name='data', is_dropna=True, user_input=data)

    method = ValidateArgs.check_member(
        par_name='method', valid_items=['iqr', 'std'], user_input=method
    )

    std = ValidateArgs.check_member(
        par_name='std', valid_items=[1, 2, 3], is_string=False, user_input=std
    )

    if isinstance(replace_with, str):
        method = ValidateArgs.check_member(
            par_name='method', 
            valid_items=['mean', 'median'], 
            user_input=replace_with
        )
        replace_with = (data.mean() if replace_with == 'mean' else data.median())
    else:
        replace_with = ValidateArgs.check_numeric(
            par_name='replace_with', user_input=replace_with
        )

    if method == 'iqr':
        Q1, Q3 = data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_limit, upper_limit = Q1 - (1.5 * IQR), Q3 + (1.5 * IQR)
    else:
        lower_limit = data.mean() - std * data.std()
        upper_limit = data.mean() + std * data.std()

    outliers = data[(data < lower_limit) | (data > upper_limit)].tolist()
    data = data.replace(to_replace=outliers, value=replace_with)
            
    return data


def dm_scale(
    data: list,
    method: Literal['mas', 'mmas', 'zscore'] = 'zscore',
    is_sample: bool = True
) -> Series:
    """
    Scale / adjust values that exist on different scales into a common 
    scale.

    Parameters
    ----------
    data : list_like
        An iterable with data values.
    method : str, {'mas', 'mmas', 'zscore'}, optional (default='zscore')
        The scaling method to be applied.
    ddof : bool, optional (default=True)
        If `True` (i.e. sample), n-1 will be used, otherwise n applies.

    Returns
    -------
    data : Series
        A Series with the scaled values.
        
    Examples
    --------
    >>> import stemlab as stm
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='sales')

    Apply the dm_scale() function to the columns, `Jan`, `Jun` and `sep`
    
    >>> df[['Jan', 'Jun', 'Sep']].apply(
        sta.dm_scale, method='zscore', is_sample=True
    )
            Jan       Jun       Sep
    0 -0.062767 -2.035303 -0.637960
    1  0.356320  0.182932 -0.284652
    2 -0.431103  0.846201  1.527577
    3  0.822315  1.009266 -1.203457
    4  0.560097  0.541400  0.548872
    5  0.716966 -0.433550  0.620955
    6 -2.256632 -0.604184 -1.255235
    7  0.294803  0.493237  0.683901
    
    Apply the dm_scale() function to all the columns of the `df` 
    DataFrame
    
    >>> df.apply(sta.dm_scale, method='zscore', is_sample=True)
    (output omitted)  
    """
    data = ValidateArgs.check_series(
        par_name='data', is_dropna=True, user_input=data
    )

    method = ValidateArgs.check_member(
        par_name='method', 
        valid_items=['mas', 'mmas', 'zscore'], 
        user_input=method
    )
    
    is_sample = ValidateArgs.check_boolean(user_input=is_sample, default=True)
   
    scaling_methods = {
        'mas': data / data.abs().max(),
        'mmas': (data - data.min()) / (data.max() - data.min()),
        'zscore': (data - data.mean()) / data.std(ddof=1 if is_sample else 0)
    }
    data = scaling_methods.get(method, 'zscore')

    return data


def dm_drop_contains(
    dframe: DataFrame, 
    strng: str, 
    col_names: list[str] | None = None, 
    re_index: bool = False, 
    axis: Literal[0, 1, 'index', 'rows', 'columns'] = 0
) -> DataFrame:
    """
    Drop rows or columns containing a specific string in their values.

    Parameters
    ----------
    dframe : DataFrame
        Input DataFrame.
    strng : str
        String to search for in the DataFrame values.
    col_names : {list, None}, optional (default=None)
        List of column names to search for the string. 
        If None, search is performed on all object-type columns.
    re_index : bool, optional (default=False)
        Whether to re-index the DataFrame after dropping rows.
    axis : {0 or 'index' or 'rows', 1 or 'columns'}, optional (default=0)
        Whether to drop rows or columns containing the string.
        If 0 or 'index' or 'rows', drops rows. If 1 or 'columns', 
        drops columns.

    Returns
    -------
    dframe : pandas.DataFrame
        DataFrame with rows or columns containing the string dropped.

    Raises
    ------
    ValueError
        If axis is not 0, 1, 'index', 'rows', or 'columns'.

    Examples
    --------
    >>> import pandas as pd
    >>> import stemlab.statistical as sta
    >>> df = pd.DataFrame({'A': ['abc', 'def', 'ghi'],
    ... 'B': ['abc', 'pqr', 'xyz']})
    
    Drop rows that contain `def`
    
    >>> sta.dm_drop_contains(dframe=df, strng='def', axis=0)
         A    B
    0  abc  abc
    2  ghi  xyz
    
    Drop columns that contain `def`
    
    >>> sta.dm_drop_contains(dframe=df, strng='def', axis=1)
       A
    0  abc
    1  pqr
    2  ghi
    """
    dframe = ValidateArgs.check_dframe(par_name='dframe', user_input=dframe)
    strng = ValidateArgs.check_string(par_name='strng', user_input=strng)
    col_names = ValidateArgs.check_dflabels(
        par_name='col_names', user_input=col_names
    )
    re_index = ValidateArgs.check_boolean(user_input=re_index, default=False)
    axis = ValidateArgs.check_member(
        par_name='axis',
        valid_items=[0, 1, 'index', 'rows', 'columns'],
        is_string=False,
        user_input=axis
    )
    if axis in [1, 'columns']:
        dframe = dframe.drop(
            [
                col for col in dframe.columns 
                if dframe[col].apply(lambda s: strng in str(s)).any()
            ],
            axis=1
        )
    else:
        if col_names is None or col_names == 0: # note `val_df_labels()` updates
            col_names = list(dframe.select_dtypes('object').columns)
        if col_names: # at least one column with strings should be found
            if not isinstance(col_names, (list, tuple)):
                if isinstance(col_names, (int, str)):
                    col_names = [col_names]
                else:
                    raise TypeError(
                        f"'col_names' must be a list or tuple "
                        f"but got {type(col_names).__name__}"
                    )
            for col_name in col_names:
                dframe = dframe[~dframe[col_name].str.contains(strng, na=False)]
            if re_index:
                dframe = dframe.reset_index(drop=True)
        
    return dframe


def dm_na_replace(
    data: list,
    replace_with: int | float | Literal['mean', 'median'] = 'median'
) -> Series:
    """
    Replace missing values with the mean or median.

    Parameters
    ----------
    data : pandas Series or array-like
        The input data containing missing values.
    replace_with : {'median', 'mean', int, float}, optional (default='median')
        The value to replace the missing value(s) with.

    Returns
    -------
    data : pandas.Series
        The input data with missing values replaced with the specified 
        `replace_with`.

    Examples
    --------
    >>> import pandas as pd
    >>> import stemlab.statistical as sta
    >>> data = data = pd.Series([1, 2, 3, None, 5, 9, None, 7])
    >>> print(data)
    >>> sta.dm_na_replace(data, replace_with='median')
    0    1.0
    1    2.0
    2    3.0
    3    4.0
    4    5.0
    5    9.0
    6    4.0
    7    7.0
    dtype: float64
    
    >>> sta.dm_na_replace(data, replace_with='mean')
    0    1.0
    1    2.0
    2    3.0
    3    4.5
    4    5.0
    5    9.0
    6    4.5
    7    7.0
    dtype: float64
    
    >>> sta.dm_na_replace(data, replace_with=-99)
    0     1.0
    1     2.0
    2     3.0
    3   -99.0
    4     5.0
    5     9.0
    6   -99.0
    7     7.0
    dtype: float64
    """
    data = ValidateArgs.check_series(par_name='data', is_dropna=False, user_input=data)
    if isinstance(replace_with, str):
        replace_with = ValidateArgs.check_member(
            par_name='replace_with', 
            valid_items=['mean', 'median'], 
            user_input=replace_with
        )
    else:
        replace_with = ValidateArgs.check_numeric(
            par_name='replace_with', user_input=replace_with
        )
    na_drop = Series(data).dropna()
    if replace_with in ['mean', 'median']:
        replace_value = (
            na_drop.median() if replace_with == 'median' else na_drop.mean()
        )
    else:
        replace_value = replace_with
    data = data.fillna(value = replace_value)
    
    return data