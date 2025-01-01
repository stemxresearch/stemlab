from typing import Any, Callable, Literal
import warnings

from numpy import asarray, ceil, inf, ndarray, unique, all, array
from sympy import MatrixBase, sympify, Float, Integer, Matrix
from pandas import DataFrame, Series

from stemlab.core.validators.errors import (
    NumericError, IterableError, NotMemberError, NumpifyError, RequiredError,
    SquareMatrixError, PandifyError, SerifyError,
    CoeffConstantsError, VectorLengthError, LengthDifferError, FloatifyError,
    NotArrayError, DifferenceError, IntegifyError, IntervalError,
    LowerGteUpperError
)
from stemlab.core.datatypes import (
    ArrayMatrixLike, ListArrayLike, NumpyArray, NumpyArraySympyMatrix
)
from traitlets import Instance


class ValidateArgs:
    
    @staticmethod
    def check_args_count(ftn_name: str, args_dict: dict) -> None:
        """
        Validate that `n` out of a list of arguments are provided.

        Parameters
        ----------
        ftn_name : str
            The name of the function.
        args_dict : dict
            A dictionary containing the function arguments (keys) and their 
            respective values (values).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the number of provided parameters is not exactly `n`.
        
        Notes
        -----
        This function is typically used to ensure that exactly `n` out of 
        all possible parameters are supplied, which is essential for 
        performing calculations that require `n` known values to determine 
        the other.
        """
        lst = args_dict.values()
        args_names = args_dict.keys()
        n = len(args_names)
        args_counts = sum(param is not None for param in lst)
        if args_counts != n - 1:
            raise ValueError(
                f"Expected exactly {n - 1} arguments of {args_names} to be "
                f"provided for the '{ftn_name}()' function but got: {args_counts} "
                f"arguments."
            )


    @staticmethod
    def check_elements_not_in_range(lst, lower=0, upper=4):
        """
        Extract elements from the list that are not between the specified 
        lower and upper bounds.

        Parameters
        ----------
        lst : list
            List of numerical elements to check.
        lower : int or float, optional (default=0)
            The lower bound of the range (inclusive).
        upper : int or float, optional (default=4)
            The upper bound of the range (inclusive).

        Returns
        -------
        list
            A list of elements that are outside the specified range.

        Examples
        --------
        >>> import stemlab as stm
        >>> numbers1 = [1, 2, 5, -1]
        >>> check_elements_not_in_range(numbers1)
        [5, -1]

        >>> numbers2 = [0, 2, 3]
        >>> check_elements_not_in_range(numbers2)
        []

        >>> numbers3 = [0, 1, 2, 3, 4, 5]
        >>> check_elements_not_in_range(numbers3, lower=1, upper=3)
        [0, 4, 5]
        
        >>> numbers4 = []
        >>> check_elements_not_in_range(numbers4)
        []
        """
        return [x for x in lst if x < lower or x > upper]


    @staticmethod
    def check_decimals(x: Any, a: int = -1, b: int = 14) -> int:
        """
        Validate the number of decimal points.

        Parameters
        ----------
        x : int
            The input to be validated.
        a : int, optional (default=-1)
            The minimum allowed value for decimal points.
        b : int, optional (default=14)
            The maximum allowed value for decimal points.

        Returns
        -------
        decimal_points : int
            The validated number of decimal points.

        Raises
        ------
        NumericError
            If decimal_points is not an integer or falls outside the 
            specified range.
        """
        try:
            a, b = (int(a), int(b))
        except Exception:
            raise NumericError(
                par_name='x', limits=[a, b], is_integer=True, user_input=x
            )
        
        if not isinstance(x, int) or not (a <= x <= b):
            raise NumericError(
                par_name='x', limits=[a, b], is_integer=True, user_input=x
            )
        
        return x


    @staticmethod
    def check_axis(user_input: Any, par_name: str = 'axis') -> int:
        """
        Validate the array axis.
        
        Parameters
        ----------
        x : Any
            The input to validate.
        par_name : str
            Name to use in error messages to describe the parameter being 
            checked.
        
        Returns
        -------
        axis : int
            Integer, 0 for rows and 1 for columns.
        """
        axis = user_input
        if 'row' in str(axis).lower() or 'index' in str(axis).lower():
            axis = 0
            
        if 'col' in str(axis).lower():
            axis = 1
            
        if not isinstance(axis, int):
            raise ValueError(
                f"Expected '{par_name}' to be 0, 1, 'index', "
                f"'rows' or 'columns', got {axis}"
            )
        
        return axis

    
    @staticmethod
    def check_array_matrix(
        A: ArrayMatrixLike,
        nrows: int = None,
        ncols: int = None,
        is_square: bool = False,
        par_name: str = 'array'
    ) -> ndarray | Matrix:
        """
        Validate and convert the input to a NumPy array or SymPy Matrix. 
        Optionally, check for specific dimensions and square matrix 
        properties.

        Parameters
        ----------
        A : {ndarray, Matrix}
            Input array or matrix to validate. Can be a NumPy ndarray, 
            a SymPy Matrix, or a DataFrame (which will be converted to a 
            NumPy array).
        nrows : int, optional (default=None)
            Expected number of rows in the input matrix. If provided, the 
            input matrix must have this number of rows.
        ncols : int, optional (default=None)
            Expected number of columns in the input matrix. If provided, 
            the input matrix must have this number of columns.
        is_square : bool, optional (default=False)
            If `True`, the function checks if the matrix is square 
            (i.e., number of rows equals number of columns).
        par_name : str, optional (default='array')
            Name to use in error messages to describe the parameter being 
            checked.

        Returns
        -------
        A : NumpyArraySympyMatrix
            The validated and possibly converted NumPy array or SymPy Matrix.

        Raises
        ------
        ValueError
            If the input matrix does not meet the specified `nrows` or 
            `ncols` requirements.
        NumpifyError
            If the input cannot be converted to a NumPy array and is 
            neither a valid ndarray nor Matrix.
        """
        if not isinstance(A, (ndarray, MatrixBase)):
            if isinstance(A, DataFrame):
                try:
                    A = A.values
                except:
                    raise NumpifyError(par_name=par_name)
            else:
                try:
                    if not isinstance(A, MatrixBase):
                        A = array(A)
                except Exception as e:
                    raise e
                        
        if nrows is not None:
            if nrows != A.shape[0]:
                raise ValueError(
                    f"Expected '{par_name}' to have {nrows} rows, "
                    f"but got {A.shape[0]}"
                )
        
        if ncols is not None:
            if ncols != A.shape[1]:
                raise ValueError(
                    f"Expected '{par_name}' to have {ncols} rows, "
                    f"but got {A.shape[1]}"
                )
            
        if is_square:
            A = ValidateArgs.check_square_matrix(par_name='A', A=A)

        return A
    
    
    @staticmethod    
    def check_dframe(par_name: str, user_input: Any) -> DataFrame:
        """
        Validate the a pandas DataFrame

        Parameters
        ----------
        par_name : str
            Name to use in error messages to describe the parameter being 
            checked.
        user_input : Any
            The input to validate.

        Returns
        -------
        dframe : pandas.DataFrame
            The validated pandas DataFrame

        Raises
        ------
        PandifyError
            If `dframe` is not a pandas DataFrame, or cannot be converted.
        """
        try:
            dframe = DataFrame(user_input)
        except ValueError:
            raise PandifyError(par_name=par_name)

        return dframe


    @staticmethod
    def check_series(
        par_name: str, is_dropna: bool = False, user_input: Any = 'user_input'
    ) -> Series:
        """
        Validate the a pandas DataFrame

        Parameters
        ----------
        par_name : str
            Name to use in error messages to describe the parameter being 
            checked.
        is_dropna : bool, optional (default=False)
            If `True`, missing values (NaN) will be dropped from the Series.
        user_input : Any
            The input to validate.

        Returns
        -------
        data : pandas.Series
            The validated pandas Series

        Raises
        ------
        PandifyError
            If `dframe` is not a pandas DataFrame, or cannot be converted.
        """
        try:
            data = Series(user_input)
        except ValueError:
            raise SerifyError(par_name=par_name)

        data = data.dropna() if is_dropna else data

        return data


    @staticmethod
    def check_member(
        par_name: str, 
        valid_items: list[str | int | float], 
        is_string: bool = True,
        to_lower: bool = True,
        user_input: Any = 'user_input', 
        default: str | int | float | None = None
    ) -> list | tuple:
        """
        Validate if the user input belongs to a list or tuple of valid 
        items.

        Parameters
        ----------
        par_name : str
            Name to use in error messages to describe the parameter being 
            checked.
        valid_items : array_like
            List or tuple of valid items.
        is_string : bool, optional (default=True)
            Whether or not `user_input` is string.
        to_lower : bool, optional (default=True)
            If `True`, `user_input` will be converted to lowercase.
        user_input : Any
            The input to validate.
        default: {None, str, int, float}, optional (default=None)
            Default value if `user_input` is not a member of the valid items.

        Raises
        ------
        NotMemberError
            If the `user_input` is not a member of the valid items and 
            `default` is `None`.

        Returns
        -------
        user_input: str
            The validated `user_input`.
        """
        try:
            user_input = round(user_input, 15) # avoid floating decimals
        except:
            pass
        if is_string:
            user_input = ValidateArgs.check_string(
                par_name=par_name, user_input=user_input
            )
        if user_input not in valid_items:
            if default:
                user_input = default
            else:
                raise NotMemberError(
                    par_name=par_name, 
                    valid_items=valid_items, 
                    user_input=user_input
                )
        user_input = (
            user_input.lower() 
            if isinstance(user_input, str) and to_lower else user_input
        )
        
        return user_input


    @staticmethod
    def check_string(
        par_name: str, 
        to_lower: bool = True, 
        default: str = None, 
        user_input: Any = 'user_input'
    ) -> str:
        """
        Validate if the input is a string and optionally convert it to 
        lowercase.

        Parameters
        ----------
        par_name : str
            Name to use in error messages to describe the parameter being 
            checked.
        user_input : Any
            The input to validate.
        to_lower : bool, optional (default=False)
            If `True`, `user_input` will be converted to lowercase.
        default : str, optional (default=None)
            Default string to be used if `user_input` is empty.

        Raises
        ------
        TypeError
            If the input is not a string.

        Returns
        -------
        user_input : str
            The validated user input (optionally converted to lowercase).
        """
        if user_input is None:
            user_input = ''
        to_lower = ValidateArgs.check_boolean(
            user_input=to_lower, default=False
        )
        if not isinstance(user_input, str):
            raise TypeError(
                f"'{par_name}' must be a string, "
                f"got >> {type(user_input).__name__}"
            )
        user_input = (
            default if not user_input and default is not None else user_input
        )
        user_input = user_input.lower() if to_lower else user_input

        return user_input


    @staticmethod
    def check_numeric(
        par_name, 
        limits: list[int | float] = None, 
        boundary: Literal['inclusive', 'exclusive'] = 'inclusive', 
        is_positive: int = False,
        is_integer: bool | None = None, 
        to_float: bool | None = None, 
        user_input: Any = 'user_input'
    ) -> int | float | Integer | Float:
        """
        Validate numeric values.

        Parameters
        ----------
        par_name : str
            Name to use in error messages to describe the parameter being 
            checked.
        limits : list[int | float], optional (default=None)
            List-like representing the lower and upper limits of the 
            value.
        boundary : {'inclusive', 'exclusive'}, optional (default='inclusive')
            String indicating whether the limits include boundary values 
            (inclusive / closed interval) or exclude boundary values 
            (open interval).
        is_positive : bool, optional (default=False)
            If `True`, the value must be positive, otherwise a TypeError 
            error is raised.
        is_integer : bool, optional (default=None)
            If `True`, `user_input` will be converted to integer.
        to_float : bool, optional (default=None)
            If `True`, `user_input` will be converted to float.
        user_input : Any, optional (default=None)
            The numeric input to validate.

        Raises
        ------
        FloatifyError
            If the input cannot be converted to a float.
        VectorLengthError
            If the length of the limits vector is not `2`.
        ValueError
            If the input value is negative yet `is_positive` is `True`.
        NumericError
            If the input value is not within the interval specified by 
            `limits`.
        
        Returns
        -------
        user_input : {int, float, Integer, Float}
            The validated user input.
        """
        if user_input is None:
            raise RequiredError(par_name=par_name)
        try:
            # will crush if not numeric
            _ = float(sympify(user_input))
        except Exception:
            raise FloatifyError(par_name=par_name, user_input=user_input)
        
        if limits is not None:
            try:
                a, b = limits
            except Exception:
                raise VectorLengthError(
                    par_name=par_name, n=2, user_input=user_input
                )
            if not (a <= user_input <= b):
                boundary = ValidateArgs.check_member(
                    par_name='boundary', 
                    valid_items=['inclusive', 'exclusive'], 
                    default='inclusive'
                )
                raise NumericError(
                    par_name=par_name, 
                    limits=[a, b], 
                    boundary=boundary, 
                    is_integer=is_integer,
                    user_input=user_input
                )

        is_positive = ValidateArgs.check_boolean(
            user_input=is_positive, default=False
        )
        if is_positive and user_input < 0:
            raise ValueError(
                f"Expected '{par_name}' to be positive but got: {user_input}"
            )
        if to_float is not None:
            to_float = ValidateArgs.check_boolean(
                user_input=to_float, default=True
            )
            if to_float:
                try:
                    user_input = float(user_input)
                except Exception:
                    raise FloatifyError(
                        par_name=par_name, user_input=user_input
                    )
        
        if is_integer:
            try:
                user_input = int(ceil(user_input))
            except Exception:
                raise IntegifyError(par_name=par_name, user_input=user_input)
            
        return user_input


    @staticmethod
    def check_boolean(user_input: Any, default: bool) -> bool:
        """
        Validate boolean values. If invalid, then use the default.

        Parameters
        ----------
        user_input : Any
            The input to validate.
        default : bool
            The default value to use if the `user_input` is not boolean.

        Returns
        -------
        user_input : bool
            The validated boolean value or the default if the `user_input`  
            is not a boolean.
        """
        user_input = user_input if isinstance(user_input, bool) else default
        
        return user_input


    @staticmethod
    def check_member_of(par_name: str, user_input: Any, valid_values: list):
        """
        Remove invalid values from the input list or convert a string 
        input to a list.

        Parameters
        ----------
        user_input : Any
            The input to process.
        valid_values : list_like
            List of valid values to filter against.

        Returns
        -------
        user_input : list
            List containing valid values after removing any invalid values 
            or converting a string input.
        """
        from stemlab.core.arraylike import list_join, conv_to_arraylike
        try:
            user_input = [user_input] if isinstance(user_input, str) else user_input
            entered_values = conv_to_arraylike(array_values=user_input)
            valid_values = conv_to_arraylike(array_values=valid_values)
            # remove values that may not be in valid_values
            user_input = [item for item in entered_values if item in valid_values]
            # get invalid values
            invalid_values = [
                value for value in entered_values if value not in user_input
            ]
            if invalid_values or not user_input:
                invalid_values = list_join(
                    lst=invalid_values, delimiter=", ", html_tags=False
                )
                valid_values = list_join(
                    lst=valid_values, delimiter=', ', html_tags=False
                )
                raise ValueError(
                    f"Expected {par_name} to be {valid_values} but got "
                    f"{invalid_values} as invalid values"
                )
        except Exception as e:
            raise ValueError(e)
        
        return user_input


    @staticmethod
    def check_len_equal(
        x: ListArrayLike, 
        y: ListArrayLike, 
        par_name: list = ['x', 'y']
    ) -> tuple[ListArrayLike, ListArrayLike]:
        """
        Check if arrays have equal number of elements.

        Parameters
        ----------
        x : array_like
            First list or tuple for comparison.
        y : array_like
            Second list or tuple for comparison.
        par_name : list-like, optional (default=['x', 'y'])
            Name to use in error messages to describe the parameter being 
            checked.

        Returns
        -------
        tuple
            A tuple with the two input lists `x`, and `y`.

        Raises
        ------
        LengthDifferError
            If the number of elements in arrays `x` and `y` are not equal.
        """
        from stemlab.core.arraylike import is_len_equal
        
        if not is_len_equal(x, y, par_name=par_name):
            raise LengthDifferError(par_name=par_name, user_input=[x, y])
        
        return x, y


    @staticmethod
    def check_diff_constant(
        user_input: Any, par_name: str = 'x', decimal_points: int = -1
    ) -> list:
        
        """
        Validate whether the difference between consecutive elements of 
        a list is constant.

        Parameters
        ----------
        user_input : list
            A list of numbers.
        par_name : str, optional (default='x')
            Name to use in error messages to describe the parameter being 
            checked.
        decimal_points : int, optional (default=8)
            Number of decimal points or significant figures for symbolic
        expressions.

        Returns
        -------
        user_input : list
            The validated list of numbers if the difference between 
            consecutive elements is constant.

        Raises
        ------
        DifferenceError
            If the difference between consecutive elements is not constant.

        Examples
        --------
        >>> stm.check_diff_constant([1, 3, 5, 7, 9])
        [1, 3, 5, 7, 9]

        >>> check_diff_constant([1, 3, 5, 8, 9])
        Traceback (most recent call last):
            ...
        DifferenceError: Difference between elements of 'x' must be 
        constant but got: [1, 3, 5, 8, 9]

        >>> check_diff_constant([0.1, 0.2, 0.3, 0.4, 0.5])
        [0.1, 0.2, 0.3, 0.4, 0.5]

        >>> stm.check_diff_constant([0.1, 0.3, 0.6, 1.0])
        Traceback (most recent call last):
            ...
        DifferenceError: Difference between elements of 'x' must be 
        constant but got: [1, 3, 5, 8, 9]
        """
        from stemlab.core.arraylike import  is_diff_constant
        
        if decimal_points == -1:
            # float with highest number of digits after decimal point
            decimal_points = max(
                len(str(number).split('.')[1]) if '.' in str(number) 
                else 0 for number in user_input
            ) - 1 # minus 1 is important because of round-off errors
        
        if not is_diff_constant(
            user_input=user_input, decimal_points=decimal_points):
            raise DifferenceError(par_name=par_name, user_input=user_input)
        
        return user_input


    @staticmethod
    def check_dflabels(par_name: str, user_input: Any) -> str | list[str]:
        """
        Validate DataFrame labels (rows and columns).

        Parameters
        ----------
        par_name : str
            Name to use in error messages to describe the parameter being 
            checked.
        user_input : Any
            The input to validate.

        Raises
        ------
        TypeError
            If the `user_input` is not array-like, integer or None.

        Returns
        -------
        user_input : list
            The validated `user_input`.
        """
        from stemlab.core.arraylike import is_iterable, conv_to_arraylike

        if (
            not is_iterable(array_like=user_input, includes_str=True) and 
            not isinstance(user_input, int) and not user_input is None
        ):
            raise TypeError(
                f"{user_input} is an invalid value for '{par_name}', expected "
                f"list/tuple, int, str[lower, upper, title, capitalize] or None"
            )
        
        if isinstance(user_input, str):
            return user_input
        # only for iterables, not for `int` or `None` values
        if is_iterable(array_like=user_input):
            user_input = conv_to_arraylike(user_input, par_name=par_name)

        return user_input
        

    @staticmethod
    def check_square_matrix(
        par_name: str, A: ArrayMatrixLike
    ) -> NumpyArraySympyMatrix:
        """
        Check if an array is square.

        Parameters
        ----------
        par_name : str
            Name to use in error messages to describe the parameter being 
            checked.
        A : ArrayMatrixLike
            A numpy ndarray or sympy Matrix object representing the array 
            to validate.

        Raises
        ------
        IterableError
            If the input is not iterable.
        NotArrayError
            If the input is not a 2D array (matrix).
        SquareMatrixError
            If the matrix is not square.

        Returns
        -------
        M : numpy.ndarray or sympy.Matrix
            The validated matrix.
        """
        from stemlab.core.arraylike import is_iterable

        if not is_iterable(array_like=A):
            raise IterableError(
                par_name=par_name, includes_str=False, user_input=A
            )
        try:
            nrow, ncol = asarray(A).shape
        except ValueError:
            raise NotArrayError(
                par_name=par_name, 
                array_type="2D array (matrix)", 
                object_type="values", 
                user_input=A
            )
        if nrow != ncol:
            raise SquareMatrixError(par_name=par_name, dims=[nrow, ncol])
        
        return A


    @staticmethod
    def check_coeff_const(
        A: NumpyArray, b: NumpyArray
    ) -> tuple[NumpyArray, NumpyArray]:
        """
        Validate matrix of coefficients and the vector of constants.

        Parameters
        ----------
        A : numpy.ndarray
            Matrix of coefficients.
        b : numpy.ndarray
            Vector of constants.

        Raises
        ------
        CoeffConstantsError
            If the number of rows in `A` does not match the length of `b`.

        Returns
        -------
        tuple
            Tuple containing the validated matrix of coefficients `A` 
            and the vector of constants `b`.
        """
        try:
            A = asarray(A)
            b = asarray(b)
        except Exception as e:
            raise ValueError(f"Both 'A' and 'b' must be array_like. {e}")
        try:
            nrows = A.shape[0]
        except Exception:
            raise TypeError(
                f"Expected 'A' to be a square matrix, got {type(A).__name__}"
            )
        
        A = ValidateArgs.check_square_matrix(par_name='A', A=A)
        
        if nrows != len(b): # use len(b) instead of b.shape[0]
            raise CoeffConstantsError(user_input=[A, b])

        return A, b


    @staticmethod
    def check_interval(
        lower: int | float, upper: int | float, interval: int | float
    ) -> tuple[int | float, int | float, int | float]:
        """
        Check that `interval` is not greater than `|lower - upper|`.
        
        Parameters
        ----------
        lower : {int, float}
            Lower bound of the interval.
        upper : {int, float}
            Upper bound of the interval.
        interval : {int, float}
            The step-size.
        
        Returns
        -------
        tuple
            A tuple with the entered values after validation.
        """
        lower = ValidateArgs.check_numeric(par_name='lower', user_input=lower)
        upper = ValidateArgs.check_numeric(par_name='upper', user_input=upper)
        interval = ValidateArgs.check_numeric(
            par_name='interval', user_input=interval
        )
        if interval >= abs(upper - lower):
            raise IntervalError(par_name='interval', gte=True)
        
        return lower, upper, interval


    @staticmethod
    def check_limits(
        lower: int | float, 
        upper: int | float, 
        lower_par_name: str = 'a', 
        upper_par_name: str = 'b', 
        user_input: list[int | float] = [-inf, inf]
    ) -> tuple[int | float, int | float]:
        """
        Check that `lower` is less than `upper`.
        
        Parameters
        ----------
        lower : {int, float}
            Lower bound of the interval.
        upper : {int, float}
            Upper bound of the interval.
        lower_par_name : str, optional (default='a')
            Name to use in error messages to describe the parameter being 
            checked.
        upper_par_name : str, optional (default='b')
            Name to use in error messages to describe the parameter being 
            checked.
        user_input : List-like
            Input to be validated.
        
        Returns
        -------
        tuple
            A tuple with the entered values after validation.
        """
        lower = ValidateArgs.check_numeric(
            par_name=lower_par_name, user_input=lower
        )
        upper = ValidateArgs.check_numeric(
            par_name=upper_par_name, user_input=upper
        )
        if lower >= upper:
            raise LowerGteUpperError(
                par_name='Limits', 
                lower_par_name=lower_par_name, 
                upper_par_name=upper_par_name, 
                user_input=user_input
            )

        return lower, upper


    @staticmethod
    def check_identical(
        x: ListArrayLike,
        y: ListArrayLike,
        x_par_name: str = 'x',
        y_par_name: str = 'y'
    ) -> tuple[ListArrayLike, ListArrayLike]:
        """
        Check if `x` and `y` are identical.
        
        Parameters
        ----------
        x : array_like
            First array of values.
        y : array_like
            Second array of values.
        x_par_name : str, optional (default='x')
            Name to use in error messages to describe the parameter
            being checked.
        y_par_name : str, optional (default='y')
            Name to use in error messages to describe the parameter
            being checked.
        
        Returns
        -------
        tuple
            A tuple with the entered values of `x` and `y` after 
            validation.
        """
        if all([x == y for x, y in zip(x, y)]):
            raise ValueError(
                f"'{x_par_name}' and '{y_par_name}' are identical"
            )
        
        return x, y


    @staticmethod
    def check_constant(par_name: str, user_input: Any) -> ListArrayLike:
        """
        Check if `user_input` if constant.
        
        Parameters
        ----------
        par_name : str
            Name to use in error messages to describe the parameter being 
            checked.
        user_input : Any
            The input to be validated.
            
        Returns
        -------
        user_input : array_like
            The validated user input.    
        """
        if len(unique(user_input)) < 2:
            raise ValueError(f"All the values of '{par_name}' are the same")

        return user_input


    @staticmethod
    def check_conf_level(user_input: Any) -> float:
        """
        Validate confidence interval values.
        
        Parameters
        ----------
        user_input : Any
            The input to be validated.
            
        Returns
        -------
        user_input : float
            Validated input.
        """
        ValidateArgs.check_member(
            par_name='conf_level', 
            valid_items=[.90, .95, .99], 
            is_string=False, 
            user_input=user_input
        )

        return user_input


    @staticmethod
    def check_sig_level(user_input: Any) -> float:
        """
        Validate significant level values.
        
        Parameters
        ----------
        user_input : Any
            The input to be validated.
            
        Returns
        -------
        user_input : float
            Validated input.
        """
        ValidateArgs.check_member(
            par_name='sig_level', 
            valid_items=[.10, .05, .01],
            is_string=False,
            user_input=user_input
        )

        return user_input


    @staticmethod
    def check_alternative(user_input: Any) -> str:
        """
        Validate alternative options.
        
        Parameters
        ----------
        user_input : Any
            The input to be validated.
            
        Returns
        -------
        user_input : str
            Validated input.
        """
        user_input = ValidateArgs.check_member(
            par_name='alternative', 
            valid_items=['less', 'two-sided', 'greater'], 
            user_input=user_input
        )

        return user_input


    @staticmethod
    def check_function(
        f: Any,
        is_univariate: bool = False,
        variable: list[str] | None = None,
        par_name='f'
    ) -> Callable:
        """
        Check if the given argument is a callable function and return
        it.
        
        Parameters
        ----------
        f : Any
            The input to validated.
        is_univariate : bool, optional (default=False)
            Whether the equation is univariate.
        variables : array_like, optional (default=None)
            List of variable names in the equation.
        par_name : str, optional (default='fexpr')
            Name to use in error messages to describe the parameter 
            being checked.
        
        Returns
        -------
        f : Callable
            The `user_input` converted to a function.
            
        Raises
        ------
        Exception
            If the input argument `f` is not a callable function.
        
        Notes
        -----
        This function ensures that the provided argument is a callable  
        function. If it is not, it raises a `Exception` with a 
        descriptive message.
        """
        from stemlab.core.symbolic import sym_lambdify_expr
        try:
            f = sym_lambdify_expr(
                fexpr=f,
                is_univariate=is_univariate,
                variables=variable,
                par_name=par_name
            )
        except Exception as e:
            raise e
        
        return f
    
    
    @staticmethod
    def check_divisibility(n: int, divisor: int, method: str) -> None:
        """
        Check if a number is divisible by a given divisor using 
        a specified method.

        Parameters
        ----------
        n : int
            The number to check for divisibility.
        divisor : int
            The divisor to check divisibility against.
        method : str
            The method used for checking divisibility.

        Returns
        -------
        None

        Raises
        ------
        UserWarning
            If 'n' is not divisible by the given divisor according to 
            the specified method.
            
        """

        if n % divisor != 0:
            label = 'even' if divisor == 2 else f'a multiple of {divisor}'
            message = (
                f"'{method}' rule is best suited when 'n' is {label} "
                f"but got {n}"
            )
            warnings.warn(message, UserWarning)