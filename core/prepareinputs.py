from sympy import Matrix, sympify, flatten, Integer
from numpy import array, ravel, float64, nan
from pandas import DataFrame

from stemlab.core.symbolic import is_symexpr
from stemlab.core.datatypes import (
    ArrayMatrixLike, NumpyArraySympyMatrix, is_nested_list
)
from stemlab.core.htmlatex import sta_dframe_to_html

ArrayMatrixLike

def split_input(
    user_input: str | list,
    flatten_array: bool = False,
    np_array: bool = False,
    sym_matrix: bool = False,
    unequal_lists: bool = False,
    dict_user_inputs: dict = {},
    dict_update: bool = True,
) -> tuple[int | list | NumpyArraySympyMatrix | dict]:
    """
    Splits user input into object name and values.

    Parameters
    ----------
    user_input : {str, list_like}
        Input string or list.
    flatten_array : bool, optional (default=False)
        Whether to flatten array.
    np_array : bool, optional (default=False)
        Whether to convert to NumPy array.
    sym_matrix : bool, optional (default=False)
        Whether to convert to SymPy matrix.
    unequal_lists : bool, optional (default=False)
        Whether to allow unequal lists, by default False.
    dict_user_inputs : dict, optional  (default={})
        Dictionary to store user inputs.
    dict_update : bool, optional (default=True)
        Whether to update the dictionary.

    Returns
    -------
    tuple
        Tuple containing object values and updated dictionary.
    """
    strings = (
        ["lambda", "beta", "gamma"]
        + ["E", "I", "N", "O", "Q", "S"]
        + [
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atan2",
            "atanh",
            "cos",
            "cosh",
            "exp",
            "log",
            "log2",
            "log10",
            "pi",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
            "Heaviside",
        ]
    )
    user_input = user_input.replace('Q:=', 'Q :=')
    for strng in strings:
        for char in [",", ")", "]", " ", "+", "-", "*", "/"]:
            user_input = user_input.replace(
                f"{strng}{char}", f"{strng}_{char}"
            )
    # takes care of inequalities
    user_input = user_input.replace(">=", "gte").replace("<=", "lte")
    if ":=" in user_input:
        object_name, object_values = user_input.split(":=", maxsplit=1)
    else:
        object_name, object_values = "In", user_input

    if "=" in object_values:
        object_values = object_values.split("=", maxsplit=1)
        object_values = f"{object_values[0]} - ({object_values[1]})"
    # reverses what was done above
    object_values = object_values.replace("gte", ">=").replace("lte", "<=")
    object_name = "".join(object_name.split())
    object_values = sympify(" ".join(object_values.split()))

    if flatten_array:
        try:
            try:
                object_values = flatten(object_values)
            except Exception:  # this could be a number, not a list
                pass
        except Exception as e:
            raise Exception(e)

    if object_name == "In":
        if isinstance(object_values, (list, tuple)):
            if is_nested_list(object_values):  # matrix
                object_name = "M"
            else:  # vector
                object_name = "v"
        else:
            if is_symexpr(object_values):  # function
                object_name = "f"
            else:  # scalar
                object_name = "k"
    if dict_update:
        try:
            # this avoids duplicates if the name already exists
            if (object_name in dict_user_inputs.keys()):  
                object_name = f"{object_name}{len(dict_user_inputs)}"
            # exclude symbolic expressions, operations here are meant 
            # for arrays
            if not is_symexpr(object_values):
                try:
                    if flatten_array:
                        object_values_ = flatten(object_values)
                    else:
                        object_values_ = Matrix(object_values).tolist()
                # any that may not pass the above operations, e.g. scalar
                except Exception:
                    object_values_ = object_values
            else:
                object_values_ = object_values

            # display only the first 25 rows or values
            try:
                # arrays
                dict_user_inputs[object_name] = object_values_[:25, :25]
            except Exception:
                try:
                    # lists
                    dict_user_inputs[object_name] = object_values_[:25]
                except Exception:
                    try:
                        # equations
                        dict_user_inputs[object_name] = object_values_
                    except Exception:
                        pass
        except Exception as e:
            raise Exception(e)

    if not is_symexpr(object_values):
        if unequal_lists:
            object_values = list(sympify(object_values))
        else:
            if np_array:
                object_values = array(object_values, dtype=float64)
                if flatten_array:
                    object_values = ravel(object_values)

            if sym_matrix:
                object_values = Matrix(object_values)
                if flatten_array:
                    object_values = flatten(object_values)

    if type(object_values) == Integer:
        object_values = int(object_values)

    return (object_values, dict_user_inputs)

def unequal_lists(
    user_input,
    row_label: str = "Group",
    dict_user_inputs: str = {},
    dict_update: str = True
) -> tuple[int | list | NumpyArraySympyMatrix | dict]:
    """
    Processes input containing unequal lists and creates a DataFrame.

    Parameters
    ----------
    user_input : str
        Input string.
    row_label : str, optional (default="Group")
        Label for rows.
    dict_user_inputs : dict, optional (default={})
        Dictionary to store user inputs.
    dict_update : bool, optional (default=True)
        Whether to update the dictionary.

    Returns
    -------
    tuple
        Tuple containing processed object values and updated dictionary.
    """

    # takes care of inequalities
    user_input = user_input.replace(">=", "gte").replace("<=", "lte")
    if ":=" in user_input:
        object_name, object_values = user_input.split(":=", maxsplit=1)
    else:
        object_name, object_values = "In", user_input

    if "=" in object_values:
        object_values = object_values.split("=", maxsplit=1)
        object_values = f"{object_values[0]} - ({object_values[1]})"

    object_values = object_values.replace("gte", ">=").replace(
        "gle", "<="
    )  # reverses what was done above

    object_values = array(
        sympify(" ".join(object_values.split())), dtype=object
    )

    if dict_update:
        max_length = max(len(sublist) for sublist in object_values)
        lists_na = []
        for k in range(len(object_values)):
            if len(object_values[k]) < max_length:
                n_diff = max_length - len(object_values[k])
                new_list = list(object_values[k]) + [nan] * n_diff
                lists_na.append(list(new_list))
            else:
                lists_na.append(object_values[k])

        lists_na = array(lists_na)
        df = DataFrame(lists_na)
        df = df.fillna("")
        df.index = [f"{row_label}{k + 1}" for k in range(lists_na.shape[0])]
        df.columns = range(1, lists_na.shape[1] + 1)
        dict_user_inputs[object_name] = sta_dframe_to_html(df, row_title=row_label)

    return object_values, dict_user_inputs