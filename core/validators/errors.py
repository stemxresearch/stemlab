import traceback
import sys
import re

import numpy as np
from sympy import flatten
from pandas import DataFrame
from stemlab.core.base.strings import str_singular_plural, str_strip_all
from stemlab.core.base.constraints import max_rows


def error_split(error_except: str):

    if str(error_except).count("<<>>") >= 2:
        return str(error_except).split("<<>>")
    else:
        return ["<strong>ERROR</strong>", error_except, "&nbsp;"]


def detect_link(strng: str):

    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, strng)
    return [x[0] for x in url]


def split_get_first(string_label: str) -> str:
    """
    Splits a string by '>>>>' and returns the first part with all whitespaces 
    removed.
    
    Parameters
    ----------
    string_label: str
        The input string to be processed.
    
    Returns
    -------
    str 
        The first part of the input string after splitting by '>>>>'
        and stripping leading and trailing whitespace.

    Examples
    --------
    label1 = "First part >>>> Second part"
    stm.split_get_first(label1)

    label2 = "Only one part"
    stm.split_get_first(label2)

    label3 = "   Spaces before and after >>>>     Second part   "
    stm.split_get_first(label3)
    """
    try:
        string_label = string_label.replace(" >>>> ", ">>>>").split(">>>>")[0]
    except Exception:
        pass
    string_label = str_strip_all(string_label)
    
    return string_label


def join_error_list(par_name: str, error_message: str, to_list_join: bool):

    if to_list_join:
        if '-->' in error_message: 
            message, user_input = error_message.split(" but got: ")
            if message.endswith(','):
                message = message[:-1]
            error_message = [par_name, f"{message}", f"{user_input}"]
            error_message = error_message.replace("left_", "<strong>")
            error_message = error_message.replace("_right", "</strong>")
        else:
            error_message = [par_name, error_message, "&nbsp;"]
    else:
        error_message = error_message.replace("left_", "'")
        error_message = error_message.replace("_right", "'")
    error_message = '<<>>'.join(map(str, [error_message]))
    error_message = error_message.replace(' input_field', '')
    error_message = error_message.replace(".,", ".")

    return error_message


def ErrorModal(
        error_list: list,
        error_bg: str = ' orange',
        error_color: str = 'brown',
        tab_field: bool = True, 
        query_string: str = ''
    ):

    if isinstance(error_list, Exception):
        error_list = str(error_list)
    if isinstance(error_list, str):
        if "<<>>" not in error_list:
            error_list = f'API error: UE<<>>{error_list}<<>>&nbsp;'
        error_list = error_list.split("<<>>")
    try:
        error_list = [np.array(error_list, dtype=object).ravel().tolist()]
        error_table = []
        if len(error_list) == 1:
            error_correct = "this error"
            errors_found = "error was"
        else:
            error_correct = (
                f"these <strong>{len(error_list)} errors<strong>"
            )
            errors_found = (
                f"<strong>{len(error_list)}</strong> errors were"
            )

        error_table.append(
            '\n\n\t\t<p style="margin-bottom:0px;padding:0px;'
            'padding-bottom:5px;line-height:23px;">\n\t\t\tThe '
            f'following {errors_found} encountered while processing your '
            'request.\n\t\t</p>\n\n'
        )
        error_table.append(
            '\t\t<table class="table table-bordered" '
            'style = "border:1px solid #ccc;min-width:150px;'
            'max-width:600px;font-size:15px;">\n\t\t\t'
            '<tr style = "background:aliceblue !important;'
            'line-height:23px;">\n\t\t\t\t<th align = "left" '
            'style="border:1px solid #ccc;width:30%;'
            'background:aliceblue;"><span style = "white-space:nowrap;'
            'margin-left:0px;">Field label</span> </th>\n\t\t\t\t'
            '<th align = "left" style="background:aliceblue;">'
            '<div style = "white-space:nowrap;margin:0px 0px;">'
            'Error message</div></th>\n\t\t\t</tr>\n'
        )

        for item in error_list:
            try:
                label_name = item[0].replace(" >>>> ", ">>>>").split(">>>>")[0]
                tab_name = item[0].replace(" >>>> ", ">>>>").split(">>>>")[1]
            except Exception:
                label_name = item[0]
                tab_name = "Main tab"

            error_table.append(
                '\t\t\t<tr style="line-height:20px;border:1px solid #ccc;">\n'
            )
            error_table.append(
                f'\t\t\t\t<td rowspan = "2" style = "background:white;'
                'border:1px solid #ccc;"><div style="margin:margin:3px 0px;">'
                f'<strong>{label_name}</strong></div></td>\n'
            )
            error_table.append(
                f'\t\t\t\t<td><div style = "font-weight:normal;margin:3px 0px;">'
                f'{item[1]}</div></td>\n'
            )
            error_table.append("\t\t\t</tr>\n")
            error_table.append('\t\t\t<tr style="line-height:23px;">\n')
            error_table.append(
                '\t\t\t\t<td style = "background:white;">'
                f'<div style = "font-weight:600;color:{error_color};margin:margin:3px 0px;">'
                f'{str(item[2])}</div></td>\n'
            )
            error_table.append("\t\t\t</tr>\n")

        error_table.append("\t\t</table>\n\n")
        error_table = (
            f'<div style="margin-top:10px;">{" ".join(error_table)}</div>'
        )
        tab_field_str = ''
        if tab_field:
            tab_field_str = (
                f'in the <strong>{label_name}</strong> '
                f'field found in the <strong>{tab_name}</strong>'
            )
        if "API error" in label_name:
            error_below = (
                '\t\t<p style = "margin-top:10px;margin-bottom:0px;padding-bottom:5px;">'
                '\n\t\t\tPlease check your inputs and correct the above '
                'error then try again. If you are sure that your inputs '
                'are correctly entered, you could try '
                f'<a style="color:blue;font-weight:600;" href="/pages/report-errors?pageLink={query_string.replace("&", "AND")}" target="_blank">' 
                'reporting this error</a>.\n\t\t</p>\n\n'
            )
        else:
            error_below = (
                '\t\t<p style = "margin-top:5px;margin-bottom:0px;padding-bottom:5px;">'
                f'\n\t\t\tPlease correct {error_correct} {tab_field_str} '
                'then try again.\n\t\t</p>\n\n'
            )
        error_heading = (
            f'<div style="font-weight:600;background:{error_bg};padding:3px;padding-left:7px;">'
            'Request Failed to Execute</div>'
        )
        error_message = (
            '\n\n\t<div style="background:white;border:0px solid #ccc;color:black;">'
            f'{error_heading}{error_table}<div style="max-width:600px;">'
            f'{error_below}</div>\t</div>\n\n'
        )
    except Exception as error_except :
        error_message = (
            f'<p><strong>Request Failed to Execute</strong></p>'
            f'<p class="py-2 pb-0" style="font-weight:600;color:brown;">'
            f'Your request failed with the error: '
            f'{str(error_except).capitalize()}.</p>'
        )

    return error_message


class IterableError(TypeError):
    def __init__(
            self, 
            par_name: str = 'object', 
            includes_str: bool = False, 
            user_input: str = 'user_input', 
            to_list_join: bool = False
    ):
        self.user_input = user_input
        self.par_name = split_get_first(par_name)
        self.data_types = 'str, tuple, list, set, NDArray, Series'
        if not includes_str:
            self.data_types = 'tuple, list, set, NDArray, Series'
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Expected left_{self.par_name}_right to be one of: "
            f"{self.data_types} but got: {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class RowColumnsLengthError(Exception):
    def __init__(
            self, 
            par_name: str = 'col_names', 
            rows: bool = True, 
            dframe: DataFrame | None = None, 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.rows = rows
        self.dframe = DataFrame(dframe)
        self.to_list_join = to_list_join
    def __str__(self):
        nrows, ncols = self.dframe.shape
        label_name = 'rows' if self.rows else 'columns'
        ndims = nrows if self.rows else ncols
        expected = (
            f", expected {ndims} {label_name} labels" 
            if self.dframe is not None else ""
        )
        par_name = (
            f" in left_{self.par_name}_right" 
            if self.par_name is not None else ""
        )
        error_message = (
            f'Number of elements{par_name} must be equal to the number of '
            f'{label_name} in the DataFrame{expected}'
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
           

class RequiredError(ValueError):
    def __init__(
        self, 
        par_name: str, 
        required_when: str | None = None, 
        to_list_join: bool = False):
        self.par_name = split_get_first(par_name)
        self.required_when = required_when
        self.to_list_join = to_list_join
    def __str__(self):
        required = (
            f". It is required when '{self.required_when}'" 
            if self.required_when is not None else ""
        )

        error_message = (
            f"You have not provided an argument for "
            f"Expected left_{self.par_name}_right{required}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class EmptyFieldError(ValueError):
    def __init__(
            self,
            par_name: str,
            required_when: str | None = None,
            required_where: str | None = None,
            tabname: str = 'Main', 
            field_type: str = 'input',
            to_list_join: bool = False     
        ):

        self.par_name = split_get_first(par_name)
        self.required_when = required_when
        self.required_where = required_where
        self.tabname = tabname
        self.field_type = field_type
        self.to_list_join = to_list_join
    def __str__(self):
        enter_select = (
            "selected a value for " if self.field_type == "dropdown" 
            else "provided a value for "
        )
        if self.required_when is None:
            required_when_str = ""
        else:
            if self.tabname:
                tabname = f" of the left_{self.tabname}_right tab"
            else:
                tabname = "."
            required_when_str = (
                f"This value is required when {self.required_when} "
                f"is specified in left_{self.required_where}_right "
                f"input field{tabname}"
            )
        error_message = (
            f"You have not {enter_select} left_{self.par_name}_right "
            f"{self.field_type} field. {required_when_str} but got: blank"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
        

class MaxNError(ValueError):
    def __init__(
        self, 
        par_name: str, 
        user_input, maxn: int | None = None, 
        to_list_join: bool = False
    ):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.maxn = max_rows() if self.maxn is None else maxn
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Expected left_{self.par_name}_right to be less than or equal to "
            f"{self.maxn} but got: {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )
        
        return error_message
     

class NotInColumnsError(ValueError):
    def __init__(self, par_name: str, user_input, to_list_join: bool = False):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"'{self.user_input}' as specified in left_{self.par_name}_right "
            "is not one of the DataFrame columns"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class DataFrameError(TypeError):
    def __init__(
            self, par_name: str, user_input: str, to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Expected left_{self.par_name}_right to be a DataFrame "
            f"but got: {type(self.user_input).__name__}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
       

class StringError(TypeError):
    def __init__(self, par_name:str, user_input, to_list_join: bool = False):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join=to_list_join
    def __str__(self):
        error_message = (
            f"Expected left_{self.par_name}_right to be a string but got "
            f"{type(self.user_input).__name__}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class StringFoundError(TypeError):
    def __init__(self, par_name:str, user_input, to_list_join: bool = False):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join=to_list_join
    def __str__(self):
        error_message = (
            f"Expected numeric value(s) for left_{self.par_name}_right "
            f"but got {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message

class NumpifyError(ValueError):
    def __init__(
            self, par_name: str = 'input', to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Unable to convert left_{self.par_name}_right to a valid array"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class PandifyError(ValueError):
    def __init__(self, par_name: str, to_list_join: bool = False):
        self.par_name = split_get_first(par_name)
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Unable to convert left_{self.par_name}_right to a DataFrame"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class SerifyError(ValueError):
    def __init__(self, par_name: str, to_list_join: bool = False):
        self.par_name = split_get_first(par_name)
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Unable to convert left_{self.par_name}_right to a Series"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class FloatifyError(ValueError):
    def __init__(self, par_name:str, user_input, to_list_join: bool = False):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Unable to convert the value of left_{self.par_name}_right to "
            f"a float (decimal) but got: {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class IntegifyError(ValueError):
    def __init__(self, par_name:str, user_input):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
    def __str__(self):
        error_message = (
            f"Unable to convert the value of left_{self.par_name}_right to "
            f"an integer (whole number) but got: {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
 

class SympifyError(ValueError):
    def __init__(
            self, 
            par_name: str | None = None, 
            to: str = 'a symbolic expression', 
            user_input: str = 'value', 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
        self.to = to
    def __str__(self):
        if self.par_name is None:
            error_message = (
                f"Failed to convert '{self.user_input}' to {self.to}"
            )
        else:
            error_message = (
                f"Expected left_{self.par_name}_right to be a valid {self.to} "
                f" but got: {self.user_input}"
            )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class SymbolicExprError(ValueError):
    def __init__(
            self, 
            par_name:str, 
            user_input, 
            variable: bool = False, 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.variable = variable
        self.to_list_join = to_list_join
    def __str__(self):
        symbolic_type = (
            "have at least one unknown variable" if self.variable 
            else "be a variable"
        )
        error_message = (
            f"Expected left_{self.par_name}_right input field must "
            f"{symbolic_type} but got: {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class SymbolicVarsError(ValueError):
    def __init__(
            self, par_name: str = 'the expression', 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Unable to obtain symbolic variables from " 
            f"Expected left_{self.par_name}_right"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class NotSymbolicVar(TypeError):
    def __init__(
            self, par_name: str, user_input: str, to_list_join: bool = False
        ):
        self.user_input = user_input
        self.par_name = split_get_first(par_name)
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Expected left_{self.par_name}_right to be a symbolic variable "
            f"but got {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
   

class ScalarError(ValueError):
    def __init__(
            self, par_name: str, user_input,  to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"List/tuple not allowed in the left_{self.par_name}_right. Only "
            f"a scalar (single value) is allowed but got: {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class ToleranceError(ValueError):
    def __init__(self, par_name: str, user_input):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
    def __str__(self):
        error_message = (
            f"Expected {self.par_name } input field to be less than 1 "
            f"but got {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class NumericError(ValueError):
    def __init__(
            self,
            par_name,
            limits: list | tuple | None = None,
            boundary: str = 'inclusive',
            is_integer: bool | None = None,
            user_input=None,
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.limits = limits
        self.boundary = boundary
        self.is_integer = is_integer
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        if self.is_integer is None:
            integer_float = "a numeric value"
        else:
            integer_float = "a float (number with decimal points)"
            if self.is_integer:
                integer_float = "an integer (whole number)"
        if self.limits is not None:
            a, b = self.limits
            error_message = (
                f"Expected left_{self.par_name}_right to be {integer_float} "
                f"between {a} and {b} {self.boundary} but got: "
                f"{self.user_input}"
            )
        else:
            error_message = (
                f"Expected left_{self.par_name}_right to be {integer_float} "
                f"but got {self.user_input}"
            )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class NumericVectorError(ValueError):
    def __init__(self, par_name:str, user_input, to_list_join: bool = False):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Expected all values in left_{self.par_name}_right to be numeric "
            f"but got {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class BooleanError(ValueError):
    def __init__(self, par_name: str, user_input, to_list_join: bool = False):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Expected left_{self.par_name}_right to be boolean (True/False) "
            f"but got {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class UnivariatePolyError(ValueError):
    def __init__(self, par_name: str, user_input, to_list_join: bool = False):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Expected left_{self.par_name}_right to be a univariate "
            f"polynomial but got {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class ListTupleError(TypeError):
    def __init__(
            self, 
            par_name:str, 
            lst: bool = False, 
            user_input=None, 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.lst = lst
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        if self.lst:
            error_message = (
                f"You have entered a single element where at least two "
                f"elements are required but got: {self.user_input}"
            )
        else:
            error_message = (
                f"Expected left_{self.par_name}_right to be a list, tuple, or 1D "
                f"array but got: {self.user_input}"
            )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class SingularMatrixError(ValueError):
    def __init__(self, par_name: str, user_input, to_list_join: bool = False):
        self.par_name = split_get_first(par_name)
        self.user_input = str(user_input)
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Expected left_{self.par_name}_right is a singular matrix (i.e. det = 0)"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class DictionaryError(TypeError):
    def __init__(self, par_name: str, user_input):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
    def __str__(self):
        error_message = (
            f"Expected left_{self.par_name}_right to be dictionary " 
            f"but got {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class NotMemberError(ValueError):
    def __init__(
            self, 
            par_name: str, 
            valid_items: list, 
            user_input, 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.valid_items = valid_items
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        from stemlab.core.arraylike import list_join
        valid_items = list_join(
            lst=self.valid_items, quoted=True, html_tags=False
        )
        try:
            valid_items = list_join(
                lst=self.valid_items, quoted=True, html_tags=False
            )
        except Exception:
            pass
        error_message = (
            f"Expected left_{self.par_name}_right to be one of: {valid_items}; "
            f"but got {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class VectorLengthError(ValueError):
    def __init__(
            self, 
            par_name: str, 
            n: int, 
            label: str = 'exactly', 
            user_input=None, 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.n = n
        self.label = label
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        sing_plural = str_singular_plural(n=self.n)
        error_message = (
            f"Expected left_{self.par_name}_right to have "
            f"Expected left_{self.label} {self.n}_right element{sing_plural} but got: "
            f"{self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
        

class LengthDifferError(ValueError):
    def __init__(
            self, 
            par_name: list | tuple, 
            user_input, 
            to_list_join: bool = False
        ):
        self.par_names = par_name
        self.x, self.y = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        try:
            pars = f"'{self.par_names[0]}' and '{self.par_names[1]}'"
        except Exception:
            pars = f"'x' and 'y'"
        error_message = (
            f"Expected {pars} to have the same number of elements but "
            f"got {len(self.x)} and {len(self.y)} elements respectively"
        )
        self.par_names = (
            f"{self.par_names}..." if len(self.par_names) > 16 else self.par_names
        )
        error_message = join_error_list(
            self.par_names, error_message, self.to_list_join
        )

        return error_message


class SquareMatrixError(ValueError):
    def __init__(
            self, 
            par_name: str, 
            dims: list | tuple = ['r', 'c'], 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.dims = dims
        self.to_list_join = to_list_join
    def __str__(self):
        dims = f'{self.dims[0]} rows, {self.dims[1]} columns'
        error_message = (
            f"Expected left_{self.par_name}_right to be a square matrix "
            f"but got {dims}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class MatrixCompatibilityError(ValueError):
    def __init__(
            self, 
            A: str = 'first array',
            B: str = 'second array',
            multiplication: bool = True, 
            dims_A: list | tuple = ('m', 'p'), 
            dims_B: list | tuple = ('p', 'q'),
            to_list_join: bool = False
    ):
        self.multiplication = multiplication
        self.A = A
        self.B = B
        self.dims_A = dims_A
        self.dims_B = dims_B
        self.to_list_join = to_list_join
        self.par_name = f'{self.A} and {self.B}'
    def __str__(self):
        if self.multiplication:
            text = (
                f"Expected number of columns of '{self.A}' to be equal to the "
                f"number of rows of '{self.B}' but got '{self.A}' size = "
                f" = {self.dims_A} and '{self.B}' size = {self.dims_B}"
            )
        else:
            text = (
                f"Expected both '{self.A}' and '{self.B}' to have the same "
                f"size but got '{self.A}' size = {self.dims_A} and '{self.B}' "
                f"size = {self.dims_B}"
            )
        error_message = f"The entered arrays are not compatible. {text}"
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class LowerGteUpperError(ValueError):
    def __init__(
            self, 
            par_name: str, 
            lower_par_name: str, 
            upper_par_name: str, 
            user_input=None, 
            to_list_join: bool = False
    ):
        self.par_name = par_name
        self.lower_par_name, self.upper_par_name = (lower_par_name, upper_par_name)
        self.lower, self.upper = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f'Lower limit left_{self.lower_par_name}_right cannot be greater '
            f'than or equal to upper limit left_{self.upper_par_name}_right, '
            f'but got {self.lower} and {self.upper}'
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )
                
        return error_message
    

class IntervalError(ValueError):
    def __init__(
            self, 
            par_name: str = 'h', 
            gte: bool = False, 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.gte = gte
        self.to_list_join = to_list_join
    def __str__(self):
        gt = 'greater than' + (' or equal to' if self.gte else '')
        error_message = f"Expected left_{self.par_name}_right cannot be {gt} |b - a|"
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class DifferenceError(ValueError):
    def __init__(self, par_name: str, user_input, to_list_join: bool = False):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        maxn = 10
        error_message = (
            f"Difference between elements of left_{self.par_name}_right must "
            f"be constant but got: ..."
        )
        x = flatten(list(self.user_input))
        if len(x) > maxn:
            new_x = f'{x[:maxn - 1]}, ...,, {x[-1]}]'.replace(']', '', 1)
            error_message = (
                error_message.replace('...', new_x)
            )
        else:
            error_message = error_message.replace('...', str(x))
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class NoRootInIntervalError(ValueError):
    def __init__(self, expr='', user_input='', to_list_join: bool = False):
        self.par_name=''
        self.expr = str(expr)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        a, b = self.user_input
        error_message =  (
            f"The function '{self.expr}' does not have a root in the interval "
            f"[{round(a, 8)}, {round(b, 8)}]"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class CoeffConstantsError(ValueError):
    def __init__(
            self, 
            par_name: list | tuple = ['A', 'b'], 
            user_input=None, 
            to_list_join: bool = False
        ):
        self.par_name = par_name
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        self.par_name = ', '.join(map(str, self.par_name))
        A, b = self.user_input
        size_a, size_b = (A.shape, b.shape)
        error_message = (
            "The coefficients matrix and vector of constants are not "
            f"compatible but got: 'matrix_or_eqtns' size = {size_a}, "
            f"'b' size = {size_b}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message

    
class NotSubsetError(ValueError):
    def __init__(
            self, 
            list_to_check: set | list | tuple, 
            check_in_list: set | list | tuple, 
            to_list_join: bool = False
        ):
        self.list_to_check = list_to_check
        self.check_in_list = check_in_list
        self.to_list_join = to_list_join
    def __str__(self):
        check_set = set(self.list_to_check)
        check_in_list_set = set(self.check_in_list)
        if not check_set.issubset(check_in_list_set):
            not_in_list = list(check_set.difference(check_in_list_set))
            not_in_list = ', '.join(map(str, not_in_list))
            not_in_list = (
                f"'{not_in_list[0]}' was" if len(not_in_list) == 1 
                else f"'{not_in_list}' were"
            )
        error_message = (
            f"{not_in_list} not found in {self.check_in_list}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )
        
        return error_message
    

class NormOrderError(ValueError):
    def __init__(
            self, 
            par_name: str = "la_norm_order", 
            la_norm_order: str ="", 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.la_norm_order = la_norm_order
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (f'{self.la_norm_order} is an invalid norm.')
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    
    
class NotNestedArrayError(ValueError):
    def __init__(
            self, 
            par_name: str, 
            nested_form: list, 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.nested_form = nested_form
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"The value of the {self.par_name}' input field must be "
            f"a nested array of the form: {self.nested_form}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    
class NotUnivariateError(ValueError):
    def __init__(self, par_name: str, user_input, to_list_join: bool = False):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            "The symbolic expression must be univariate (i.e. have only one "
            f"variable) but got: {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class NotArrayError(ValueError):
    """
    array_type: 1D array (vector), 2D array (matrix)
    """
    def __init__(
            self, 
            par_name: str, 
            array_type: str = "an array", 
            object_type: str = "values", 
            user_input=None, 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.array_type = array_type
        self.object_type = object_type
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Expected the value of left_{self.par_name}_right input field to "
            f"to left_{self.array_type}_right of "
            f"left_{self.object_type}_right but got {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class NotArrayOrValueError(ValueError):
    def __init__(
            self, par_name: str, user_input, to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"The value of the left_{self.par_name}_right input field must "
            "be a single numeric value or an array of values, "
            f"but got {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class NotVariableError(ValueError):
    def __init__(self, par_name: str, user_input, to_list_join: bool = False):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):   
        error_message = (
            f"Expected left_{self.par_name}_right input field must be a string "
            f"but got {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class NotRationalError(ValueError):
    def __init__(self, par_name: str, user_input, to_list_join: bool = False):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):   
        error_message = (
            f"Expected left_{self.par_name}_right input field to be a rational "
            f"symbolic expression (i.e. have numerator and denominator) but "
            f"got: {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class PositiveIntegerError(ValueError):
    def __init__(self, par_name: str, user_input, to_list_join: bool = False):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):   
        error_message = (
            f"The value of the left_{self.par_name}_right input field must "
            f"be a postive integer (whole number) but got: {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class MoreOptionsError(ValueError):
    def __init__(
            self, 
            par_name: str = 'result name and decimal_points', 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"An error occured while preparing {self.par_name} input fields"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class QueryStringError(ValueError):
    def __init__(
            self, 
            par_name: str = 'API error: QS', 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"An error occured while preparing a left_query string_right "
            "from your inputs"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class ExtractError(ValueError):
    def __init__(
            self, 
            par_name: str = "the specified",
            user_input: str = "&nbsp;",
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Unable to get values of left_{self.par_name}_right input "
            f"field but got: {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class GetPropertiesError(ValueError):
    def __init__(
            self, 
            par_name: str = "API error: Object properties",
            text: str = "computed result",
            user_input: str = "&nbsp;",
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.text = text
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Unable to get the properties of the left_{self.text}_right "
            f"but got {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class SaveToDatabaseError(ValueError):
    def __init__(
            self, 
            par_name: str = "API error: Save to database", 
            user_input: str = "&nbsp;",
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.user_input= user_input
        self.to_list_join = to_list_join
    def __init__(self):
        error_message = "Unable to save results to database."
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class UserInputsError(ValueError):
    def __init__(
            self, 
            par_name: str = "API error: User inputs", 
            user_input: str = "&nbsp",
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.user_input= user_input
        self.to_list_join = to_list_join
    def __init__(self):
        error_message = (
            "The following error occured while preparing your inputs "
            f"{self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class ExtractColumnIndicesError(ValueError):
    def __init__(
            self, 
            par_name: str = "User inputs", 
            user_input: str = "&nbsp;",
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __init__(self):
        error_message = (
            "Unable to extract column indices from the value specified in "
            f"the left_{self.par_name}_right"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class MismatchRowsColumnsError(ValueError):
    def __init__(self, rows_columns: str, row_names: str, array_rows: str):
        self.rows_columns = rows_columns
        self.row_names = row_names
        self.array_rows = array_rows
    def __str__(self):
        plural = str_singular_plural(self.array_rows)
        error_message = (
            f"The number of {self.rows_columns.lower()} you entered for the "
            f"DataFrame is not equal to the number of "
            f"{self.rows_columns.lower()} of the array. Please enter "
            f"{str(self.array_rows)} {self.rows_columns.lower().split()[0]}{plural} "
            f"label_name{plural}.<<>>{', '.join(map(str, self.row_names))} "
            f"but got {', '.join(map(str, self.row_names))}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class DataFrameLabelsError(ValueError):
    def __init__(self, par_name: str = "Row / column names"):
        self.par_name = split_get_first(par_name)
    def __str__(self):
        error_message = (
            "Unable to create row and/or column names for your DataFrame"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class DuplicateRowsColumnsError(ValueError):
    def __init__(self, rows_columns: str, row_column_names: str):
        self.rows_columns = rows_columns.lower()
        self.row_column_names = row_column_names.lower()
    def __str__(self):
        error_message = (
            f"Duplicate entries found in the {self.rows_columns} you "
            f"entered. Please enter unique {self.rows_columns}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class RowColumnNamesCountError(ValueError):
    def __init__(self, rows: bool, array_rows: int, row_names: list | tuple):
        self.rows = rows
        self.array_rows = array_rows
        self.row_names = row_names
    def __str__(self):
        plural = str_singular_plural(self.array_rows)
        row_cols_label = 'row' if self.rows else 'column'
        error_message = (
            f"The number of {row_cols_label} names you entered for the "
            f"DataFrame is not equal to the number of {row_cols_label}s of "
            f"the array. Please enter {self.array_rows} {row_cols_label} "
            f'name{plural}, {", ".join(map(str, self.row_names))}'
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class MaximumDimensionsError(ValueError):
    def __init__(
            self, 
            par_name: str = "Array values", 
            dimensions: list | tuple = [1, 1], 
            max_dimensions: list | tuple = [50, 50]
        ):
        self.par_name = split_get_first(par_name)
        self.dimensions = dimensions
        self.max_dimensions = max_dimensions
    def __str__(self):
        error_message = (
            f"Maximum number of rows exceeded. This app allows a maximum of "
            f"{self.max_dimensions[0]} rows and {self.max_dimensions[1]} "
            f"columns but got: {self.dimensions[0]} by {self.dimensions[1]} "
            f"array"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class ConvertError(ValueError):
    def __init__(
            self, 
            par_name: str, 
            data_structure: str = "the mathematical object", 
            user_input: str = "&nbsp;", 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.data_structure = data_structure
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):   
        error_message = (
            f"Unable to convert the value of left_{self.par_name}_right to "
            f"'{self.data_structure}' but got: {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class StructureConversionError(ValueError):
    def __init__(
            self, convert_from, 
            convert_to, 
            to_list_join: bool = False
        ):
        self.convert_from = convert_from
        self.convert_to = convert_to
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Conversion of '{self.convert_from}' to {self.convert_to} failed."
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class ComputationError(ValueError):
    def __init__(
            self, 
            par_name: str, 
            error_message:str, 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.error_message = error_message
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Execution of the '{self.par_name.lower()}' procedure failed: "
            f"{self.error_message}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class CreateObjectError(ValueError):
    def __init__(
            self, par_name: str, 
            object_name: str, 
            user_input: any,
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.object_name = object_name
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Unable to create '{self.object_name}' from the value entered "
            f"in the left_{self.par_name}_right input field but got "
            f"{self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class VariableNameError(ValueError):
    def __init__(self, par_name: str, user_input, to_list_join):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Unable to create a variable from the value entered in the "
            f"left_{self.par_name}_right input field but got "
            f"{self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class NotDependentVariableError(ValueError):
    def __init__(
            self, 
            par_name: str, 
            valid_variables: str, 
            user_input,
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.valid_variables = valid_variables
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"The value you have entered for left_{self.par_name}_right "
            f"input field is not a variable for the given expression. Valid "
            f"variable(s) include: '{self.valid_variables}' but got: "
            f"{self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class VariableNotFoundError(ValueError):
    def __init__(
            self, 
            par_name: str, 
            user_variable:str, 
            user_input,
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.user_variable = user_variable
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"The variable '{self.user_variable}' was not found in the "
            f"selected expression."
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class LimitsError(ValueError):
    def __init__(
            self, par_name: str, user_input=None, to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            "Unable to get the lower and upper limit from the entered input "
            f"but got {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    
    
class IndexBoundsError(IndexError):
    def __init__(
            self, par_name: str, user_input=None, to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        n = len(self.user_input)
        error_message = f"Expected index to be an integer between 0 and {n}"
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    
    
class InvalidInputError(ValueError):
    def __init__(
            self, 
            par_name: str = 'parameter', 
            user_input: str = 'Input', 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"'{self.user_input}' is an invalid value for {self.par_name}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message


class DuplicatesError(ValueError):
    def __init__(
            self, par_name: str = 'Field', user_input=None, to_list_join=False
        ):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"Expected left_{self.par_name}_right cannot have duplicates "
            f"(repeated variables) but got: {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message

   
class ValidationError(ValueError):
    def __init__(
            self, 
            validation_name: str = "your inputs", 
            to_list_join: bool = False
        ):
        self.validation_name = validation_name
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"An error occurred while validating {self.validation_name}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message

 
class ExceptError(ValueError):
    def __init__(
            self, 
            par_name: str = "API error: EE", 
            user_input: str = "Error", 
            show_location: bool = True, 
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.user_input = user_input
        self.show_location = show_location
        self.to_list_join = to_list_join
    def __str__(self):
        error_location = f' {ExceptLineNumber()}' if self.show_location else ''
        error_message = (
            f'{str(self.user_input)}.{error_location}'.replace('..', '.')
        )
        error_message = (
            error_message[2:] if error_message.startswith('. ') 
            else error_message
        )
        error_message = (
            f'{error_message}.' if not error_message.endswith('.') 
            else error_message
        )
        
        if self.to_list_join:
            error_message = join_error_list(
                self.par_name, error_message, self.to_list_join
            )
        
        return error_message


class ExceptionError(ValueError):
    def __init__(
            self, 
            par_name: str, 
            e: Exception,
            user_input: str,
            to_list_join: bool = False
        ):
        self.par_name = split_get_first(par_name)
        self.e = e
        self.user_input = user_input
        self.to_list_join = to_list_join
    def __str__(self):
        error_message = (
            f"{self.e} but got: {self.user_input}"
        )
        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    

class ExceptLineNumber(ValueError):

    def __init__(
            self, 
            par_name: str = "API error: LN", 
            e: str = '', 
            to_list_join: bool = True
        ):
        self.par_name = split_get_first(par_name)
        self.e = str(e)
        self.to_list_join = to_list_join
    def __str__(self):
        try:
            line_numbers = traceback.extract_tb(sys.exc_info()[2])
            if isinstance(line_numbers, str):
                line_numbers = [line_numbers]
            errors_location = []
            for line_number in line_numbers:
                file_name, line_number = str(line_number).split('line')
                file_name = file_name.split('\\')
                file_name = f'{file_name[-2]}/{file_name[-1]}'.replace('.py', '')
                error_message = (
                    (f'Line{line_number.replace(">]", "")} function of '
                    f'{file_name[:-2]} module.')
                )
                errors_location += [error_message]
            if '<<>>' in self.e:
                self.e = self.e.split('<<>>')[1]
            errors_location[0] = f'{self.e}<br /><br />> {errors_location[0]}'
            error_message = '<br /><br />> '.join(errors_location)
            print(f"\n===========\n{error_message}\n===========\n")
        except Exception:
            error_message = 'Unknown error location.'

        error_message = join_error_list(
            self.par_name, error_message, self.to_list_join
        )

        return error_message
    