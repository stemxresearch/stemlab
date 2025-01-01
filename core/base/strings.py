import re
import string
import random
from typing import Literal

from sympy import Expr
from pandas import DataFrame, Series
from stemlab.core.variables import variable_names
from stemlab.core.htmlstyles import result_name_html
from stemlab.core.datatypes import NumpyArray


def str_info_messages(maxit: int, tolerance: float, k: int) -> str:
    """
    Generate an informational message based on convergence.

    Parameters
    ----------
    maxit : int
        Maximum number of iterations allowed.
    tolerance : float
        The acceptable error tolerance.
    k : int
        The current iteration count.

    Returns
    -------
    msg : str
        A string with convergence information.

    Notes
    -----
    The message informs about whether the tolerance was achieved 
    before reaching the maximum number of iterations or whether
    the maximum number of iterations was reached before achieving 
    the tolerance.
    """
    if maxit == k:
        msg = (
            f"The maximum number of iterations ({int(maxit)}) was reached "
            f"before achieving the tolerance ({tolerance})."
        )
    else:# k < maxit:
        msg = (
            f"The tolerance ({tolerance}) was achieved before "
            f"reaching the maximum number of iterations ({int(maxit)})."
        )
    
    return msg


def str_get_info_messages(InfoMessages: list[str] | None = None) -> str:
    """
    Format informational messages into HTML for display.

    Parameters
    ----------
    InfoMessages : list_like, optional (default=None)
        List-like object containing informational messages.

    Returns
    -------
    message : str
        HTML-formatted messages enclosed in a styled <div> container.

    Raises
    ------
    ValueError
        If the provided InfoMessages is not a list, Series, or any 
        iterable containing strings.
    """
    try:
        InfoMessages = Series(
            data=InfoMessages, dtype="object"
        ).drop_duplicates().values.tolist()
    except ValueError as e:
        raise e

    message = " ".join([f'<p>{message}</p>' for message in InfoMessages])
    message = (
        f"""
        <div style="background:#F1F1F1;border-radius:3px;margin-top:10px;
        margin-right:-5px;padding:1px 8px;">{message}</div>
        """
    )

    return message


def str_print(text: str = "This is OK") -> None:
    """
    Print a string with surrounding markers for visual separation.

    This function prints the specified string surrounded by markers for
    visual separation. By default, it prints the specified text 
    surrounded by equal signs.

    Parameters
    ----------
    text : str, optional (default="This is OK")
        The text to be printed.

    Returns
    -------
    None
    
    Examples
    --------
    stm.str_print()
    
    stm.str_print(12345)
    """
    print(f"\n===========\n{text}\n===========\n")
    
    
def str_separate_alphanumeric(x: str, si_unit: str) -> float | list[float, str]:
    
    from stemlab.core.validators.validate import ValidateArgs
    
    si_unit = ValidateArgs.check_string(par_name='si_unit', user_input=si_unit)
    match = re.match(r'(\d+)(.*)', x.lower())
    if match is None:
        raise ValueError(
            f"Expected 'x' to be numeric or alphanumeric but got: {x}"
        )
    result = list(match.groups())
    result[0] = float(result[0])
    if not result[1]:
        result[1] = si_unit
    
    return result
    

def str_remove_special_symbols(
    strng: str | list[str]
) -> str | list[str]:
    """
    Remove special characters from a string or a list/tuple of strings.

    Parameters
    ----------
    strng : {str, list_like}
        The input string or list/tuple of strings from which special 
        characters will be removed.

    Returns
    -------
    result : {str, list_like}
        - If the input is a string, the function returns the modified 
        string with special characters removed.
        - If the input is a list or tuple of strings, the function 
        returns a list with each string modified to remove special 
        characters.

    Examples
    --------
    >>> import stemlab as stm

    >>> stm.str_remove_special_symbols("Hello! How are you?")
    'Hello How are you'

    >>> stm.str_remove_special_symbols(["Hello!", "How are you?"])
    ['Hello', 'How are you']

    >>> stm.str_remove_special_symbols(("Hello!", "How are you?"))
    ['Hello', 'How are you']
    """
    pattern = r'[^A-Za-z0-9- ]' # remove special charactes
    if isinstance(strng, str):
        result = re.sub(pattern, '', strng)
    else:
        result = [re.sub(pattern, '', item) for item in strng]
    
    return result


def str_change_case(
    strng: str | list[str], 
    case: Literal['lower', 'upper', 'title', 'capitalize'] = 'lower', 
    replace_space_with=None
) -> str | list[str]:
    """
    Convert a string or a list of strings to lower case and replace 
    spaces with dashes.
    
    Parameters
    ----------
    strng : {str, list_like}
        The input string or list of strings to be converted to lower 
        case.
    case : str, {'lower', 'upper', 'title', 'capitalize'}, optional (default='lower')
        The case to change to
    replace_space : bool, optional (default=True)
        If `True`, a hiphen (-) will be used to replace spaces.

    Returns
    -------
    str_change_case_str : {str, list}
        If the input is a string, the function returns the modified 
        string.
        If the input is a list of strings, the function returns a list 
        with each string modified.

    Examples
    --------
    >>> import stemlab as stm

    >>> stm.str_change_case("Hello World")
    'hello-world'

    >>> stm.str_change_case(["Hello", "World"])
    ['HELLO', 'WORLD']
    """
    from stemlab.core.arraylike import conv_list_to_dict
    from stemlab.core.validators.validate import ValidateArgs
    
    valid_cases = ['lower', 'upper', 'title', 'capitalize']
    case = ValidateArgs.check_member(
        par_name='case', valid_items=valid_cases, user_input=case
    )
    replace_space_with = ' ' if replace_space_with is None else replace_space_with
    
    cases_dict = conv_list_to_dict(keys_list=valid_cases, values_list=valid_cases)
    case_method = cases_dict[case]
    if isinstance(strng, str):
        transformed_case = strng.__getattribute__(case_method)().replace(
            ' ', replace_space_with
        )
    else:
        try:
            transformed_case = [
                str(item).__getattribute__(case_method)().replace(
                    ' ', replace_space_with
                ) for item in strng
            ]
        except Exception as e:
            raise e
    
    return transformed_case


def str_normal_case(
    strng: str | list[str], 
    replace_space_with = None,
    remove_special_symbols: bool = True
) -> str | list[str]:
    """
    Convert strings or lists of strings to normal case 
    (capitalize the first letter of each word) and optionally replace 
    spaces with dashes or vice versa. Additionally, remove all special 
    symbols.

    Parameters
    ----------
    strng : {str, list}
        The input string or list of strings to be converted to normal 
        case.
    replace_space : bool, optional (default=True)
        If `True`, spaces will be replaced with dashes.

    Returns
    -------
    str_normal_case : {str, list-like}
        - If the input is a string, the function returns the modified 
        string.
        - If the input is a list of strings, the function returns a list 
        with each string modified.

    Examples
    --------
    >>> import stemlab as stm

    >>> stm.str_normal_case("hello world? Looks great!",
    ... remove_special_symbols=True)
    'Hello world looks great'

    >>> stm.str_normal_case(["hello world? Looks great!", "STEM RESEARCH"],
    ... remove_special_symbols=False)
    ['Hello world? looks great!', 'Stem research']
    """
    from stemlab.core.validators.validate import ValidateArgs
    
    remove_special_symbols = ValidateArgs.check_boolean(
        user_input=remove_special_symbols, default=True
    )
    str_normal_case =str_change_case(
        strng=strng, case='capitalize', replace_space_with=replace_space_with
    )
    if remove_special_symbols:
        str_normal_case = str_remove_special_symbols(str_normal_case)
    
    return str_normal_case


def str_replace_space(strng: str, delimiter: str = ' ') -> str:
    """
    Remove spaces from a string and replace `()` and `{}` brackets 
    with square brackets `[]`.

    Parameters
    ----------
    strng : str
        The input string.
    delimiter : str, optional (default=' ')
        The delimiter to use for joining the string after spaces are 
        removed.

    Returns
    -------
    transformed_string : str
        The modified string with spaces removed and `()` & `{}` 
        brackets replaced with square brackets `[]`.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.str_replace_space("hello world", delimiter="-")
    'hello-world'
    >>> stm.str_replace_space("(1, 2, 3)")
    '[1, 2, 3]'
    """
    try:
        strng = strng.replace("{", "[")\
            .replace("}", "]")\
            .replace("(", "[")\
            .replace(")", "]")
        transformed_string = delimiter.join(strng.split())
    except Exception as e:
        raise e
    
    return transformed_string


def str_capitalize_nohiphen(strng: str) -> str:
    """
    Capitalize the first letter of a string or a list of strings and 
    remove any hyphens present.

    Parameters
    ----------
    strng : str
        The input string.

    Returns
    -------
    transformed_string : str
        The modified string with the first letter capitalized and 
        hyphens removed.
        
    Examples
    --------
    >>> import stemlab as stm
    >>> stm.str_capitalize_nohiphen("Hello-World")
    'Helloworld'

    >>> stm.str_capitalize_nohiphen("good-bye")
    'Goodbye'

    >>> stm.str_capitalize_nohiphen("python-Programming")
    'Pythonprogramming'

    >>> stm.str_capitalize_nohiphen("sample-Test")
    'Sampletest'
    """
    transformed_string = str_change_case(
        strng=strng, case='capitalize', replace_space_with=None
    ).replace('-', '')
    
    return transformed_string


def str_args_to_dict(string_args: str) -> dict:
    """
    Convert a string to a dictionary.

    Parameters
    ----------
    string_args : str
        The string to be converted to a dictionary.

    Returns
    -------
    dict
        A dictionary generated from the input string.

    Raises
    ------
    TypeError
        If `string_args`, or if the function fails to create a dictionary 
        from the given input value.
        
    Examples
    --------
    >>> import stemlab as stm
    >>> string_args = 'A = fname(x=3.8979, y=3 / 5, k=cos(pi / 4), z=[4, 5, 6], p=[[5, 6], [8, 3], [0, 3]], method=rk4)'
    >>> stm.str_args_to_dict(string_args)
    ('a',
    'fname',
    {'x': '3.8979',
    'y': '3/5',
    'k': 'cos(pi/4)',
    'z': '[4,5,6]',
    'p': '[[5, 6],[8, 3],[0, 3]]',
    'method': 'rk4'})
    """
    from stemlab.core.validators.validate import ValidateArgs
    
    string_args = ValidateArgs.check_string(
        par_name='string_args', to_lower=False, user_input=string_args
    )
    if string_args.endswith(')'):
        string_args = string_args[:-1]
        
    obj_func_name = string_args.split('(', 1)[0].strip()
    
    if '=' in obj_func_name:
        has_obj_name = True
        object_name = obj_func_name.split('=', 1)[0].strip()
    else:
        has_obj_name = False
        object_name = 'ans'
    
    if '=' in obj_func_name:
        func_name = obj_func_name.split('=', 1)[1].strip()
    else:
        func_name = obj_func_name
    
    if has_obj_name:
        string_args = string_args.split('=', 1)[1].strip()
    pattern = re.compile(r'(\w+)\s*=\s*(.*?)(?=,\s*\w+\s*=|$)')
    matches = pattern.findall(string_args)
    
    try:
        args_dict = {}
        for key, value in matches:
            args_dict[key] = value.strip()
    except Exception:
        raise TypeError(
            "Could not create a dictionary from the given input value"
        )
    
    result = object_name, func_name, args_dict
    
    return result


def more_options_inputs(
    result_name: str, 
    description: bool,
    decimal_points: int, 
    query_string: dict, 
    force: bool = False
) -> tuple[str, bool, int, dict]:
    """
    Get additional fields for form input.

    Parameters
    ----------
    result_name : str
        The name of the result.
    include_description : bool
        Whether to include a description in the form input.
    decimal_points : int
        The number of decimal points to display.
    query_string : dict
        Additional parameters to include in the query string.
    force_decimal_points : bool, optional (default=False)
        If `True`, forces the decimal points to be a specific value
        if parsing `decimal_points` fails.

    Returns
    -------
    tuple : {str, bool, int, dict}
        A tuple containing the result name, a boolean indicating 
        whether description is included, the number of decimal points, 
        and the updated query string.
    """
    from stemlab.core.validators.validate import ValidateArgs

    result_name = variable_names(result_name_html(result_name))
    try:
        decimal_points = int(decimal_points)
    except Exception:
        decimal_points = 8 if force else -1
    description = ValidateArgs.check_boolean(user_input=description, default=True)
    query_string.update(
        {"resultName": result_name, "decimalPoints": decimal_points}
    )

    return result_name, description, decimal_points, query_string


def str_random_string(
    n: int = 12, 
    digits: bool = True, 
    include_lower: bool = False, 
    symbols: bool = False
) -> str:
    """
    Generate a random string.

    Parameters
    ----------
    n : int, optional (default=12)
        The length of the random string.
    digits : bool, optional (default=True)
        Whether to include digits (0-9) in the random string.
    include_lower : bool, optional (default=False)
        Whether to include lowercase letters (a-z) in the random 
        string.
    symbols : bool, optional (default=False)
        Whether to include symbols in the random string.

    Returns
    -------
    rand_string
        A random string generated based on the specified criteria.

    Examples
    --------
    >>> import stemlab as stm
    You will get different results because of the randomness
    
    >>> stm.str_random_string()
    'N5YD9OIGPI2H'
    
    >>> stm.str_random_string(n=8, include_lower=True)
    'ThbYgq2B'
    
    >>> stm.str_random_string(n=12, include_lower=False)
    'R4956OWDWJZS'
    """
    chars = string.ascii_uppercase
    if digits:
        chars += string.digits
    if include_lower:
        chars += string.ascii_lowercase
    if symbols:
        chars += string.punctuation
    
    rand_string = ''.join(random.choices(chars, k=n))
    
    return rand_string


def str_singular_plural(
    n: int, singular_form: str = '', plural_form: str = 's'
) -> str:
    """
    Return plural or singular form of a word based on the count.

    Parameters
    ----------
    n : int
        The count used to determine whether to return the singular or 
        plural form.
    singular_form : str, optional (default='')
        The singular form of the word. Defaults to an empty string.
    plural_form : str, optional (default='s')
        The plural form of the word.

    Returns
    -------
    str
        The singular or plural form of the word based on the count.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.str_singular_plural(1, 'is', 'are')
    'is'
    >>> stm.str_singular_plural(3, 'book', 'books')
    'books'
    """
    from stemlab.core.validators.validate import ValidateArgs
    
    n = ValidateArgs.check_numeric(par_name='n', is_integer=True, user_input=n)
    plural_form = ValidateArgs.check_string(
        par_name='plural_form', user_input=plural_form
    )
    singular_form = ValidateArgs.check_string(
        par_name='singular_form', user_input=singular_form
    )
    
    return (plural_form if n > 1 else singular_form)


def str_strip_all(strng: str) -> str:
    """
    Remove all white spaces (leading, trailing, and internal).

    Parameters
    ----------
    strng : str
        A string containing the white spaces to be removed.

    Returns
    -------
    stripped_string : str
        A string with all white spaces removed.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.str_strip_all("  Hello  World  ")
    'Hello World'
    >>> stm.str_strip_all("  Hello  World  with  spaces  ")
    'Hello World with spaces'
    >>> stm.str_strip_all("Hello,\\tworld!\\n")
    'Hello, world!'
    >>> stm.str_strip_all("Hello,\\u00A0world!")
    'Hello, world!'
    """
    from stemlab.core.validators.validate import ValidateArgs
    
    strng = ValidateArgs.check_string(
        par_name='strng', user_input=strng, to_lower=False
    )
    stripped_string = ' '.join(strng.split())

    return stripped_string


def str_replace_characters(
    df: DataFrame,
    chars_to_replace: str,
    replace_with: str = '',
    columns: str | list = 'all',
    regex: bool = True
) -> DataFrame:
    """
    Replace specified characters in a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The DataFrame that contains the values to be replaced.
    chars_to_replace : str
        The characters in the DataFrame that need to be replaced.
    replace_with : str, optional
        The string that should replace the characters specified in 
        ``replace_with``. Defaults to an empty string.
    columns : str, list_like, optional (default='all')
        Column(s) that contain the characters to be replaced. 
        If 'all', applies to all columns.
    regex : bool, optional (default=True)
        Whether to interpret the replacement strings as regular 
        expressions.

    Returns
    -------
    DataFrame
        A DataFrame with the characters replaced.

    Examples
    --------
    >>> import pandas as pd
    >>> import stemlab as stm
    >>> df = {
    ...     'Fruit': ['Apple pie', 'Banana bread', 'Cherry tart','Date cake'],
    ...     'Color': ['Red apple', 'Yellow banana', 'Red cherry', 'Brown date'],
    ...     'Type': ['Sweet apple', 'Ripe banana', 'Tangy cherry', 'Sweet date']}
    >>> df = pd.DataFrame(df)
    >>> stm.str_replace_characters(df, chars_to_replace='e',
    ... replace_with='E')
                   Fruit              Color             Type
    0          ApplE piE          REd applE      SwEEt applE
    1       Banana brEad      YEllow banana      RipE banana
    2        ChErry tart         REd chErry     Tangy chErry
    3          DatE cakE         Brown datE       SwEEt datE
    """
    from stemlab.core.arraylike import is_iterable
    
    chars_to_replace = '[{' + re.escape(chars_to_replace) + '}]'
    if isinstance(columns, str) and columns == 'all':
        columns = df.columns
    elif is_iterable(columns):
        columns = list(columns)
    else:
        raise ValueError("Columns must be 'all' or a list-like object.")
    
    dframe_transformed = df[columns].replace(
        to_replace=chars_to_replace, value=replace_with, regex=regex
    )
        
    return dframe_transformed


def str_plus_minus(x: int | float| Expr) -> Literal['-', '+']:
    """
    Returns the sign of a numerical value x.

    Parameters
    ----------
    x : {int, float, Expr}
        The numerical value or sympy expression for which the sign 
        needs to be determined.

    Returns
    -------
    sgn : str
        A string representing the sign of the value. Returns '+' if 
        the value is positive or zero, and '-' if the value is 
        negative.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.str_plus_minus(5)
    '+'
    >>> stm.str_plus_minus(-3.14)
    '-'
    >>> stm.str_plus_minus(0)
    '+'
    """
    from stemlab.core.datatypes import is_negative
    
    # Check if the value is negative or zero
    sgn = '-' if is_negative(x) else '+'
    
    return sgn


def str_partial_characters(list_string: list) -> NumpyArray:
    """
    Return partial substring for each of the words in the specified 
    iterable.

    Parameters
    ----------
    list_string: {str, list}
        A string or list of strings containing the word/text whose
        characters are to be returned as partial substrings.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.str_partial_characters(list_string='subtract')
    array(['s', 'su', 'sub', 'subt', 'subtr', 'subtra', 'subtrac', 'subtract'],
        dtype=object)

    >>> stm.str_partial_characters(list_string=['add', 'subtract'])
    array(['a', 'ad', 'add', 's', 'su', 'sub', 'subt', 'subtr', 'subtra',
       'subtrac', 'subtract'], dtype=object)
    
    Returns
    -------
    result : numpy.ndarray
        A 1D Numpy array.
    """
    from stemlab.core.arraylike import conv_to_arraylike

    list_string = conv_to_arraylike(
        array_values=list_string, includes_str=True, par_name='list_string'
    )
    result = []
    for word in list_string:
        result.extend([word[:i + 1] for i in range(len(word))])
    result = Series(result).drop_duplicates(keep = False).values
            
    return result


def str_remove_trailing_zeros(
    matrix_string: str, remove_decimal: bool = True
) -> str:
    """
    Remove trailing zeros and decimals in numerical strings.

    Parameters
    ----------
    input_string : str
        The string containing numerical values.
    remove_decimal : bool, optional
        Whether to remove the decimal point as well, by default True.

    Returns
    -------
    result : str
        The string with trailing zeros and decimals removed.

    Examples
    --------
    >>> import stemlab as stm

    >>> stm.str_remove_trailing_zeros('10.00 + 20.000')
    '10 + 20'

    >>> stm.str_remove_trailing_zeros('3.000 * 5.0 / 2.0')
    '3 * 5 / 2'

    >>> stm.str_remove_trailing_zeros('0.000 + 0.0 - 0.00')
    '0 + 0 - 0'

    >>> stm.str_remove_trailing_zeros('1.000^2 + 2.0^2')
    '1^2 + 2^2'

    >>> stm.str_remove_trailing_zeros('4.00 + 6.000 / 2.0 * 3.0000')
    '4 + 6 / 2 * 3'
    """
    try:
        matrix_string = str(matrix_string)
        # Match floating-point numbers with trailing zeros
        regex = r'\b(\d+\.\d*?)0+\b'
        if remove_decimal:
            result = re.sub(
                # Replace the matched numbers with the ones without  
                # trailing zeros and decimal point if it becomes an 
                # integer
                regex, lambda match: match.group(1).rstrip('.') 
                if '.' in match.group(1) else match.group(1), matrix_string
            )
        else:
            # Replace the matched numbers with the ones without 
            # trailing zeros
            result = re.sub(regex, r'\1', matrix_string)
    except Exception:
        result = matrix_string
    
    return result


def str_replace_dot_zero(result: str) -> str:
    """
    Replace trailing zeros with an empty string in the result string.

    Parameters
    ----------
    result : str
        The string to process.

    Returns
    -------
    result : str
        The processed string.

    Examples
    --------
    >>> import stemlab as stm

    >>> stm.str_replace_dot_zero('10.0 + 20.0')
    '10 + 20'

    >>> stm.str_replace_dot_zero('10.0 + 20.000')
    '10 + 20'

    >>> stm.str_replace_dot_zero('3.000 * 5.000 / 2.0')
    '3 * 5 / 2'

    >>> stm.str_replace_dot_zero('0.000 + 0.0 - 0.000')
    '0 + 0 - 0'

    >>> stm.str_replace_dot_zero('1.000^2 + 2.0^2')
    '1^2 + 2^2'

    >>> stm.str_replace_dot_zero('4.00 + 6.000 / 2.0 * 3.0000')
    '4 + 6 / 2 * 3'
    """
    # Define patterns to replace
    patterns = [
        ".0 ", ".0}", ".0^", ".0\\", ".0,", ".0*", ".0)", ".0]", ".0+", ".0-", 
        ".0/", ".000000000000}", ".000000000000)", ".000000000000]", ".0"
    ]

    # Remove trailing zeros and decimals
    try:
        result = str_remove_trailing_zeros(
            str(result), remove_decimal=True
        )
    except Exception:
        pass

    for pattern in patterns:
        result = result.replace(pattern, "")

    return result