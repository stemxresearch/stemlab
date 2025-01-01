import re
import urllib


from sympy import sympify, latex
from stemlab.core.symbolic import is_symexpr
from stemlab.core.arraylike import conv_list_to_string
from stemlab.core.decimals import (
    str_remove_trailing_zeros, str_replace_dot_zero
)
from stemlab.core.htmlstyles import results


def display_pretty(obj: any) -> None:
    """
    Display an object in a visually appealing format.

    This function utilizes IPython's display functionality to render
    objects in a visually pleasant way.

    Parameters
    ----------
    obj : any
        The object to be displayed.

    Returns
    -------
    None
    """
    from IPython.display import display

    display(obj)


def result_steps_latex(
    application: str, 
    input_dict: dict, 
    result_dict: dict, 
    steps_latex: str, 
    remove_decimal: bool = True
) -> str:
    """
    Combine the result string with LaTeX steps.

    Parameters
    ----------
    application : str
        The name of the application.
    input_dict : dict
        Dictionary containing input data.
    result_dict : dict
        Dictionary containing result data.
    steps_latex : str
        LaTeX formatted steps to be combined with the result.
    remove_decimal : bool, optional (default=True)
        Flag indicating whether to remove trailing decimals.

    Returns
    -------
    str
        Combined string of result and LaTeX steps.
    """
    result_str = str_remove_trailing_zeros(
        str_replace_dot_zero(
            results(application, input_dict, result_dict)
        ),
        remove_decimal=remove_decimal
    ) 
    return (
        f'<div style="margin-bottom:15px;">{result_str}</div>{steps_latex}'
    )


def success_message_nextjs(application: str = "Your") -> str:
    """
    Generate a success message for Next.js applications.

    Parameters
    ----------
    application : str, optional (default="Your")
        The name of the application.

    Returns
    -------
    str
        A success message indicating that the request was successfully 
        processed.
    """
    return (
        f'<span style="font-weight:600;">{application}</span> request was '
        'successfully processed.'
    )


def fail_message_nextjs(application: str = "Your") -> str:
    """
    Generate a failure message for Next.js applications.

    Parameters
    ----------
    application : str, optional (default="Your")
        The name of the application.

    Returns
    -------
    str
        A failure message indicating that the request was unsuccessful.
    """
    return (
        f'<span style="font-weight:600;">{application}</span> request was '
        '<strong>unsuccessful.</strong>'
    )


def page_url(app_url: str) -> str:
    """
    Generate a complete URL for a given application URL.

    Parameters
    ----------
    app_url : str
        The application URL.

    Returns
    -------
    str
        The complete URL for the application.
    """
    return urllib.parse.urljoin(
        "https://stemxresearch.com/", urllib.parse.quote(app_url)
    )



def query_to_tuple(user_input: list[str]) -> str:
    """
    Convert a list or NumPy array to a string representing a tuple.

    Parameters
    ----------
    user_input : list | numpy.ndarray
        The input list or NumPy array.

    Returns
    -------
    result : str
        String representing the input as a tuple.
    """
    try:
        user_input = str(tuple(user_input.tolist()))
    except Exception:
        pass

    replace_dict = {
        ".0,": ",",
        ".0)": ")",
        ".0%": "%",
        ".0+": "+",
        ".0-": "-",
        ".0*": "*",
        ".0/": "/",
        ".0&": "&",
        "//": "/",
        "  ": " ",
        ". ": ", ",
        "[": "(",
        "]": ")"
    }
    result = ''.join(replace_dict.get(char, char) for char in result)

    return result


def url_replace(url_path: str, no_brackets: bool = False) -> str:
    """
    Replace special characters and perform URL encoding for a given 
    URL path.

    Parameters
    ----------
    url_path : str
        The URL path to be processed.
    no_brackets : bool, optional (default=False)
        Whether to remove brackets from the URL.

    Returns
    -------
    result : str
        The processed URL path.
    """
    replace_dict = {
        "+": "%2B",
        "&&": "&",
        "True": "true",
        "False": "false",
        ".0,": ",",
        ".0)": ")",
        ".0%": "%",
        ".0+": "+",
        ".0-": "-",
        ".0*": "*",
        ".0/": "/",
        ".0&": "&",
        "//": "/"
    }

    # Apply replacements
    result = url_path
    for key, value in replace_dict.items():
        result = result.replace(key, value)

    # Optionally remove brackets
    if no_brackets:
        result = re.sub(r'[\[\]\(\)]', '', result)

    return result


def objectname_display(
    object_name: str, object_values: str | None = None
) -> str:
    """
    Display the object name in LaTeX format along with its values if 
    provided.

    Parameters
    ----------
    object_name : str
        The name of the object.
    object_values : str, optional (default=None)
        The values associated with the object.

    Returns
    -------
    object_name : str
        The LaTeX-formatted string representing the object name and 
        values (if provided).
    """
    if isinstance(object_values, str):
        try:
            object_values = sympify(object_values)
        except Exception:  # e.g. graphs
            pass
    
    try:  # if this works, then it is  a function
        expr_symbols = object_values.free_symbols
        object_symbols = conv_list_to_string(expr_symbols, delimiter=", ")
        if expr_symbols:  # if not an empty set (i.e. symbolic expression)
            if is_symexpr(object_values):
                object_name = f"{latex(sympify(object_name))}({object_symbols})"
            else:
                object_name = f"{latex(sympify(object_name))}"
        else:
            object_name = f"{latex(sympify(object_name))}"
    except Exception:
        try:
            object_name = f"{latex(sympify(object_name))}"
        except Exception:
            pass
    
    return object_name