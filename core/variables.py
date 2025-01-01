from keyword import iskeyword
from sympy import sympify
from stemlab.core.datetime import datetime_now_number


def symbolic_variable(var_name):
    """
    Convert a string to a symbolic variable name.
    
    Parameters
    ----------
    var_name : str
        Name of the variable.
    
    Returns
    -------
    str
        Symbolic variable name.
    """
    var_name = var_name.split(":=")[-1].strip() if ":=" in var_name else var_name
    var_name += "_" if var_name in ["lambda", "beta", "gamma"] else var_name
    try:
        sympify(var_name)
        return var_name
    except Exception: # not a valid variable name
        return "x_"


def variable_names(variable_name: str) -> str:
    """
    Convert a string to a valid variable name.

    Parameters
    ----------
    variable_name : str
        Name of the variable.

    Returns
    -------
    str
        Valid variable name.
    """
    try:
        variable_name = variable_name.replace(" ", "_")
        if not variable_name:
            variable_name = "ans"
        if iskeyword(variable_name):
            variable_name += "_"
        if not variable_name[0].isalpha():
            variable_name = f"v_{variable_name}"
        if not variable_name.isidentifier():
            variable_name = "ans"

        reserved_names = ["lambda", "beta", "gamma", "E", "I", "N", "O", "Q", "S"]
        math_functions = [
            "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh",
            "cos", "cosh", "exp", "log", "log2", "log10", "pi",
            "sin", "sinh", "sqrt", "tan", "tanh", "Heaviside"
        ]
        if variable_name in reserved_names + math_functions:
            variable_name += "_v"
        try:
            sympify(variable_name)
        except Exception:
            variable_name = "ans"
    except Exception:
        variable_name = "ans"

    return str(variable_name[:16])


def default_varnames():
    """
    Generate default variable names based on current datetime.

    Returns
    -------
    str
        Default variable name.
        
    """
    try:
        return f'v{datetime_now_number()[-6:]}'
    except Exception:
        return "ans"