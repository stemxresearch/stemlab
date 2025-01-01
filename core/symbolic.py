import sys
import re
from typing import Literal, Callable

from sympy import (
    sympify, collect, lambdify, Matrix, flatten, Integer, Float, Symbol,
    Expr, Add, simplify, Number
)
from numpy import ndarray, prod, zeros, append, array
from pandas import DataFrame

from stemlab.core.base.strings import str_singular_plural
from stemlab.core.arraylike import is_iterable, conv_to_arraylike
from stemlab.core.datatypes import is_function
from stemlab.core.validators.errors import (
    SympifyError, SymbolicExprError, UnivariatePolyError, IterableError
)
from stemlab.core.validators.validate import ValidateArgs


def is_symexpr(fexpr: str | Expr) -> bool:
    """
    Check if expression is symbolic (i.e. contains unknown variables).

    Parameters
    ----------
    fexpr : {str, sympy.Expr}
        An object representing the value to be tested.

    Returns
    -------
    result: bool
        True/False

    Examples
    --------
    >>> import stemlab as stm

    >>> stm.is_symexpr('pi/4')
    False

    >>> stm.is_symexpr('pi/x')
    True

    >>> g = 'x^2 + x * y - 5'
    >>> stm.is_symexpr(g)
    True
    """
    try:
        # use of str() is required in Sympy >= 1.6
        result = len(sympify(str(fexpr)).free_symbols) > 0
    except Exception:
        result = False

    return result


def sym_poly_constant(
    fexpr: str | Expr, 
    simp_expr: bool = True,
    to_float: bool = False, 
    decimal_points: int = 4
) -> int | float | Integer | Float:
    """
    Extracts the constant term in a specified expression.

    Parameters
    ---------
    fexpr : {str, sympy.Expr}
        An expression containing the symbolic expression.
    simp_expr : bool, optional (default=True)
        If `True`, the expression will be simplified first.
    to_float : bool, optional (default=False)
        If `True`, the constant will be converted to float (decimal).
    decimal_points : int, optional (default=4)
        Number of decimal points or significant figures for symbolic
        expressions. 

    Returns
    -------
    constant : {int, float, Integer, Float, None}
        Returns a constant in a specified expression if it exists 
        and `None` otherwise.

    Notes
    -----
    If `simp_expr` is True, the expression will be simplified first. 
    For example sin(x ** 2) + cos(x ** 2) will reduce to 1, 
    sin(x ** 2) - cos(x ** 2) will reduce to cos(2*x) and so forth.

    Examples
    --------
    >>> import stemlab as stm

    >>> stm.sym_poly_constant('2*x^2 + 2/5')
    2/5

    >>> stm.sym_poly_constant('2*x^2 + 2/5', to_float = True)
    0.4

    >>> stm.sym_poly_constant('2*x^2')
    0

    >>> stm.sym_poly_constant('cos(x) ** 2 + sin(x) ** 2 + 3')
    4

    >>> stm.sym_poly_constant('cos(x) ** 2 + sin(x) ** 2 + 3',
    ... simp_expr=False)
    3
    """ 
    fexpr = sym_sympify(expr_array=fexpr, par_name='fexpr')
    terms = Add.make_args(simplify(fexpr) if simp_expr else fexpr)
    constant_term = [term for term in terms if term.is_constant()]
    constant_term = constant_term[0] if constant_term else 0
    if to_float:
        constant_term = sympify(round(float(constant_term), decimal_points))
    
    return constant_term


def sym_simplify_expr(
    fexpr: str | Expr, 
    simplify_method: Literal[
        'cancel', 'collect', 'expand', 'factor', 'simplify', 
        'together', None
    ] = 'factor', 
    collect_term: str | Expr | None = None
) -> Expr:
    """
    Simplify a symbolic expression using a specified simplification 
    method.

    Parameters
    ----------
    expression : {str, sympy.Expr}
        The expression that is to be simplified.
    simplify_method : str, optional (default='factor')
        The simplification method to be applied.
        Valid options: {'cancel', 'collect', 'expand', 'factor', 'simplify', 'together'}
    collect_term : {str, sympy.Expr}, optional (default=None)
        An expression or list of expressions that should be used to 
        group the input expression. Required when 
        `simplify_method='collect'`.

    Returns
    -------
    fexpr : sympy.Expr
        A simplified form of the entered symbolic expression.

    Raises
    ------
    NotMemberError
        If an invalid `simplify_method` is provided.
    EmptyFieldError
        If `collect_term` is not provided when `simplify_method='collect'`.
    SymbolicExprError
        If `collect_term` is not a valid symbolic expression.
    
    Examples
    --------
    >>> import stemlab as stm
    
    >>> f = '(x + x^2)/(x*sin(y)^2 + x*cos(y)^2)'
    >>> stm.sym_simplify_expr(f, simplify_method='simplify')
    x + 1
    
    >>> g = 'x^2 + y*x^2 + x*y + y + a*y'
    >>> stm.sym_simplify_expr(g, simplify_method='collect',
    ... collect_term=['x', 'y'])
    x ** 2*(y + 1) + x*y + y*(a + 1)
    """
    # expression
    # ----------
    if not is_symexpr(fexpr):
        return fexpr # if not symbolic, just return the result as given
    else:
        fexpr = sym_sympify(expr_array=fexpr, par_name='fexpr')
    
    # simplify_method
    # ---------------
    simplify_method = 'none' if simplify_method is None else simplify_method
    valid_methods = [
        'none', 'cancel', 'collect', 'expand', 'factor', 'simplify', 'together'
    ]
    simplify_method = ValidateArgs.check_member(
        par_name='simplify_method', 
        valid_items=valid_methods, 
        user_input=simplify_method, 
        default=None
    )
    simplify_method = None if simplify_method == 'none' else simplify_method
    
    # collect_term
    # ------------
    if simplify_method == 'collect':
        collect_term = sym_sympify(
            expr_array=collect_term,
            is_expr=False if isinstance(collect_term, (tuple, list)) else True,
            par_name='collect_term'
        )
        # a loop is used just in case there are multiple variable 
        # specified e.g. [x, y]
        if isinstance(collect_term, (list, tuple)):
            for term in collect_term:
                if not is_symexpr(term):
                    raise SymbolicExprError(
                        par_name=f'collect_term: {term}', user_input=term
                    )
        else:
            if not is_symexpr(collect_term):
                raise SymbolicExprError(
                    par_name='collect_term', user_input=collect_term
                )

    # begin
    # -----
    methods_dict = {
        None: lambda expr: expr,
        'collect': lambda expr: collect(expr, collect_term),
        'simplify': lambda expr: expr.simplify(),
        'together': lambda expr: expr.together(),
        'expand': lambda expr: expr.expand(),
        'factor': lambda expr: expr.factor(),
        'cancel': lambda expr: expr.cancel()
    }
    fexpr = methods_dict[simplify_method](fexpr)
        
    return fexpr


def sym_poly_terms(
    fexpr: str | Expr, simp_expr: bool = False
) -> tuple[tuple[Expr], int, bool]:
    """
    Returns the term(s) of a symbolic expression, including the 
    constant, if any.

    Parameters
    ----------
    fexpr : {str, sympy.Expr}
        A univariate polynomial.
    simp_expr : bool, optional (default=True)
        If `True`, the expression will be simplified first.

    Returns
    -------
    terms_list : list
        The term(s) of the symbolic expression.
    terms_count : int
        The number of terms.
    constant_bool : bool
    result: tuple
        Boolean value indicating whether or not the expression has 
        a constant.

    Notes
    -----
    If simp_expr is True, the expression will be simplified first. 
    For example sin(x ** 2) + cos(x ** 2) will reduce to 1, 
    sin(x ** 2) - cos(x ** 2) will reduce to cos(2*x) and so forth.

    Examples
    --------
    >>> import stemlab as stm

    >>> stm.sym_poly_terms('4*x^2 - 5*x + 1')
    ((1, -5*x, 4*x ** 2), 3, True)

    >>> stm.sym_poly_terms('4*x^2 - 5*x')
    ((-5*x, 4*x ** 2), 2, False)

    >>> stm.sym_poly_terms('sin(x) ** 2 + cos(x) ** 2')
    ((cos(x) ** 2, sin(x) ** 2), 2, False)

    >>> stm.sym_poly_terms('sin(x) ** 2 + cos(x) ** 2', simp_expr=True)
    ((1,), 1, True)
    """
    fexpr = sym_sympify(expr_array=fexpr, par_name='fexpr')
    terms = Add.make_args(simplify(fexpr) if simp_expr else fexpr)
    constant_term_found = any(term.is_constant() for term in terms)
    
    return terms, len(terms), constant_term_found


def sym_remove_zeros(
    fexpr: str | Expr, threshold: float = 1e-16
) -> Expr:
    """
    Remove terms with coefficients that are close to zero from an 
    nth order polynomial.

    Parameters
    ----------
    fexpr : {str, sympy.Expr}
        A symbolic expression with the term(s) to be truncated.

    threshold : float
        Smallest value for which terms should be removed. Terms 
        below this value will be removed from the polynomial.

    Examples
    --------
    >>> import stemlab as stm

    >>> g = 'x^3/2000 + x^2/50000 + 5*x + 8'
    >>> stm.sym_remove_zeros(g, threshold=1e-6) 
    x ** 3/2000 + x ** 2/50000 + 5*x + 8

    >>> stm.sym_remove_zeros(g, threshold=1e-4)
    x ** 3/2000 + 5*x + 8
    
    >>> h = '5*x + 8'
    >>> stm.sym_remove_zeros(h, threshold=1e-3)
    5*x + 8

    Returns
    -------
    fexpr : sympy.Expr
        A symbolic expression with terms whose coefficients are 
        greater than `threshold`, or a constant value.
    """
    try:
        fexpr = sym_sympify(expr_array=fexpr, par_name='fexpr')
        small_numbers = set(
            [term for term in fexpr.atoms(Number) if abs(term) < threshold]
        )
        if small_numbers:
            small_numbers_map_zero = {
                small_number: 0 for small_number in small_numbers
            }
            # substitute the small numbers with zero
            fexpr = fexpr.subs(small_numbers_map_zero)
    except:
        pass

    return fexpr

def sym_sympify(
    expr_array: str | list, 
    is_expr: bool = True,
    par_name: str = 'input'
) -> Expr | Matrix:
    """
    Convert a string or list of strings to a symbolic expression or 
    matrix.

    Parameters
    ----------
    expr_input : {str, array_like}
        The expression(s) to be converted to symbolic expression(s) or 
        matrix.
    is_expr : bool, optional (default=True)
        If `True`, the input will be assumed to be a symbolic expression.
    par_name : str, optional (default='input')
        Name to use in error messages to describe the parameter being 
        checked.

    Returns
    -------
    expr_array : {sympy.Expr, sym.Matrix}
        The converted symbolic expression or matrix.

    Raises
    ------
    SympifyError
        If the input cannot be converted to a symbolic expression or matrix.
    SymbolicExprError
        If the input is not a valid symbolic expression.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.sym_sympify('x + y')
    x + y
    
    >>> stm.sym_sympify(['x + y', '2*x - y'], is_expr=False)
    Matrix([
    [x + y],
    [2*x - y]])
    """
    if is_function(expr_array):
        raise ValueError(f"'expr_array' cannot be a callable function")
    try:
        expr_array = sympify(expr_array)
    except Exception:
        raise SympifyError(par_name=par_name, user_input=expr_array)
    
    is_expr = ValidateArgs.check_boolean(user_input=is_expr, default=True)
    expr_array = expr_array if is_expr else Matrix(expr_array)
    
    if is_expr and not is_symexpr(expr_array):
        raise SymbolicExprError(par_name=par_name, user_input=expr_array)
    
    return expr_array

        
def sym_lambdify_expr(
    fexpr: str | Callable | list[str], 
    is_univariate: bool = False, 
    variables: list[str] | None = None, 
    par_name: str = 'fexpr'
) -> Callable:
    """
    Converts a symbolic equation into a NumPy-compatible function.

    Parameters
    ----------
    fexpr : {str, callable, list_like}
        The symbolic equation (or a list of symbolic equations) to be 
        converted into a Numpy-compatible function.
    is_univariate : bool, optional (default=False)
        Whether or not the equation is univariate.
    variables : array_like, optional (default=None)
        List of variable names in the equation.
    par_name : str, optional (default='fexpr')
        Name to use in error messages to describe the parameter being 
        checked.

    Returns
    -------
    callable
        A NumPy-compatible function representing the input equation.

    Raises
    ------
    SympifyError
        If equation cannot be converted to a symbolic expression.
    UnivariatePolyError
        If equation is not univariate as specified.
    SymbolicExprError
        If equation is not a symbolic expression.
    
    Examples:
    --------
    >>> import stemlab as stm
    >>> f = stm.sym_lambdify_expr('x ** 2 + 2*x + 1')
    >>> f(2)
    9
    >>> equations = ['sin(x1) + x2 ** 2 + log(x3) - 7',
    ... '3*x1 + 2 ** x2 - x3 ** 3 + 1', 'x1 + x2 + x3 - 5']
    >>> f = stm.sym_lambdify_expr(equations)
    >>> f([4, 5/7, 3])
    array([ 8.80464777, 25.63556851,  2.71428571])
    """
    if is_function(fexpr):
        return fexpr
    elif isinstance(fexpr, (list, tuple)):
        f = list_to_numpy_function(fexpr)
        return f
    else:
        try:
            f = sym_sympify(expr_array=fexpr)
        except Exception:
            raise SympifyError(par_name=par_name, user_input=fexpr)
        
        # univariate
        is_univariate = ValidateArgs.check_boolean(user_input=is_univariate, default=False)

        # f - continued
        if is_symexpr(f):
            fvars = f.free_symbols
            nvars = len(fvars)
            if is_univariate and nvars != 1:
                raise UnivariatePolyError(par_name=par_name, user_input=fexpr)
        else:
            raise SymbolicExprError(par_name=par_name, user_input=fexpr)
        
        if variables is None:
            fvars = tuple(fvars)
        else: 
            fvars = conv_to_arraylike(
                array_values=variables, 
                includes_str=True,  
                to_tuple=True,
                par_name='variables'
            )
        fexpr = lambdify(fvars, f, 'numpy')

    return fexpr


def list_to_numpy_function(equations: list[str]) -> Callable:
    """
    Convert a list of equations given as strings into 
    a NumPy-compatible function.

    Parameters
    ----------
    equations: list, of str
        List of equations in string format.

    Returns
    -------
    f: A Numpy function.
    """
    symbols = []
    sympy_eqs = []
    
    for eq in equations:
        sympy_eq = sympify(eq)
        sympy_eqs.append(sympy_eq)
        for symbol in sympy_eq.free_symbols:
            if symbol not in symbols:
                symbols.append(symbol)
    
    num_vars = len(symbols)
    lambda_funcs = [lambdify(symbols, eq, 'numpy') for eq in sympy_eqs]
    
    def f(x):
        """
        Evaluate the set of equations with the given input values.

        Parameters
        ----------
        x: {list, tuple}
            Input array where each element corresponds to a variable.

        Returns
        -------
        numpy_array: Results of the equations.
        """
        # Ensure the input array has the correct length
        if len(x) != num_vars:
            raise ValueError(
                f"Expected input array to have {num_vars} elements but got {len(x)}"
            )
        
        # Evaluate each lambda function with the input values
        results = [f(*x) for f in lambda_funcs]
        return array(results)
    
    return f


def lambdify_system(
    system_eqtns: list[str] | Callable,
    is_univariate: bool = False, 
    variables: list[str] | None = None, 
    par_name: str = 'fexpr'
) -> Callable:
    """
    Lambdify a system of symbolic equations.

    Parameters
    ----------
    system_eqtns : list_like
        List of symbolic equations to be lambdified.

    Returns
    -------
    f : Callable
        A function representing the lambdified system.
    """
    if is_function(system_eqtns):
        f = system_eqtns
    else:
        f = zeros(0)
        for kth_eqtn in system_eqtns:
            try:
                g = sym_lambdify_expr(
                    fexpr=kth_eqtn, 
                    is_univariate=is_univariate, 
                    variables=variables,
                    par_name=f'{par_name}: {kth_eqtn}'
                )
                f = append(f, g)
            except Exception as e:
                raise Exception(e)
        
    return f


def sym_get_expr_vars(expr_array: str | list[str]) -> tuple[Symbol]:
    """
    Get unknown variables in a single or a system of equations.

    Parameters
    ----------
    expr_array : str or list of list of str
        Single or multiple equations represented as strings.

    Returns
    -------
    fvars : tuple of Symbol
        Tuple containing the unknown variables found in the equations.

    Raises
    ------
    IterableError
        If the input is not iterable.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.sym_get_expr_vars('x ** 2 + y')
    (x, y)
    >>> stm.sym_get_expr_vars(['x ** 2 + y', 'z - 1'])
    (x, y, z)

    """
    if not is_iterable(expr_array, includes_str=True):
        raise IterableError(par_name='expr_array', user_input=expr_array)
    if isinstance(expr_array, str):
        expr_array = str(expr_array).replace('=', '-')
    else:
        expr_array = [
            str(item).replace('=', '-') for item in flatten(expr_array)
        ]
    eqtns = sympify(expr_array)
    if not isinstance(eqtns, (list, tuple)):
        fvars = eqtns.free_symbols
    else:
        fvars = set(flatten([eqtn.free_symbols for eqtn in flatten(eqtns)]))
    fvars = tuple(set(fvars))
    
    return fvars


def variables_count(fexpr: Symbol) -> int:
    """
    Count the number of unique variables in a symbolic expression.

    Parameters
    ----------
    fexpr : sympy.Expr
        The symbolic expression for which variables are to be counted.

    Returns
    -------
    int
        The count of unique variables present in the expression.

    Raises
    ------
    SymbolicExprError
        If the input is not a symbolic expression.

    Examples
    --------
    >>> from sympy import symbols
    >>> x, y, z = symbols('x y z')
    >>> equation = x ** 2 + y*z - z
    >>> variables_count(equation)
    3
    """
    if not is_symexpr(fexpr):
        raise SymbolicExprError(par_name='equation', user_input=fexpr)
    return len(fexpr.free_symbols)


def object_properties(
    result_name: str,
    object_values: ...,
    system: bool = False,
    data_frame: bool = False,
    df_labels: list | None = None,
    category: str = "",
    application: str = "",
    query_string: str = "",
):
    """
    Get the properties of the provided `object_values`.

    Parameters
    ----------
    result_name : str
        Name of the result.
    object_values : object
        The object for which properties are to be determined.
    system : bool, optional (default=False)
        Indicates whether the object represents a system of equations.
    data_frame : bool, optional (default=False)
        Indicates whether the object is a DataFrame.
    df_labels : list_like, optional  (default=False)
        Labels for DataFrame rows and columns.
    category : str, optional (default='')
        Category of the object.
    application : str, optional (default='')
        Application context of the object.
    query_string : str, optional (default='')
        Additional query string.

    Returns
    -------
    Tuple
        A tuple containing the properties of the object:
        - result_name : str
        - object_values : object
        - object_latex : str
        - object_structure : str
        - object_dimensions : str
        - object_size : str
        - df_labels : str
        - category : str
        - application : str
        - query_string : str
    """
    from stemlab.core.htmlstyles import dataframe_to_html
    from stemlab.core.htmlatex import tex_to_latex
    from stemlab.statistical.dataframes import dataframe_labels

    error_list = []
    row_names, column_names = [""] * 2
    
    if isinstance(object_values, str):
        try:
            object_values = sympify(object_values)
        except Exception:  # e.g. for graphs
            pass

    if system:
        object_structure = "System"
        try:
            object_values = Matrix(object_values)
        except Exception:  # it is possibly a scalar or symbolic expression
            object_values = Matrix([object_values])

        if len(list(object_values)) > 1:
            symbols_count = systems_constants(object_values)[-1]
            if symbols_count > 0:
                object_latex = tex_to_latex(object_values)
                eqtn_count = len(object_values)
                unknowns = f"unknown{str_singular_plural(n=symbols_count)}"
                object_dimensions = (
                    f"{eqtn_count} equations: {symbols_count} {unknowns}"
                )
            else:  # there are constants
                r, c = object_values.shape
                object_dimensions = f"{r}R x {c}C"
                structure_mapping = {
                    (r > 1 and c > 1) or (r == 1 and c == 1): "Matrix",
                    r > 1 and c == 1: "Column vector",
                    r == 1 and c > 1: "Row vector"
                }
                object_structure = structure_mapping.get(True, "Scalar")
                object_latex = tex_to_latex(object_values)
        else:  # scalar or function
            if is_symexpr(object_values) or is_iterable(object_values):
                if is_symexpr(object_values) or prod(object_values.shape) == 1:
                    if is_iterable(object_values):
                        object_values = flatten(object_values)[0]
                    object_latex = tex_to_latex(object_values)
                    symbols_ = object_values.free_symbols
                    function_variables = sorted(list(map(str, symbols_)))
                    n = len(function_variables)
                    object_structure = "Function"
                    s = str_singular_plural(n=n)
                    object_dimensions = f"{n} variable{s}: {symbols_}"
            else:
                object_latex = tex_to_latex(object_values)
                object_structure = "Scalar"
                object_dimensions = "0R x 0C"
    else:
        if (
            isinstance(object_values, (tuple, list, ndarray)) 
            or "Matrix" in str(object_values)
        ):
            object_values = Matrix(object_values)
            r, c = object_values.shape
            if data_frame:  # data frame
                object_structure = "DataFrame"
                object_dimensions = f"{r}R x {c}C"
                row_names, column_names = df_labels.replace(" ", "").split(";")
                # call the dataframe_labels() function
                row_names, column_names, error_list = dataframe_labels(
                    object_values, row_names, column_names, error_list=[]
                )
                
                if not error_list:
                    df_labels = f"{row_names}; {column_names}"
                    df = DataFrame(
                        object_values.tolist(), 
                        index=row_names, 
                        columns=column_names
                    )
                    object_latex = dataframe_to_html(df).replace(
                            '<table border="1" class="dataframe" rules = "all">',
                            '<table style="border:1>'
                    )
            else:  # matrix
                object_latex = tex_to_latex(object_values)
                object_dimensions = f"{r}R x {c}C"
                structure_mapping = {
                    (r > 1 and c > 1) or (r == 1 and c == 1): "Matrix",
                    r > 1 and c == 1: "Column vector",
                    r == 1 and c > 1: "Row vector"
                }
                object_structure = structure_mapping.get(True, "Scalar")
        elif is_symexpr(object_values):
            object_latex = tex_to_latex(object_values)
            function_variables = object_values.free_symbols
            function_variables = list(map(str, function_variables))
            function_variables.sort()
            object_structure = "Function"
            n = len(function_variables)
            if n == 0:
                object_structure = "Scalar"
                object_dimensions = "0R x 0C"
            object_dimensions = f"{n} variable: {object_values.free_symbols}"
            if n == 1:
                object_dimensions.replace('s:', ':')
        elif isinstance(object_values, dict):
            object_latex = tex_to_latex(object_values)
            object_structure = "Dictionary"
            object_dimensions = f"{len(object_values)} elements"
        elif isinstance(object_values, str) and category == "Visualilzation":
            object_latex = object_values
            object_structure = "Graph"
            object_dimensions = "NA"
        else:
            object_latex = tex_to_latex(object_values)
            object_structure = "Scalar"
            object_dimensions = "0R x 0C"

    # correct scalar that may have passed
    if "0 variables" in object_dimensions:
        object_structure = "Scalar"
        object_dimensions = "0R x 0C"

    # correct if object value is 1 x 1 matrix 
    # (case when user clicks system for a scalar object)
    if object_structure == "Scalar" and is_iterable(object_values):
        try:
            object_values = object_values[0, 0]
        except Exception:
            pass

    # polynomials
    if "Poly" in str(object_values):
        object_structure = "Polynomial"
        poly_expr = object_values.as_expr()
        n = len(poly_expr.free_symbols)
        s = str_singular_plural(n=n)
        object_dimensions = f"{n} variable{s}"
    elif any(op in str(object_values) for op in ["<", ">", "|", "&"]):
        object_structure = "Inequality"
        try:
            n = len(object_values.free_symbols)
        except Exception:
            n = 0
            
        object_dimensions = f"{n} unknown{str_singular_plural(n=n)}"

    object_size = f"{round(sys.getsizeof(object_values)/1000, 2)}KB"

    try:
        object_values = object_values.tolist()
    except Exception:
        pass

    # ensure that row and column names are strings.
    if data_frame:
        if row_names:
            if isinstance(row_names, (list, tuple)):
                row_names = ", ".join(row_names)

        if column_names:
            if isinstance(column_names, (list, tuple)):
                column_names = ", ".join(column_names)

        df_labels = f"{row_names}; {column_names}"
    
    return (
        result_name,
        object_values,
        object_latex,
        object_structure,
        object_dimensions,
        object_size,
        df_labels,
        category,
        application,
        query_string
    )
        

def systems_constants(
    system_matrix: list
) -> tuple[bool, int, int]:
    """
    Check if list has symbolic variables and return the count of 
    symbolic variables and constants (if any).

    Parameters
    ----------
    system_matrix : list
        The array-like object to be checked.

    Returns
    -------
    tuple
        A tuple containing three elements:
        - constants_found : bool
            Indicates whether constants are found in the array.
        - constants_count : int
            The count of constants in the array.
        - symbols_count : int
            The count of symbolic variables in the array.
    """
    if isinstance(system_matrix, (list, tuple, ndarray)):
        system_matrix = Matrix(system_matrix)

    if isinstance(system_matrix, (int, float, str)):
        system_matrix = Matrix(sympify([system_matrix]))

    system_matrix = flatten(system_matrix)  # reshape to 1 column
    constants_count = 0
    symbols_list = []

    for equation in system_matrix:
        symbols = equation.free_symbols
        if symbols:  # symbolic variables found
            symbols_list.append(list(symbols))
        else:
            constants_count += 1

    constants_found = constants_count > 0
    symbols_count = len(set(flatten(symbols_list)))

    return (constants_found, constants_count, symbols_count)


def inf_oo(strng: str) -> str:
    """
    Replace variations of infinity-related strings with 'oo'.

    Parameters
    ----------
    strng : str
        The input string where infinity-related strings are to be replaced.

    Returns
    -------
    str
        The string with infinity-related strings replaced by 'oo'.
    """
    pattern = r'\b(inf(inity)?|Inft?y?)\b'
    return re.sub(pattern, 'oo', strng)
