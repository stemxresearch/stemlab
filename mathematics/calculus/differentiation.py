from typing import Literal, Callable

from sympy import diff, Expr, Symbol
from stemlab.core.symbolic import sym_simplify_expr, sym_sympify
from stemlab.core.validators.validate import ValidateArgs


SYMPLIFY_METHODS = [
    'none', 'cancel', 'collect', 'factor', 'expand', 'simplify', 'together'
]


def sym_diff_parametric(
    f: str | Expr | Callable, 
    g: str | Expr | Callable, 
    dependent: str | Symbol, 
    n: int = 1, 
    simplify_method: Literal[
        'cancel', 'collect', 'factor', 'expand', 'symplify', 'together', None
    ] = 'factor', 
    collect_term: str | Expr | None = None
):
    """
    Computes the nth derivative of `g` with respect to a dependent 
    variable t, divided by the nth derivative of `f` with respect to `t`.

    Parameters
    ----------
    f : {str, sympy.Expr, callable}
        The expression representing the function `f`.
    g : {str, sympy.Expr, callable}
        The expression representing the function `g`.
    dependent : {str, sympy.Expr}
        The variable with respect to which differentiation is 
        performed.
    n : int, optional (default=1)
        The order of differentiation. Must be an integer between 
        `1` and `10`, inclusive.
    simplify_method : str, optional (default='factor')
        The method used to simplify the result. It will be further 
        validated in the `sym_simplify_expr()` function.
        ==============================================================
        simplify_method     Description  
        ==============================================================
        cancel ........... Cancel common factors in a rational 
                           function `f`.
        collect .......... Collect additive terms of an expression.
        expand ........... Expand an expression.
        factor ........... Compute the factorization of expression, 
                           `f`, into irreducibles.
        simplify ......... Simplifies the given expression.
        together ......... Combine rational expressions using symbolic 
                           methods.
        ==============================================================
    collect_term : {None, str, sym.Expr}, optional (default=None)
        The term (expression) by which the collection of terms should
        performed.

    Returns
    -------
    result : sympy.Expr
        The result of the differentiation, simplified according to the 
        specified method.
        
    Examples
    --------
    >>> import stemlab as stm

    >>> x = 't^3 + 3 * t^2'
    >>> y = 't^4 - 8 * t^2'
    >>> stm.sym_diff_parametric(f=x, g=y, dependent='t', n=3)
    -(3*t ** 2 - 4)/(9*t ** 3*(t - 2) ** 3*(t + 2) ** 2)

    >>> x = 'sin(t)'
    >>> y = 'cos(t)'
    >>> stm.sym_diff_parametric(f=x, g=y, dependent='t', n=2,
    ... simplify_method='simplify')
    1/(sin(t) ** 2*cos(t))

    >>> x = 't * sin(t)'
    >>> y = 't + t^2'
    >>> stm.sym_diff_parametric(f=x, g=y, dependent='t', n=2,
    ... simplify_method='simplify')
    (-2*t*cos(t) + (2*t + 1)*(-t*sin(t) + 2*cos(t)) - 2*sin(t))/((2*t + 1) ** 2*(t*cos(t) + sin(t)))
    """
    f = sym_sympify(expr_array=f, is_expr=True, par_name='f')
    g = sym_sympify(expr_array=g, is_expr=True, par_name='g')
    t = sym_sympify(
        expr_array=dependent, is_expr=True, par_name='dependent'
    )
    n = ValidateArgs.check_numeric(
        par_name='n', 
        limits=[1, 10], 
        is_positive=True, 
        is_integer=True, 
        user_input=n
    )
    simplify_method = 'none' if simplify_method is None else simplify_method
    simplify_method = ValidateArgs.check_member(
        par_name='simplify_method', 
        valid_items=SYMPLIFY_METHODS,
        user_input=simplify_method, 
        default=None
    )

    # begin
    # -----
    if n == 1:
        derivative = diff(g, t) / diff(f, t)
    else:
        # perform the normal differentiation recurssively
        derivative = diff(sym_diff_parametric(g, f, t, n - 1), t) / diff(f, t)
    
    # simplify result
    result = sym_simplify_expr(
        fexpr=derivative, 
        simplify_method=simplify_method, 
        collect_term=collect_term
    )
    
    return result