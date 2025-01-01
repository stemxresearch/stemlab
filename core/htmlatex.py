from turtle import shape
from typing import Any, Callable, Literal

from numpy import array, argwhere, insert, vectorize, vstack, hstack, matrix
from sympy import latex, sympify, Matrix, flatten, Integral, Expr
from pandas import DataFrame
from IPython.display import display, Latex
from IPython.core.display import HTML

from stemlab.core.datatypes import ArrayMatrixLike, ListArrayLike, NumpyArray, is_function
from stemlab.core.css import color_bluish
from stemlab.core.arraylike import is_iterable, conv_to_arraylike
from stemlab.core.symbolic import is_symexpr, inf_oo
from stemlab.core.validators.errors import (
    CoeffConstantsError, PandifyError, SympifyError
)
from stemlab.core.base import str_remove_trailing_zeros
from stemlab.core.base.dictionaries import Dictionary
from stemlab.core.decimals import fround
from stemlab.core.validators.validate import ValidateArgs


def color_values(value, color: str = '#01B3D1', html: bool = False) -> str:
    """
    Apply color to a value.

    Parameters
    ----------
    value : any
        The value to be colored.
    color : str, optional (default='#01B3D1')
        The name or HEX code of the color to be applied.
    html : bool, optional (default=False)
        If `True`, return HTML formatted string.

    Returns
    -------
    colored : str
        A colored string.
    """
    try:
        if html:
            colored = (
                f'<span style="color:{color};font-weight:600;">{value}</span>'
            )
        else:
            colored = f'{{\\color{{{color}}}{{{value}}}}}'
    except Exception as e:
        raise Exception(f'color_values(): {e}')
    
    return colored

    
def tex_aligned_latex(lhs: str ='A', rhs: list = []) -> str:
    """
    Return latex aligned equations.

    Parameters
    ----------
    lhs : str, optional (default='A')
        Left hand side of the aligned equation.
    rhs : list_like, (optional (default=[]))
        Right hand side of the aligned equation.

    Returns
    -------
    html_latex_str: str
        A string containing the Latex syntax for an aligned equation.

    Examples
    --------
    >>> import stemlab as stm

    >>> lhs = 'k'
    >>> rhs = ['4 * (5 + 3) - 15', '4 * 8 - 15', '32 - 15', 17]
    >>> stm.tex_aligned_latex(lhs, rhs)
    \\(\\begin{aligned} k &= 4 * (5 + 3) - 15 \\\\[5pt] &= 4 * 8 - 15 \\\\[5pt] &= 32 - 15 \\\\[5pt] &= 17 \\\\[5pt] \\end{aligned}\\)
    
    >>> lhs = '\\Big(\\cos^2x + \\sin^2 x \\Big)^3'
    >>> rhs = [
    '\\Big(\cos^4x + 2 \\sin^2 \cos^2x + \\sin^4x \Big) \Big(\cos^2x + \sin^2x \\Big)',
    '\\cos^6x + \cos^4x \\sin^2x + 2 \\cos^4x \\sin^2x + 2 \\sin^4x \\cos^2x + \\sin^4x \\cos^2x + \\sin^6x',
    '\\cos^6x + \sin^6x + 3 \\cos^4x \\sin^2x + 3 \\sin^4x \\cos^2x',
    '\\cos^6x + \sin^6x + 3 \\sin^2x \\cos^2x \Big(\\cos^2x + \\sin^2x \Big)',
    '\\cos^6x + \sin^6x + \\frac{3}{4} \\Big(4 \\sin^2x \\cos^2 x \\Big)',
    '1-\\frac{3}{4} \\sin x 2x &= \\cos^6x + \\sin^6 x'
    ]
    >>> stm.tex_aligned_latex(lhs, rhs)
    \\(\\begin{aligned} \\Big(\\cos^2x + \\sin^2 x \\Big)^3 &= \\Big(\\cos^4x + 2 \\sin^2 \\cos^2x + \\sin^4x \\Big) \\Big(\\cos^2x + \\sin^2x \\Big) \\\\[5pt] &= \\cos^6x + \\cos^4x \\sin^2x + 2 \\cos^4x \\sin^2x + 2 \\sin^4x \\cos^2x + \\sin^4x \\cos^2x + \\sin^6x \\\\[5pt] &= \\cos^6x + \\sin^6x + 3 \\cos^4x \\sin^2x + 3 \\sin^4x \\cos^2x \\\\[5pt] &= \\cos^6x + \\sin^6x + 3 \\sin^2x \\cos^2x \\Big(\\cos^2x + \\sin^2x \\Big) \\\\[5pt] &= \\cos^6x + \\sin^6x + \\frac{3}{4} \\Big(4 \\sin^2x \\cos^2 x \\Big) \\\\[5pt] &= 1-\\frac{3}{4} \\sin x 2x &= \\cos^6x + \\sin^6 x \\\\[5pt] \\end{aligned}\\)
    """
    tex_list_to_latex = []
    [
        tex_list_to_latex.append(f'& {color_values(item, color="gray", html=False)} \\\\[5pt]') 
        if 'TextMode' in str(item) else tex_list_to_latex.append(f'&= {str(item)} \\\\[5pt]') for item in rhs
    ]
    html_latex_str = (
        f"\\(\\begin{{aligned}} {lhs} {' '.join(map(str, tex_list_to_latex))} \\end{{aligned}}\\)"
    ).replace('TextMode', '')
    
    return html_latex_str


def tex_array_to_latex(
    M: list,
    align: Literal['left', 'center', 'right'] = "right",
    brackets: Literal['[', '(', '|', ''] = '['
) -> str:
    """
    Convert a list-like object to LaTeX array format.

    Parameters
    ----------
    M : list
        The list-like object to be converted.
    align : {'left', 'center', 'right'}, optional (default='right')
        The alignment of columns in the LaTeX array.
    brackets : {'[', '(', '|', ''}, optional (default='[')
        The type of brackets to surround the LaTeX array.

    Returns
    -------
    latex_syntax : str
        The LaTeX representation of the array.

    Examples
    --------
    >>> u = [4, 5, 6]
    >>> stm.tex_array_to_latex(M=u)
    \\left[\\begin{array}{rrr} 4 & 3 & 9\\end{array}\\right]
    >>> v = [[4, 5, 6]]
    >>> stm.tex_array_to_latex(M=v, brackets='(')
    \\left(\\begin{array}{rrr} 4 & 3 & 9\\end{array}\\right)
    >>> w = [[4, 5, 6]]
    >>> stm.tex_array_to_latex(M=w, brackets='')
    \\begin{array}{rrr} 4 & 3 & 9\\end{array}
    >>> P = [[4, 5, 6], [3, 7, 9]]
    >>> stm.tex_array_to_latex(M=P, align='center', brackets='[')
    \\left[\\begin{array}{ccc} 4 & 5 & 6 \\\\ 3 & 7 & 9\\end{array}\\right]
    """
    try:
        M = matrix(M)
        ncols = M.shape[1]
        M = M.tolist()
        delimiter = " \\\\ "
        latex_matrix = f'\\begin{{array}}{{{align[0] * ncols}}} {delimiter.join([" & ".join(map(str, line)) for line in M])}\\end{{array}}'
    except Exception as e:
        raise Exception(f'tex_array_to_latex(): {e}')
    bracket_symbols = {
        '[': ('\\left[', '\\right]'),
        '(': ('\\left(', '\\right)'),
        '|': ('\\left|', '\\right| '),
        '': ('', ''),
    }
    left, right = bracket_symbols.get(brackets, ('', ''))
    latex_syntax = f'{left}{latex_matrix}{right}'

    return latex_syntax


def sta_dframe_color(
    dframe: DataFrame,
    style_indices: list = [],
    values: list = [],
    operator: Literal['==', '<', '<=', '>', '>=', '<1&2', '<=1&2', '<1|2', '<=1|2'] = '==',
    rows: list = [],
    cols: list = [],
    css_styles: str | None = None,
    decimal_points: int = 8,
) -> DataFrame:
    """
    Color DataFrame cells based on specified conditions.

    Parameters
    ----------
    dframe : DataFrame
        DataFrame whose values are to be styled.
    style_indices : array_like, optional (default=[])
        Array indicating the cells to be colored.
    values : {int, float}, optional (default=[])
        Reference value, below or above which values should be colored.
    operator : str, optional (default='')
        Operator to be used for comparison.
    rows : array_like, optional (default=[])
        Rows to be styled.
    cols : array_like, optional (default=[])
        Columns to be styled.
    css_styles : str, optional (default=None)
        CSS (Cascading Style Sheets) for customizing text appearance.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    dframe : pandas.Styler
        DataFrame with specified values (or their indices) highlighted.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> df = stm.dataset_read(name='sales')

    Color values in row 2, col 3 and row 6 column 2
    
    >>> stm.sta_dframe_color(dframe=df, style_indices=[[1, 2], [5, 1], [4, 9]])

    Color values that are less than 2000
    
    >>> stm.sta_dframe_color(dframe=df, values=[2000], operator='<', 
    ... css_styles='color:blue;font-weight:bold;')

    Color values in row 4 and columns 3, 5, and last column
    
    >>> stm.sta_dframe_color(dframe=df, rows=[4], cols=[2, 6, -1],
    ... css_styles='color:blue;font-weight:bold;')

    Color cell (4, 6), values less than 2000, rows 3, 5, and last 
    row, and second last column
    
    >>> stm.sta_dframe_color(dframe=df, style_indices=[[3, 5]],
    ... values=[2000], operator='<', rows=[2, 4, -1], cols=[-2],
    ... css_styles='color:blue;font-weight:bold;')
    """
    # if index / columns have duplicates
    if len(set(dframe.index)) != dframe.shape[0]:
        dframe.index = [f'_{k}' for k in range(dframe.shape[0])]
    if len(set(dframe.columns)) != dframe.shape[1]:
        dframe.columns = [f'_{k}' for k in range(dframe.shape[1])]
    try:
        M = dframe.values.astype(float)
    except Exception: # If DataFrame contains non-numeric values
        M = dframe.values
    
    # Process operator and values
    if operator and len(values) != 0:
        operator = operator.replace(' ', '')
        if len(values) != 1:
            raise ValueError(
                'When an operator is specified, there should be only one '
                'value provided for values.'
            )
        value = float(values[0])
        if operator in ['==', '<', '<=', '>', '>=']:
            values_indices = argwhere(eval(f'M {operator} {value}')).tolist()
        elif operator in ['<1&2', '<=1&2']:
            a, b = map(float, operator.split('&'))
            values_indices = argwhere(
                ((M > a) & (M < b)) | ((M > b) & (M < a))
            ).tolist()
        elif operator in ['<1|2', '<=1|2']:
            a, b = map(float, operator.split('|'))
            values_indices = argwhere(((M < a) | (M > b))).tolist()
        else:
            raise ValueError(
                'Invalid operator. Supported operators are: '
                '==, <, <=, >, >=, <1&2, <=1&2, <1|2, <=1|2.'
            )
    else:
        values_indices = []
    
    # Process rows and columns
    row_indices = [[row, col] for row in rows for col in range(dframe.shape[1])] if rows else []
    col_indices = [[row, col] for col in cols for row in range(dframe.shape[0])] if cols else []
    style_indices = style_indices + values_indices + row_indices + col_indices
    # Apply CSS styling
    try:
        if css_styles is None:
            css_styles = 'color:blue;background-color:#ccc;font-weight:600;'
        dframe = dframe.style.apply(
            css_by_indices, 
            indices=style_indices, 
            css_styles=css_styles, 
            axis=None
        ).format(precision=decimal_points)
    except Exception as e:
        raise ValueError(f'Error styling DataFrame: {e}')

    table_html = HTML(str_remove_trailing_zeros(dframe.to_html()))
    
    return table_html


def css_by_indices(
    data: DataFrame, indices: list, css_styles: str
) -> DataFrame:
    """
    Apply CSS styles to DataFrame cells based on specified indices.

    Parameters
    ----------
    data : DataFrame
        The DataFrame whose cells are to be styled.
    indices : list, of tuple
        List of tuples representing the row and column indices of 
        cells to be styled.
    css_styles : str
        The CSS (Cascading Style Sheets) styles to be applied to the 
        specified cells.

    Returns
    -------
    styled : DataFrame
        A DataFrame with the specified cells styled according to the provided 
        CSS styles.
    """
    try:
        if not isinstance(data, DataFrame):
            data = DataFrame(data)
    except Exception:
        raise PandifyError(par_name='data')
    styled = DataFrame('', index=data.index, columns=data.columns)
    for row, col in indices:
        styled.iat[row, col] = css_styles
    return styled


def sta_dframe_to_html(
    dframe: DataFrame,
    style_indices: list = [],
    values: list = [],
    operator: str = '',
    rows: list = [],
    cols: list = [],
    css_styles: str | None = None,
    row_title: str = '',
    remove_tzeros: bool = False,
    to_html: bool = True,
    decimal_points: int = 8,
) -> str | HTML:
    """
    Convert DataFrame to HTML code.

    Parameters
    ----------
    dframe : DataFrame
        The DataFrame to be converted to HTML.
    style_indices : list, optional (default=[])
        List of cell indices to be styled. For example, 
        [[1, 2], [5, 1]] indicates styling the value in row 2,
        column 3 and row 6, column 2.
    values : list, optional (default=[])
        Reference values for conditional styling.
    operator : str, optional (default='')
        Operator for conditional styling.
    rows : list, optional (default=[])
        Rows to be styled.
    cols : list, optional (default=[])
        Columns to be styled.
    css_styles : str, optional (default=None)
        CSS styles for customizing the appearance of text.
    row_title : str, optional (default='')
        Title for the first column (first column heading).
    remove_tzeros : bool, optional (default=False)
        Whether to remove trailing zeros.
    to_html : bool, optional (default=True)
        Whether to convert the result to an HTML object.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    dframe : str or HTML
        HTML code of the specified DataFrame.

    Examples
    --------
    >>> import stemlab as stm
    >>> df = stm.dataset_read(name='sales')

    Color values in specific rows and columns
    
    >>> stm.sta_dframe_to_html(dframe=df, rows=[4], cols=[2, 6, -1],
    ... css_styles='color:skyblue;font-weight:bold;letter-spacing:1px;font-style:italic;')
    """
    from stemlab.core.base.strings import str_replace_dot_zero

    decimal_points = 8 if decimal_points == -1 else decimal_points
    dframe_original = dframe
    # Apply CSS styling if specified
    if style_indices or values or rows or cols:
        try:
            dframe = sta_dframe_color(
                dframe=dframe,
                style_indices=style_indices,
                values=values,
                operator=operator,
                rows=rows,
                cols=cols,
                css_styles=css_styles,
                decimal_points=decimal_points,
            )
            # If no styling, revert to original DataFrame
            dframe = dframe_original if not css_styles else dframe
        except Exception as e:
            raise Exception(e)

    # Convert DataFrame to HTML and modify formatting
    dframe = dframe.data # get the html code
    dframe = dframe.replace(
        '<table border="1" class="dataframe">',
        '<table border="1" class="dataframe" rules="all">'
    )
    dframe = dframe.replace(
        "<td>",
        '<td style="text-align:right;min-width:450px;height:4px;color:red;font-family:courierxxxx;">'
    )
    dframe = dframe.replace(
        "<th>",
        '<th style="text-align:right;min-width:30px;height:4px;line-height:23px;font-weight:normal;">'
    )
    dframe = dframe.replace("<table", "\n\t\t<table")
    dframe = dframe.replace("</table>", "\n\t\t</table>\t")
    dframe = dframe.replace("<thead>", "\n<thead>")
    dframe = dframe.replace("</thead>", "\t\t\t</thead>")
    dframe = dframe.replace("<tbody>", "\n\t\t\t<tbody>")
    dframe = dframe.replace("</tbody>", "\t\t\t</tbody>")
    dframe = dframe.replace("<tr", '\t\t\t<tr style="line-height:23px;"')
    dframe = dframe.replace("</tr", "\t\t\t</tr")
    dframe = dframe.replace("<th", "\t\t\t\t<th")
    dframe = dframe.replace("<td", "\t\t\t\t<td")
    dframe = dframe.replace(
        "<th ",
        '<th style="text-align:center;font-weight:normal;line-height:23px;" '
    )
    dframe = dframe.replace(
        "></th>",
        f'><div style="padding:0px;">{row_title}</div></th>'
    )
    dframe = dframe.replace(
        ">&nbsp;</th>",
        f'><div style="padding:0px;">{row_title}</div></th>'
    )
    dframe = dframe.replace(
        "<td ",
        f'<td style="font-family:courierxxxx;font-size:18px;text-align:right;min-width:50px;"'
    )
    dframe = dframe.replace(">NaN</td>", "></td>").replace(
        ">nan</td>", "></td>"
    )
    dframe = HTML(f"<p>{dframe}</p>") if to_html else f"<p>{dframe}</p>"
    if remove_tzeros:
        try:
            dframe = str_replace_dot_zero(result=dframe)
        except:
            pass

    return dframe


def tex_display_latex(
    lhs: str | list, rhs: str | list, auto_display: bool=True
) -> bool:
    """
    Display results in Latex format.

    Parameters
    ----------
    lhs : {str, list_like}
        The left hand side of the mathematical input.
    rhs : {str, list_like}
        The right hand side of the mathematical input.
    auto_display : bool, optional (default=True)
        Print the results automatically.

    Returns
    -------
    None

    Examples
    --------
    >>> import stemlab as stm

    >>> lhs = ['f(x)']
    >>> rhs = ['(x^2 - 1) / (x^2 + 8 * x + 15)']
    >>> stm.tex_display_latex(lhs, rhs)

    >>> lhs = ['f(x)', 'g(x, y)']
    >>> rhs = ['(x^2 - 1) / (x^2 + 8 * x + 15)',
    ... '(x^2 * y * sin(y)^2 + x^2 * y * cos(y)^2) / (x^2 + x)']
    >>> stm.tex_display_latex(lhs, rhs)

    >>> lhs = ['f(x)', 'dy/dx']
    >>> rhs = ['(x^2 - 1) / (x * sqrt(x^2 + 1))',
    ... '(3*x ** 2 + 1)/(x ** 2*(x ** 2 + 1)**(3/2))',
    ... '(3*x ** 2 + 1)/(x ** 4*sqrt(x ** 2 + 1) + x ** 2*sqrt(x ** 2 + 1))',
    ... '(-x ** 2*(x ** 2 - 1) + 2*x ** 2*(x ** 2 + 1) - (x ** 2 - 1)*(x ** 2 + 1))/(x ** 2*(x ** 2 + 1)**(3/2))']
    >>> stm.tex_display_latex(lhs, rhs)

    >>> x, theta = sym.symbols('x alpha')
    >>> h = sym.sympify('exp(-2 * alpha)/x + cos(pi * x)')
    >>> lhs = ['f(x)', 'dy/dx'] + [f'd^{k} * x/(dy^{k})' for k in range(2, 6)]
    >>> rhs = [sym.diff(h, x, n) for n in range(6)]
    >>> stm.tex_display_latex(lhs, rhs)
    """
    lhs = conv_to_arraylike(
        array_values=lhs, flatten_list=False, includes_str=True, par_name='lhs'
    )
    rhs = conv_to_arraylike(
        array_values=rhs, flatten_list=False, includes_str=True, par_name='rhs'
    )
    # check number of elements
    if len(lhs) > len(rhs):
        raise ValueError(f"'lhs' cannot have more elements than 'rhs'")
    if len(lhs) < len(rhs):
        lhs_count, rhs_count = len(lhs), len(rhs)
        lhs += [''] * (lhs_count - rhs_count)
    # auto_display
    auto_display = ValidateArgs.check_boolean(auto_display, default=True)
    # begin latex
    latex_array = ['\\begin{aligned} \\\\']
    for index in range(len(lhs)):
        try:
            lhs_sym = sympify(lhs[index])
        except Exception:
            lhs_sym = ''
        try:
            rhs_sym = sympify(rhs[index])
        except Exception:
            raise SympifyError(
                par_name=f'rhs -> {rhs[index]}', user_input=rhs[index]
            )
        latex_row = (
            f'\\displaystyle {latex(lhs_sym)} &= {latex(rhs_sym)} \\\\[7pt]'
        )
        latex_array.append(latex_row)
    latex_array.append('\\end{aligned} \\\\[5pt]')
    latex_str = f"${' '.join(latex_array)}$".replace('.0 ', '')
    TEX = Latex(latex_str)
    if auto_display:
        display(TEX)
        return None
    else:
        return TEX


def tex_integral_to_latex(
    fexpr: str | Expr | Callable, 
    limits: list = [], 
    int_var: str = 'x', 
    int_times: int = 1, 
    definite: bool = True
) -> str:
    """
    Parameters
    ----------
    fexpr : {str, symbolic, callable}
        An expression containing the first parametric equation.
    limits : array_like, optional (default (default=[]))
        Lower and upper limits of integration.
    int_var : str, optional (default='x')
        The variable of integration.
    int_times : int, optional (default=1)
        Number of times `fexpr` should be integrated.
    definite : bool, optional (default=True)
        If `True`, then a definite integral will be peformed, 
        otherwise, indefinite integration will be performed.

    Returns
    -------
    latex_syntax : str
        A string containing the Latex syntax for the entered integral.

    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm
    
    >>> f = stm.tex_integral_to_latex(fexpr='cos(x) * exp(x)', int_var='x',
    ... int_times=1, definite=False)
    >>> print(f)
    $\displaystyle \int e^{x} \cos{\left(x \right)}\, \mathrm{d}x$
    
    >>> f = stm.tex_integral_to_latex(fexpr='cos(x) * exp(x)', int_var='x',
    ... int_times=3, definite=False)
    >>> print(f)
    $\displaystyle \iiint e^{x} \cos{\left(x \right)}\, \mathrm{d}x\, \mathrm{d}x\, \mathrm{d}x$
    
    >>> f = stm.tex_integral_to_latex(fexpr='cos(x) * exp(x)',
    ... limits=[0, np.inf], int_var='x, y', definite=False)
    >>> print(f)
    $\displaystyle \iiint e^{x} \cos{\left(x \right)}\, \mathrm{d}x\, \mathrm{d}x\, \mathrm{d}x$
    """
    fexpr = sympify(fexpr)
    sym_vars = list(fexpr.free_symbols)
    fexpr_symbols = sorted([str(item) for item in sym_vars])
    symbols_left = ", ".join(fexpr_symbols)
    if definite:
        if isinstance(limits, str):
            limits = inf_oo(limits)
        limits = sympify(limits)
        try:
            integral_notation = Integral(fexpr, *limits)
        except Exception:
            try:
                integral_notation = Integral(
                    fexpr, (limits[0], limits[1], limits[2])
                )
            except Exception as e:
                try:
                    if len(sym_vars) == 1:
                        integral_notation = Integral(
                            fexpr, (sym_vars[0], limits[0], limits[1])
                        )
                    else:
                        raise Exception(
                            f'Expected 1 symbolic variable but got '
                            f'{len(sym_vars)}'
                        )
                except Exception as e:
                    raise Exception(e)
    else:
        if isinstance(int_var, str):
            int_var = inf_oo(int_var)
        int_var = sympify(int_var)
        if int_times > 1 and not isinstance(int_var, (tuple, list)):
            int_var = [int_var] * int_times
        try:
            integral_notation = Integral(fexpr, *int_var)
        except Exception:  # it could be a symbolic variable (one variable given)
            integral_notation = Integral(fexpr, int_var)

    integral_notation = (
        latex(integral_notation)
        .replace("int f", f"int f({symbols_left})")
        .replace(" d", f" \\mathrm{{d}}")
    )

    latex_syntax = f"$\\displaystyle {integral_notation}$"

    return latex_syntax


def tex_list_to_latex(
    lst: list, 
    align: Literal['left', 'right'] = 'right'
) -> str:
    """
    Converts an array_like object to latex. It is particularly usefull 
    when the values of the array cannot be converted to mathematical 
    expressions using `sym.sympify()` function from the Sympy library.

    Parameters
    ----------
    lst : list_like
        Object containing the values of the array to be converted to Latex.
    align : {'left', 'right'}, optional (default='right')
        Alignment of the values of the array; 'left' or 'right'.
    
    Returns
    -------
    latex_syntax : str
        A string containing the Latex syntax for the entered array.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> u = ['Quarter 1', 'Quarter 2', 'Quarter 3', 'Quarter 4']
    >>> stm.tex_list_to_latex(lst=u)
    \\begin{array}{rrrr}Quarter 1 & Quarter 2 & Quarter 3 & Quarter 4 \\end{array}
    """
    lst = [flatten(lst)]
    ncols = align[0] * len(lst[0])
    latex_syntax = f'\\begin{{array}}{{{ncols}}} {"".join([" & ".join(map(str, line)) for line in lst])} \\end{{array}}'
    
    return latex_syntax


def tex_matrix_to_latex_eqtns(
    A: NumpyArray | Matrix,
    b: NumpyArray | Matrix,
    displaystyle: bool = True,
    hspace: int = 0,
    vspace: int = 7,
    inline: bool = False
) -> str:
    """
    Convert matrix to linear equations.

    A : {list, tuple, Series, NDarray}
        Coefficients matrix.
    b : {list, tuple, Series, NDarray}
        Constants matrix.
    hspace : int, optional (default=0)
        Horizontal space before the equations.
    vspace : int, optional (default=7)
        Vertical space between rows.
    inline : bool, optional (default=True)
        If `True`, then use '$' (i.e. equations within text), 
        otherwise use '$$' (i.e. equations on new line).

    Returns
    -------
    Axb_latex : str
        A string containing the Latex syntax for an linear equations.

    Examples
    --------
    >>> A = [[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1],
    ... [0, 3, -1, 8]]
    >>> b = [6, 25, -11, 15]
    >>> M = stm.tex_matrix_to_latex_eqtns(A, b)
    >>> print(M)
    $$
    \displaystyle
    \\begin{aligned} 
        10 x_{1} - x_{2} + 2 x_{3} &= 6 \\[7pt] 
        - x_{1} + 11 x_{2} - x_{3} + 3 x_{4} &= 25 \\[7pt] 
        2 x_{1} - x_{2} + 10 x_{3} - x_{4} &= -11 \\[7pt] 
        3 x_{2} - x_{3} + 8 x_{4} &= 15 
    \end{aligned} 
    $$
    """
    A = Matrix(A)
    A_nrows = A.shape[0]
    b = Matrix(b)
    b_nrows, b_ncols = b.shape
    if b_ncols != 1:
        b = b.T
    if A_nrows != b_nrows:
        raise Exception(
            CoeffConstantsError(
                par_name=['A', 'b'], user_input=[A, b], to_list_join=True
            )
        )
    Ax = []
    # create LHS Ax
    for row in range(A.shape[0]):
        terms_joined = " + ".join(
            [f"{value} * x{k + 1}" for k, value in enumerate(A[row, :])]
        )
        Ax.append(sympify(terms_joined))
    Ax = Matrix(Ax)
    # join LHS (Ax) to RHS (b) to form Ax = b
    Axb = []
    for row in range(A.shape[0]):
        Axb.append([f"{latex(Ax[row, 0])} &= {latex(b[row, 0])}"])

    displaystyle = "\\displaystyle" if displaystyle else ""
    dollar = "$" if inline else "$$"
    hspace = "" if hspace == 0 else "\\hspace{" + str(hspace) + "cm}"

    delimiter = f" \\\\[{str(vspace)}pt] \n\t"
    Axb = delimiter.join(flatten(Axb))
    Axb_latex = (
        dollar + "\n" + hspace
        + displaystyle + "\n\\begin{aligned} \n\t"
        + Axb + " \n\\end{aligned} \n" + dollar
    )

    return Axb_latex


def tex_table_to_latex(
    data: ListArrayLike,
    row_names: list | None = None,
    col_names: list | None = None,
    row_title: str = '',
    caption: str = 'Table',
    first_row_bottom_border: bool = False,
    last_row_top_border: bool = False,
    decimal_points: int = 8
):
    """
    Convert a table of statistics (as an array) to a Latex array
    syntax.

    Parameters
    ----------
    data : array_like
        An array_like object with the statistics.
    row_names : array_like
        Row names.
    col_names : array_like
        Column names.
    row_title : str, optional (default='')
        The title / heading of the rows.
    caption : str, optional (default='Table')
        Table caption.
    first_row_bottom_border : bool, optional (default=False)
        If `True`, a bottom border will be added on first row
    last_row_top_border : bool, optional (default=False)
        If `True`, a top border will be added on last row
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    html_latex : str
        String containing the html and Latex syntax.

    Examples
    --------
    import numpy as np
    import stemlab as stm

    >>> M = np.array(
        [[2318, 4276, 1664, 1279],
        [3431, 1246, 3558, 4503],
        [2282, 2299, 3956, 5467],
        [1855, 2805, 5212, 5704],
        [3093, 3576, 5137, 5024]]
    )
    >>> stm.tex_table_to_latex(
        data=M, col_names=['Qtr 1', 'Qtr 2', 'Qtr 3', 'Qtr 4']
    )
    ['<span style="color:#01B3D1;"><strong>Table</strong></span><br />\\(\\begin{array}{l|rrrr} \\hline 1 & \\text{Qtr 1} & \\text{Qtr 2} & \\text{Qtr 3} & \\text{Qtr 4} \\\\ 2 & 2318 & 4276 & 1664 & 1279 \\\\ 3 & 3431 & 1246 & 3558 & 4503 \\\\ 4 & 2282 & 2299 & 3956 & 5467 \\\\ 5 & 1855 & 2805 & 5212 & 5704 \\\\ 6 & 3093 & 3576 & 5137 & 5024 \\\\ \\hline \\end{array}\\)']
    """
    from stemlab.core.decimals import fround

    html_latex = []
    try:
        data = fround(data, decimal_points)
    except:
        pass
    
    if col_names is not None:
        col_names = [f'\\mathrm{{{col_name}}}' for col_name in col_names]
        last_column_name = col_names[-1].title()
        if ('Total' in last_column_name or 'Sum' in last_column_name
            or first_row_bottom_border is True):
            col_names[-1] = f'{col_names[-1]} \\hline'
        
        if col_names is not None:
            data = vstack([col_names, data])
            
    if isinstance(row_title, str):
        row_title = [row_title]
    row_title[0] = f'\\mathrm{{{row_title[0]}}}'
    
    if row_names is not None:
        row_names = array(
            [[f'\\mathrm{{{row_name}}}' for row_name in row_names]]
        ).T
        row_names = insert(arr=row_names, obj=0, values=row_title, axis=0)
        row_names = row_names.astype('<U250')
        last_row_name = row_names[-1][0].title()
        if ('Total' in last_row_name or 'Sum' in last_row_name
            or first_row_bottom_border is True):
            row_names[-1][0] = f' \\hline {row_names[-1][0]}'

        if row_names is not None:
            data = hstack([row_names, data])
    
    results_latex = f'\\({tex_array_to_latex(data, brackets="")}\\)'
    results_latex = results_latex\
        .replace('\\hline \\\\', '\\\\ \\hline', 1)\
        .replace('{r', '{l|')\
        .replace("r}", "r} \\hline")\
        .replace("hline} \\\\", "} \\\\ \\hline ")\
        .replace(f"\\end{{array}}", f" \\\\ \\hline \\end{{array}}")\
        .replace('nan', '')\
        .replace('\\hline &', '&')\
        .replace(' (', '~(')
    
    if caption:
        caption_number = f'\\color{{blue}}{{\\text{{Table No: }}}}'
        caption = f'{caption_number}\\text{{{caption}}}'
        caption = f"\\({caption}\\)<br />" if caption else ""
    else:
        caption = ""
    html_latex = f'{caption}{str_remove_trailing_zeros(results_latex)}'

    return html_latex


def tex_to_latex(
    M: ArrayMatrixLike,
    align: Literal['left', 'right'] = 'right',
    brackets: Literal['[', '(', '|', ''] = '[',
    scalar_bracket: bool = False,
    raise_exception: bool = False
):
    """
    Converts an array_like object to latex.

    Parameters
    ----------
    M : array_like
        array-like object containing the values of the array. The 
        values of the array must be converted to
    align : str, optional (default='right')
        Alignment of the values of the array.
    brackets : {'[', '(', '|', ''}, optional (default='[')
        Brackets to be used;  
        ==========================================================  
        brackets                       Description   
        ==========================================================  
        '[' .......................... Use square brackets []  
        '(' .......................... Use parenthesis brackets ()  
        '|' .......................... Use | e.g. for determinants  
        '' ........................... No brackets / parenthesis  
        ==========================================================  
    scalar_bracket : bool, optional (default=False)
        If `True`, scalars will be enclosed within `brackets`. 
    raise_exception : bool, optional (default=False)
        If `True`, an Exception will be raised if a value cannot be 
        converted, otherwise, the value will be returned as it was 
        entered.
    
    Returns
    -------
    latex_syntax : str
        A string containing the Latex syntax for the entered array.
    
    Examples
    --------
    >>> u = [4, 5, 6]
    >>> stm.tex_to_latex(M=u)
    \\left[\\begin{array}{rrr} 4 & 3 & 9\\end{array}\\right]
    >>> v = [[4, 5, 6]]
    >>> stm.tex_to_latex(M=v, brackets='(')
    \\left(\\begin{array}{rrr} 4 & 3 & 9\\end{array}\\right)
    >>> w = [[4, 5, 6]]
    >>> stm.tex_to_latex(M=w, brackets='')
    \\begin{array}{rrr} 4 & 3 & 9\\end{array}
    >>> P = [[4, 5, 6], [3, 7, 9]]
    >>> stm.tex_to_latex(M=P, align='center', brackets='[')
    \\left[\\begin{array}{ccc} 4 & 5 & 6 \\\\ 3 & 7 & 9\\end{array}\\right]
    """
    if is_function(M):
        raise ValueError(f"'M' cannot be a callable function.")
    if is_symexpr(M):  # symbolic expression
        result = latex(sympify(M))
    else:  # if not symbolic
        try:
            try: # convert to list if Matrix, or Array
                M = M.tolist()
            except Exception:
                try:
                    M = M.values # if DataFrame
                except Exception:
                    pass
            if is_iterable(M):
                M = Matrix(M)
                # convert to latex
                cols_align = align[0] * M.shape[1]
                result_latex = (
                    latex(M, mat_delim=brackets)
                    .replace(f"begin{{matrix}}", f"begin{{array}}{{{cols_align}}}")
                    .replace(f"end{{matrix}}", f"end{{array}}")
                )
                if "{cc" in result_latex:  # large matrices
                    # replaces the first M.shape[1] occurances of `c` 
                    # with `r`, this avoids replacing any other `c` 
                    # that could be a valid element of the array
                    result = result_latex.replace(
                        "c" * M.shape[1], "r" * M.shape[1], M.shape[1]
                    )
                else:
                    result = result_latex
            else:
                # if string, integer, float or other sympy numerical values
                if scalar_bracket:
                    result = f'\\left({latex(sympify(M))}\\right)'
                else:
                    result = latex(sympify(M))
        except Exception as e:
            if raise_exception:
                raise Exception(str(e))
            else:
                # values that cannot be converted to latex e.g html, latex
                # will be returned the way they were entered
                result = M
    try:
        ncols = Matrix(M).shape[1]
        cols_align = align[0] * ncols
        result = result\
        .replace(f"begin{{matrix}}", f"begin{{array}}{{{cols_align}}}")\
        .replace(f"end{{matrix}}", f"end{{array}}")
    except Exception:
        pass
    
    latex_syntax = result

    return latex_syntax

# the following are not included in __init__.py
# ---------------------------------------------

def color_values(
    value: Any,
    color: str = '#01B3D1',
    html: bool = False,
    is_bold: bool = True
) -> str:
    """
    Colorizes the given value with the specified color.

    Parameters
    ----------
    value : any
        The value to be colored.
    color : str, optional (default='#01B3D1')
        The name or HEX code of the color to be applied..
    html: bool, optional (default=False)
        If `True`, returns HTML formatted string; otherwise, returns 
        LaTeX formatted string. Default is False.
    is_bold : bool, optional (default=False)
        If `True`, bold the string.

    Returns
    -------
    colored : str
        A colored string.

    Raises
    ------
    Exception
        For any unexpected errors.

    Examples
    --------
    >>> from stemlab.core.htmlatex import color_values
    >>> color_values('Hello', color='#FF5733', html=True)
    '<span style="color:#FF5733;font-weight:600;">Hello</span>'
    >>> color_values('World', color='red', html=False)
    '\\textcolor{red}{World}'
    """
    try:
        # Format the string based on the provided options
        if html:
            font_weight = 'font-weight:600' if is_bold else ""
            colored = f'<span style="color:{color};{font_weight};">{value}</span>'
        else:
            colored = f'{{\\color{{{color}}}{{{value}}}}}'

    except Exception as e:
        raise Exception(f'color_values(): {e}')

    return colored


def ptz_table(dct: dict, decimal_points: int) -> str:
    """
    Table for presenting statistics for proportion and mean 
    comparison tests.
    """
    dct = fround(dct, decimal_points=decimal_points)
    dct = Dictionary(dct)
    
    # Determine the test type
    z_boolean = 'z' in dct.table_title
    test_name = 'z' if z_boolean else 't'
    dfn_str = '' if z_boolean else dct.dfn
    dfn_equals = '' if z_boolean else '='
    dfn_name = '' if z_boolean else dct.dfn_name

    # Convert float to int if it's a whole number
    pop_mean = dct.pop_mean
    pop_mean = int(pop_mean) if pop_mean % 1 == 0 else pop_mean

    sample1_name = dct.sample1_name[:12]
    sample2_name = (
        dct.sample2_name[:12] if 'one' not in dct.table_title.lower() else dct.sample2_name
    )
    
    conf_level_str = (
        int(dct.conf_level * 100) if dct.conf_level * 100 % 1 == 0 else dct.conf_level * 100
    )
    sig_level_str = (
        int(dct.sig_level * 100) if dct.sig_level * 100 % 1 == 0 else dct.sig_level * 100
    )

    if 'one' in dct.table_title.lower() or ('paired samples z test' in dct.table_title.lower() and dct.rho is None):
        sample1_name = sample1_name.replace('Sample 1', dct.sample1_name)
        if 'one' in dct.table_title.lower():
            table_results = (
                f"\t\t\t\t\t\\text{{Sample}} & \\text{{N}} & \\text{{Mean}} & \\text{{Pop. Mean}} & \\text{{Mean Diff.}} & \\text{{SD}} & \\text{{SE}} & {conf_level_str}\% \\, \\text{{LCI}} & {conf_level_str}\% \\, \\text{{UCI}} \\\\[5pt] \\hline \n"
                f"\t\t\t\t\t\\text{{{sample1_name.replace('_', '')}}} & {dct.n1} & {dct.mean1} & {dct.pop_mean} & {dct.mean_diff} & {dct.std1} & {dct.sem1} & {dct.LCI1} & {dct.UCI1} \\\\ \\hline \n"
            )
        else:
            table_results = (
                f"\t\t\t\t\t\\text{{Sample}} & \\text{{N}} & \\text{{Mean Diff.}} & \\text{{SD}} & \\text{{SE}} & {conf_level_str}\% \\, \\text{{LCI}} & {conf_level_str}\% \\, \\text{{UCI}} \\\\[5pt] \\hline \n"
                f"\t\t\t\t\t\\text{{{sample1_name.replace('_', ' ')}}} & {dct.n1} & {dct.mean12_diff} & {dct.std12_diff} & {dct.sem12_diff} & {dct.LCI12_diff} & {dct.UCI12_diff} \\\\ \\hline \n"
            )
        paired_null_hypothesis = f"\\mu_\\text{{{sample1_name.replace(' ', '')}}}"
        paired_alt_hypothesis = f"\\mu_\\text{{{sample2_name.replace(' ', '')}}}"
        result_table = (
            f'\n\t\t\t<div style="margin-bottom:3px;"> \n'
            f"\t\t\t\t\({{\\color{{{color_bluish()}}}{{\\text{{{dct.table_title}}}}}}}\) \n"
            f"\t\t\t</div>\n\n"
            f"\t\t\t<div> \n"
            f"\t\t\t\t\( \n"
            f"\t\t\t\t\\begin{{array}}{{l|rrrrrr}} \\hline \n"
            
            f"{table_results}"
            
            f"\t\t\t\t\\end{{array}} \n"
            f"\t\t\t\t\) \n"
            f"\t\t\t</div> \n"
            
            f"\n\t\t\t<div> \n"
            f"\t\t\t\t\( \n"
            f"\t\t\t\t\\begin{{array}}{{lrcr}} \n"
            f"\t\t\t\t\t\\text{{Hypothesis}} & \\hspace{{2.5cm}} \\text{{Description}} & \\, & \\text{{statsistic}} \\\\[2pt] \\hline \n"
            f"\t\t\t\t\t\\text{{H}}_{{0}} : {paired_null_hypothesis} = {pop_mean if 'one' in dct.table_title.lower() else paired_alt_hypothesis} & \\text{{{test_name}-calculated}} & = & {dct.test_stat} \\\\[2pt] \n"
            f"\t\t\t\t\t\\text{{H}}_{{1}} : {paired_null_hypothesis} {dct.hyp_sign} {pop_mean if 'one' in dct.table_title.lower() else paired_alt_hypothesis} & \\text{{{dfn_name}}} & {dfn_equals} & {dfn_str} \\\\[2pt] \n"
            f"\t\t\t\t\t\\text{{Conf. level}} = {dct.conf_level} \\, ({conf_level_str}\%) & \\text{{{test_name}-critical}} & = & {dct.crit_value} \\\\[2pt] \n"
            f"\t\t\t\t\t\\text{{Sig. level}} \\, (\\alpha) = {dct.sig_level} \\, ({sig_level_str}\%) & \\text{{p-value}} & = & {dct.p_value} \\\\ \\hline \n"
            f"\t\t\t\t\\end{{array}} \n"
            f"\t\t\t\t\) \n"
            f"\t\t\t</div> \n"

            f'\n\t\t\t<div style="margin-top:15px">\n\t\t\t\t\({{\\color{{{color_bluish()}}}{{\\text{{Decision}}}}}}\)\n\t\t\t</div>'
            f'\n\n\t\t\t<div style="margin-top:10px; margin-bottom:10px;">\n\t\t\t\t{dct.decision}\n\t\t\t</div>'
            f'\n\n\t\t\t<div>\n\t\t\t\t\({{\\color{{{color_bluish()}}}{{\\text{{Conclusion}}}}}}\)\n\t\t\t</div>'
            f'\n\n\t\t\t<div style="margin-top:10px;">\n\t\t\t\t{dct.conclusion}\n\t\t\t</div> \n'
        )
    else:
        sample1_name = sample1_name.replace('Sample ', 'Sample')
        sample2_name = sample2_name.replace('Sample ', 'Sample')
        if 'paired' in dct.table_title.lower():
            comb_row = ""
            diff_row = f"\\text{{Difference}} & {dct.n12_diff} & {dct.mean12_diff} & {dct.std12_diff} & {dct.sem12_diff} & {dct.LCI12_diff} & {dct.UCI12_diff} \\\\ \\hline"
        else:
            comb_row = f"\\text{{Overall}} & {dct.n12} & {dct.mean12} & {dct.std12} & {dct.sem12} & {dct.LCI12} & {dct.UCI12} \\\\ \\hline"
            diff_row = f"\\text{{Difference}} & \\, & {dct.mean12_diff} & \\, & {dct.sem12_diff} & {dct.LCI12_diff} & {dct.UCI12_diff} \\\\ \\hline"
        result_table = (
            f'\n\t\t\t<div style="margin-bottom:3px;"> \n'
            f"\t\t\t\t\({{\\color{{{color_bluish()}}}{{\\text{{{dct.table_title}}}}}}}\) \n"
            f"\t\t\t</div>\n\n"
            f"\t\t\t<div> \n"
            f"\t\t\t\t\( \n"
            f"\t\t\t\t\\begin{{array}}{{l|rrrrrr}} \\hline \n"
            f"\t\t\t\t\t\\text{{Samples}} & \\text{{N}} & \\text{{Mean}} & \\text{{SD}} & \\text{{SE}} & {conf_level_str}\% \\, \\text{{LCI}} & {conf_level_str}\% \\, \\text{{UCI}} \\\\[2pt] \\hline \n"
            f"\t\t\t\t\t\\text{{{sample1_name.replace('_', ' ')}}} & {dct.n1} & {dct.mean1} & {dct.std1} & {dct.sem1} & {dct.LCI1} & {dct.UCI1} \\\\[2pt] \n"
            f"\t\t\t\t\t\\text{{{sample2_name.replace('_', ' ')}}} & {dct.n2} & {dct.mean2} & {dct.std2} & {dct.sem2} & {dct.LCI2} & {dct.UCI2} \\\\[2pt] \\hline \n"
            f"\t\t\t\t\t{comb_row}"
            f"\t\t\t\t\t{diff_row}"
            f"\n\t\t\t\t\\end{{array}} \n"
            f"\t\t\t\t\) \n"
            f"\t\t\t</div> \n"

            f"\n\t\t\t<div> \n"
            f"\t\t\t\t\( \n"
            f"\t\t\t\t\\begin{{array}}{{lrcr}} \n"
            f"\t\t\t\t\t\\text{{Hypothesis}} & \\hspace{{2.5cm}} \\text{{Description}} & \\, & \\text{{statsistic}} \\\\[2pt] \\hline \n"
            f"\t\t\t\t\t\\text{{H}}_{{0}} : \\mu_\\text{{{sample1_name}}} = \\mu_\\text{{{sample2_name}}} & \\text{{{test_name}-calculated}} & = & {dct.test_stat} \\\\[2pt] \n"
            f"\t\t\t\t\t\\text{{H}}_{{1}} : \\mu_\\text{{{sample1_name}}} {dct.hyp_sign} \\mu_\\text{{{sample2_name}}} & \\text{{{dfn_name}}} & {dfn_equals} & {dfn_str} \\\\[2pt] \n"
            f"\t\t\t\t\t\\text{{Conf. level}} = {dct.conf_level} \\, ({conf_level_str}\%) & \\text{{{test_name}-critical}} & = & {dct.crit_value} \\\\[2pt] \n"
            f"\t\t\t\t\t\\text{{Sig. level}} \\, (\\alpha) = {dct.sig_level} \\, ({sig_level_str}\%) & \\text{{p-value}} & = & {dct.p_value} \\\\ \\hline \n"
            f"\t\t\t\t\\end{{array}} \n"
            f"\t\t\t\t\) \n"
            f"\t\t\t</div> \n"

            f'\n\t\t\t<div style="margin-top:15px">\n\t\t\t\t\({{\\color{{{color_bluish()}}}{{\\text{{Decision}}}}}}\)\n\t\t\t</div>'
            f'\n\n\t\t\t<div style="margin-top:10px; margin-bottom:10px;">\n\t\t\t\t{dct.decision}\n\t\t\t</div>'
            f'\n\n\t\t\t<div>\n\t\t\t\t\({{\\color{{{color_bluish()}}}{{\\text{{Conclusion}}}}}}\)\n\t\t\t</div>'
            f'\n\n\t\t\t<div style="margin-top:10px;">\n\t\t\t\t{dct.conclusion}\n\t\t\t</div> \n'
        )

    return str_remove_trailing_zeros(result_table)