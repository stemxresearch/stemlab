from typing import Literal

from sympy import sympify, latex
from pandas import IndexSlice, DataFrame

from stemlab.core.datetime import datetime_now


def fa_icon(fa: str = "&#8674", n: int = 1) -> str:
    """
    Generate HTML code for Font Awesome icons.

    Parameters
    ----------
    fa : str, optional (default="&#8674")
        The Font Awesome icon code or name. Default is a right arrow icon.
    n : int, optional (default=1)
        The number of times to repeat the icon.

    Returns
    -------
    str
        The HTML code for the Font Awesome icon(s).
    """
    return f'<span style="color:skyblue;font-size:25px;">{fa}</span>' * n


def hspace(n: int | float = 0.75) -> str:
    """
    Generate LaTeX code for horizontal space.

    Parameters
    ----------
    n : int or float, optional (default=0.75)
        The length of the horizontal space in centimeters.

    Returns
    -------
    str
        The LaTeX code for the horizontal space.
    """
    return f'\\hspace{{{n}cm}}'


def vspace(n: int | float = 0.75) -> str:
    """
    Generate LaTeX code for vertical space.

    Parameters
    ----------
    n : {int, float}, optional (default=0.75)
        The length of the vertical space in centimeters.

    Returns
    -------
    str
        The LaTeX code for the vertical space.
    """
    return f'\\vspace{{{n}cm}}'


def result_name_html(variable_name: str) -> str:
    """
    Generate HTML code for displaying a variable name.

    Parameters
    ----------
    variable_name : str
        The name of the variable to be displayed.

    Returns
    -------
    str
        The HTML code for displaying the variable name.
    """
    if not variable_name:
        variable_name = (
            '<strong style = "color:orange;">None.</strong> Results will not '
            'be added to your workspace.'
        )

    return variable_name


def steps_replace(html_steps: str) -> str:
    """
    Replace a specific CSS style in an HTML string.

    Parameters
    ----------
    html_steps : str
        The HTML string containing the CSS style to be replaced.

    Returns
    -------
    str
        The modified HTML string after replacing the CSS style.
    """
    html_steps = html_steps.replace(
        "background:url(/static/images/general/bgwhite.PNG);border-bottom:1px solid #ccc;",
        ";",
    )

    return html_steps


def display_input(
    dict_input: dict, 
    dollar_sign: str = "$", 
    decimal_points: int = -1
) -> str:
    """
    Generate LaTeX code for displaying input values in a formatted manner.

    Parameters
    ----------
    dict_input : dict
        A dictionary containing input values.
    dollar_sign : str, optional (default='$')
        The symbol to use for enclosing LaTeX expressions.
    decimal_points : int, optional (default=-1)
        The number of decimal points to round input values to.

    Returns
    -------
    str
        LaTeX code for displaying input values.

    Examples
    --------
    >>> display_input({'a': 10, 'b': 20})
    '<div style="margin-top:10px;">$ \\displaystyle \\begin{array}{lll} & {\\color{blue}{\\rightarrow}} & \\texttt{Input} a = \\displaystyle 10 \\[5pt] & {\\color{blue}{\\rightarrow}} & \\texttt{Input} b = \\displaystyle 20 \\[5pt]\\end{array} $</div>'
    """
    from stemlab.core.htmlatex import tex_to_latex
    from stemlab.core.output import objectname_display
    from stemlab.core.decimals import fround

    counter, latex_list = 0, []
    for dict_key, dict_value in dict_input.items():
        if isinstance(dict_value, str):
            latex_list.append(dict_value)
        else:
            try:
                if (float(sympify(dict_value)) or dict_value == 0.0):
                    dict_value = latex(sympify(fround(dict_value, decimal_points)))
            except Exception:
                pass
            object_name = objectname_display(dict_key, dict_value)
            dict_name_value = (
                f"& {{\\color{{blue}}{{\\rightarrow}}}} & {object_name} = "
                f"\\displaystyle {tex_to_latex(fround(dict_value, decimal_points))} \\\\[5pt]"
            )
            if counter == 0:
                kth_latex = (
                    f"\\hspace{{-0.05cm}}\\texttt{{Input}} {dict_name_value}"
                )
            else:
                kth_latex = dict_name_value
            latex_list.append(kth_latex)
        counter += 1
    latex_list = " ".join(map(str, latex_list))
    
    if counter > 1:
        latex_list = latex_list.replace("Input", "Inputs")
    if '<table' not in latex_list:
        latex_list = (
            f"{dollar_sign} \\displaystyle \\begin{{array}}{{lll}}{latex_list}"
            f"\\end{{array}} {dollar_sign}"
        )
    else:
        latex_list = latex_list.replace('<td>', '<td style="text-align:right;">')
    
    return f'<div style="margin-top:10px;">{latex_list}</div>'


def display_result(
    dict_output: dict, dollar_sign: str = "$", decimal_points: int = 12
) -> str:
    """
    Generate LaTeX code for displaying result values in a formatted manner.

    Parameters
    ----------
    dict_output : dict
        A dictionary containing result values.
    dollar_sign : str, optional (default='$')
        The symbol to use for enclosing LaTeX expressions.
    decimal_points : int, optional (default=12)
        The number of decimal points to round result values to.

    Returns
    -------
    str
        LaTeX code for displaying result values.

    Examples
    --------
    >>> display_result({'x': 10, 'y': 20})
    '$ \\displaystyle\\hspace{-0.05cm} {\\color{#01B3D1}{x = 10}} \\, {\\color{red}{\\checkmark\\checkmark}} $ $ \\displaystyle\\hspace{-0.05cm} {\\color{#01B3D1}{y = 20}} \\, {\\color{red}{\\checkmark\\checkmark}} $'
    """
    from stemlab.core.htmlatex import tex_to_latex
    from stemlab.core.output import objectname_display
    from stemlab.core.decimals import fround

    latex_list = []
    for dict_key, dict_value in dict_output.items():
        if isinstance(dict_value, str):
            latex_list.append(fround(dict_value, decimal_points))
        else:
            try:
                if (float(sympify(dict_value)) or dict_value == 0.0):
                    dict_value = latex(sympify(fround(dict_value, decimal_points)))
            except Exception:
                pass
            object_name = objectname_display(dict_key, dict_value)
            kth_latex = (
                f"\\displaystyle\\hspace{{-0.05cm}} "
                f"{{\\color{{#01B3D1}}{{{object_name} = {tex_to_latex(dict_value)}}}}} \\, "
                f"{{\\color{{red}}{{\\checkmark\\checkmark}}}}"
            )
            latex_list.append(f"{dollar_sign} {kth_latex} {dollar_sign}")
    latex_list = " ".join(latex_list)

    return latex_list


def steps(content: str) -> str:
    """
    Wrap content in a <div> tag.

    Parameters
    ----------
    content : str
        The content to be wrapped.

    Returns
    -------
    str
        The content wrapped in a <div> tag.
    """
    return f"<div>{content}</div>"


def python_syntax(content: str) -> str:
    """
    Wrap content in a <div> tag.

    Parameters
    ----------
    content : str
        The content to be wrapped.

    Returns
    -------
    str
        The content wrapped in a <div> tag.
    """
    return f"<div>{content}</div>"


def html_border(
    loc: Literal['top', 'bottom', 'left', 'right'] = "top", 
    width: int = 1, 
    type: Literal['solid', 'dashed', 'dotted'] = "dashed", 
    color: str = "#ddd", 
    margin: str = "13px -5px"
) -> str:
    """
    Generate an HTML border.

    Parameters
    ----------
    loc : str, optional (default="top")
        The location of the border (top, bottom, left, right).
    width : int, optional (default=1)
        The width of the border in pixels.
    type : str, optional (default="dashed")
        The type of the border (solid, dashed, dotted).
    color : str, optional (default="#ddd")
        The color of the border in hexadecimal or named color format.
    margin : str, optional (default="13px -5px")
        The margin of the border in CSS format.

    Returns
    -------
    str
        The HTML code for the border.
    """
    border = f'<div style="border-{loc}:{width}px {type} {color};margin:{margin};"></div>'
    return border


def results(
    application: str, 
    dict_input: dict, 
    dict_result: dict, 
    array_display: bool = True, 
    decimal_points: int = -1
):
    """
    Display results.

    Parameters
    ----------
    application : str
        The name of the application or calculation.
    dict_input : dict
        A dictionary containing input variables and their values.
    dict_result : dict
        A dictionary containing result variables and their values.
    array_display : bool, optional (default=True)
        Whether to display input and result arrays as tables.
    decimal_points : int, optional (default=-1)
        The number of decimal points to round the values to.

    Returns
    -------
    str
        The HTML content to display inputs and results.

    Examples
    --------
    >>> dict_input = {'x': 3, 'y': 4}
    >>> dict_result = {'result': 7}
    >>> results('Example', dict_input, dict_result)
    """
    from stemlab.core.decimals import fround

    html_list = []
    # inputs
    if array_display:
        kth_dict_input = {}
        try:
            for dict_key, dict_value in dict_input.items():
                try:
                    dict_key = sympify(dict_key)
                except Exception:
                    pass
                try:
                    dict_value = fround(sympify(dict_value), decimal_points)
                except Exception:  # for inputs that can not be sympified (e.g. html, latex)
                    pass
                kth_dict_input.update({dict_key: dict_value})
            html_list.append(display_input(kth_dict_input))
        except Exception:  # for string input display
            html_list.append(dict_input)
        html_list.append(html_border())

        # result
        kth_dict_result = {}
        try:
            for dict_key, dict_value in dict_result.items():
                try:
                    dict_key = sympify(dict_key)
                except Exception:
                    pass
                try:
                    dict_value = fround(sympify(dict_value), decimal_points)
                except Exception:  # for inputs that can not be sympified (e.g. html, latex)
                    pass
                kth_dict_result.update({dict_key: dict_value})
            html_list.append(display_result(kth_dict_result))
        except Exception:  # for string input display
            html_list.append(dict_result)
        # add margin
        html_content = " ".join(map(str, html_list))
        html_content = results_tex_display_latex(
            html_content=html_content, title=f"Results: {application}"
        )
    else:
        html_content = results_tex_display_latex(
            html_content="NA", title=f"Results: {application}"
        )

    return html_content


def title_header(
    name: str, 
    background: str = "#E5EFFF", 
    margin_top: int = 0, 
    margin_bottom: int = 5
) -> str:
    """
    Generate a styled header with the specified name.

    Parameters
    ----------
    name : str
        The text to display in the header.
    background : str, optional (default="#E5EFFF")
        The background color of the header.
    margin_top : int, optional (default=0)
        The top margin of the header.
    margin_bottom : int, optional (default=5)
        The bottom margin of the header.

    Returns
    -------
    str
        The HTML code for the styled header.
    """
    # space on the left of heading
    n = 8
    title = f'<div style="background: {background};color:black;padding:1px;padding-left:{n}px;margin:{margin_top}px -5px {margin_bottom}px -5px;border-top:1px solid #ccc;border-bottom:1px solid #ccc;font-weight:600;">{name}</div>'

    return title


def tex_list_to_latex_string(html_latex: str, mt: int = 10):
    """
    Convert a list of LaTeX strings into a single LaTeX string with HTML 
    formatting.

    Parameters
    ----------
    html_latex : str
        The list of LaTeX strings to convert.
    mt : int, optional (default=10)
        The top margin of the resulting LaTeX string.

    Returns
    -------
    str
        The LaTeX string with HTML formatting.

    Notes
    -----
    This function takes a list of LaTeX strings html_latex and 
    combines them into a single string with HTML formatting. It adds 
    a top margin to each LaTeX expression, replaces '.0' with '0' for 
    numbers, and ensures that '\displaystyle' is added after '('.

    Examples
    --------
    >>> tex_list_to_latex_string(['\\(x^2\\)', '\\(y^3\\)', '\\(z^4\\)'])
    """
    from stemlab.core.base.strings import str_replace_dot_zero

    html_latex = '</div><div style="margin-top:10px;">'.join(map(str, html_latex))
    html_latex = f'<div style="margin-top:{mt}px;">{html_latex}</div>'
    html_latex = str_replace_dot_zero(html_latex)
    html_latex = html_latex.replace('\\(', '\\(\\displaystyle ')
    
    return html_latex


def results_tex_display_latex(
    html_content: str, 
    title: str = 'Results', 
    show_title: bool = True
):
    """
    Display LaTeX format.

    Parameters
    ----------
    html_content : str
        The LaTeX content to display.
    title : str, optional (default='Results')
        The title for the LaTeX content.
    show_title : bool, optional (default=True)
        Whether to display the title.

    Returns
    -------
    str
        The HTML content with LaTeX formatting.

    Examples
    --------
    >>> results_tex_display_latex('<div>\\(x^2\\)</div>')
    """
    # space on the right of heading
    n = 6
    title = f'{title}<span style="float:right;font-weight:normal;padding-right:{n}px;color:black;">{datetime_now()}</span>'
    
    html_content = f'<div style="margin-left:-5px;">{html_content}</div>'
    if show_title:
        html_content = f'{title_header(title)}{html_content}'

    return html_content


def df_css(
    df: DataFrame, 
    coefficients: list, 
    rows: list | None = None, 
    df_names: list | None = None
) -> DataFrame:
    """
    Apply CSS styling to a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to style.
    coefficients : list, or tuple
        The list of coefficients to use for styling.
    rows : list, or tuple or None, optional
        The list of row indices to style, by default None (all rows).
    df_names : list, or tuple or None, optional
        The list of column names to style, by default None (all columns).

    Returns
    -------
    DataFrame
        The styled DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df_css(df, coefficients=[0.5, 0.8], rows=[0, 2], df_names=['A', 'B'])
    """
    dfnames_original = df.columns
    dfnames_new = [f"C{k + 1}" for k in range(df.shape[1])]
    df.columns = dfnames_new

    if rows is None:
        rows = list(map(str, list(df.index)))

    if df_names is None:
        df_names = list(map(str, list(df.columns)))

    if df_names == -1:  # all except first column
        df_names = list(map(str, list(df.columns)))[1:]

    result = df.style.map(
        css_styling, coefs=coefficients, subset=IndexSlice[rows, df_names]
    )
    result.columns = dfnames_original

    return result

def css_styling(coef, coefs) -> str:
    """
    Apply CSS styling to a DataFrame cell based on coefficients.

    Parameters
    ----------
    coef : object
        The coefficient value of the DataFrame cell.
    coefs : list, or tuple
        The list of coefficients to style.

    Returns
    -------
    str
        The CSS styling for the DataFrame cell.

    References
    ----------
    [1] Pandas documentation on styling:
    https://pandas.pydata.org/pandas-docs/version/1.1/user_guide/style.html
    """
    bgcolor, fweight = ("orange", "600") if coef in coefs else ("", "")
    css = f"background-color: {bgcolor}; font-weight:{fweight};"

    return css


def margin_topbot(
    n: int = 10, location: Literal['top', 'bottom'] = "top"
) -> str:
    """
    Generate HTML code for adding margin to the top or bottom of an element.

    Parameters
    ----------
    n : int, optional (default=10)
        The size of the margin in pixels.
    location : str, optional (default='top')
        The location where the margin should be applied ('top' or 'bottom').

    Returns
    -------
    str
        HTML code for adding margin to the specified location.
    """
    return f'<div style = "margin-{location}:{n}px;"></div>'


def no_results(name: str) -> str:
    """
    Generate HTML code indicating that the application does not have 
    certain results.

    Parameters
    ----------
    name : str
        The name of the results that the application does not have.

    Returns
    -------
    str
        HTML code indicating that the application does not have the 
        specified results.
    """
    # Construct the HTML code indicating that the application does 
    # not have the specified results
    return (
        f'<div style="margin-bottom:15px;margin-top:3px;">'
        f'This application does not have {name.lower()}.</div>'
    )


def na_steps(appname: str = "this") -> str:
    """
    Generate a message indicating that step-by-step solutions are not 
    available for a specific application.

    Parameters
    ----------
    appname : str, optional (default='this')
        The name of the application for which step-by-step solutions 
        are not available, by default "this".

    Returns
    -------
    result : str
        A message indicating that step-by-step solutions are not 
        available for the specified application.
    """
    # Construct the message indicating that step-by-step solutions are 
    # not available for the specified application
    result = (
        f"We currently do not have step-by-step solutions for the "
        f"{appname} application. We hope to have them in the future."
    )

    return result

