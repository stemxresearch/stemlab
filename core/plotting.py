from io import BytesIO
import base64
from sympy import degree
from matplotlib.pyplot import (
    plot, figure, clf, close, title, xlabel, ylabel, legend, text, tight_layout, 
    savefig, scatter
)
from IPython.display import Image
from ..core.arraylike import conv_to_arraylike
from ..core.symbolic import sym_lambdify_expr

def image_byte_code(byte_code: str) -> str:
    """
    Extract the image byte code from a data URL.

    Parameters
    ----------
    byte_code : str
        The image code containing a data URL.

    Returns
    -------
    byte_code : str
        The extracted image byte code.

    Raises
    ------
    ValueError
        If the provided image code does not contain a valid data URL.
    """
    byte_code = byte_code.replace(
        '<img src = "data:image/png;base64, ', ""
    ).replace('">', "")

    return byte_code


def figure_encoded() -> Image:
    """
    Generate base64 encoded image for a plotted graph.

    Returns
    -------
    Image
        The encoded image to be displayed.
    """
    figure_file = BytesIO()
    savefig(figure_file, format='png')
    figure_file.seek(0)
    figure_code = base64.b64encode(figure_file.getvalue()).decode('ascii')
    figure_code = f'data:image/png;base64,{figure_code}'

    return Image(url=figure_code)


def plot_html64() -> str:
    """
    Generate HTML code containing a base64 encoded image of the 
    plotted graph.

    Returns
    -------
    str
        HTML code with the embedded image.
    """
    figure_file = BytesIO()
    savefig(figure_file, format='png')
    figure_file.seek(0)
    figure_code = base64.b64encode(figure_file.getvalue()).decode('ascii')
    figure_code = f'data:image/png;base64,{figure_code}'
    styles = 'border:1px solid #ccc;width:615px;margin-bottom:20px;'
    figure_code_html = f'<div style="{styles}"><img src="{figure_code}"></div>'

    return figure_code_html


def interpolation_plot(
    x, 
    y, 
    x0, 
    poly_approx_x0, 
    poly_linspace, 
    poly_approx_linspace, 
    poly_deriv, 
    poly_var, 
    method, 
    plot_x0, 
    diff_order, 
    plot_deriv, 
    decimals = 8
):
    """
    Plot figures for interpolation results.

    Parameters
    ----------
    x : array_like
        An array with the values of x.
    y : array_like
        An array with the values of y i.e. f(x).
    x0 : int, float
        The point at which the polynomial should be approximated.
    poly_approx_x0 : int, float
        The value of the approximated polynomial at x=0. 
    poly_linspace : array_like
        An array with n linearly spaced values between min(x) and max(x)
    poly_approx_linspace : array_like
        An array with values found by substituting `poly_linspace` into 
        poly_approx.
    poly_deriv : {symbolic, str}
        An expression for the nth derivative of the function fx.
    poly_var : str
        The string to be used as the unknown variable in the polynomial.
    method : {straight-line, ..., reciprocal}
        The interpolation method to be applied.
        ===============================================================  
        Method                          Description  
        ===============================================================  
        Unequally spaced  
        ----------------  
        straight-line .................  
        lagrange ......................  
        hermite .......................  
          
        Equally spaced data, backward/forward  
        -------------------------------------  
        newton backward ............... used when x0 is towards the end  
        newton forward ................ used when x0 is at the beginning  
        gauss-backward ................  
        gauss-forward .................  
          
        Equally spaced, central formulas  
        --------------------------------  
        newton-divided : used when x0 is almost at center  
        neville .......................  
        stirling ......................  
        bessel ........................  
        laplace-everett ...............  
          
        Splines  
        -------  
        linear-splines ................  
        quadratic-splines .............  
        natural-cubic-splines .........  
        clamped-cubic-splines .........  
        not-a-knot-splines ............  
          
        Least square methods  
        --------------------  
        linear-regression .............  
        polynomial ....................  
          
        Linearization  
        -------------  
        exponential ...................  
        power .........................  
        saturation ....................  
        reciprocal ....................  
        ===============================================================  
    plot_x0 : bool, optional (default=True)
        If `True`, `x0` will be plotted on the graph.
    diff_order : int
        Order of differentiation
    plot_deriv : bool, optional (default=False)
        If `True`, the derivative will be plotted on the graph.
    decimals : int, optional (defaul=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Examples
    --------
    No examples. This function is called internally by the interplation function.

    Returns
    -------
    html_code : str
        A html string with the image code.
    """
    interp_method = f'{method.capitalize()} interpolation'
    x = conv_to_arraylike(array_values=x, par_name='x')
    y = conv_to_arraylike(array_values=y, par_name='y')
    n = len(x)
    clf()
    close()
    figure(figsize = (7, 5))
    scatter(x, y, color = 'blue', marker = 'D')
    plot(
        poly_linspace, 
        poly_approx_linspace, color = 'blue', 
        label = f'$f~({poly_var})$'
    )
    if poly_deriv is not None:
        if plot_deriv and degree(poly_deriv) > 0:
            g = sym_lambdify_expr(
                fexpr=poly_deriv, is_univariate=True, par_name='poly_deriv'
            )
            poly_deriv_linspace = g(poly_linspace)
            deriv_order = "'" * diff_order
            plot(
                poly_linspace, 
                poly_deriv_linspace,
                color = 'orange', linestyle = 'dashed',
                label = f'$f~{deriv_order}~({poly_var})$'
            )
            legend(
                loc = 'best', 
                ncol = 2,
                prop = {'family': 'monospace', 'weight': 'normal', 'size': 12}
            )
    
    if (plot_x0 and x0 < x[n-1] and x0 > x[0] and 
        method not in ['not-a-knot-splines']
    ):
        try:
            plot(
                x0, 
                poly_approx_x0, 
                color = 'red', 
                marker = '*', 
                markersize = 12
            )
            x = round(float(x0), decimals)
            y = round(float(poly_approx_x0), decimals)
            text(
                x0, poly_approx_x0,
                f'  ({x}, {y})',
                fontsize = 12,
                bbox = dict(facecolor = 'gray', alpha = 0.075),
                horizontalalignment = 'left'
            )
        except:
            pass
    xlabel(poly_var)
    ylabel(f'$f~({poly_var})$')
    title(interp_method.capitalize().replace('-', ' '))
    tight_layout()
    figure_html = figure_encoded()
    
    return figure_html

def figure_encoded():
    """
    Generate base64 code for plotted graphs.
    """
    figure_file = BytesIO()
    savefig(figure_file, format = 'png')
    figure_file.seek(0)
    figure_code = base64.b64encode(figure_file.getvalue()).decode('ascii')
    figure_code = f'data:image/png;base64,{figure_code}'
    figure_html = Image(url = figure_code)

    return figure_html