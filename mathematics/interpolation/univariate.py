from typing import Literal
from math import factorial

from numpy import (
    array, nan, linspace, arange, full, repeat, hstack, vstack, diag,
    concatenate, float64, reshape, log, log10, exp, poly1d, polyfit, delete,
    isnan, asarray, asfarray, nan_to_num
)
from numpy.linalg import solve
from sympy import (
    sympify, symbols, linear_eq_to_matrix, Poly, Piecewise, degree
)
from scipy.interpolate import lagrange, CubicSpline
from sklearn import linear_model
model = linear_model.LinearRegression()
from matplotlib.pyplot import (
    plot, figure, clf, close, title, xlabel, ylabel, legend, text, 
    tight_layout, scatter
)

from stemlab.core.display import Result, display_results
from stemlab.core.arraylike import arr_table_blank_row, conv_to_arraylike
from stemlab.core.base.strings import (
    str_plus_minus, str_remove_trailing_zeros
)
from stemlab.core.base.constraints import max_rows
from stemlab.core.symbolic import (
    sym_remove_zeros, sym_lambdify_expr, is_symexpr
)
from stemlab.core.plotting import figure_encoded
from stemlab.core.decimals import fround
from stemlab.core.htmlatex import sta_dframe_color, tex_display_latex
from stemlab.mathematics.interpolation.core import interpolation_p_terms
from stemlab.core.validators.errors import (
    NumpifyError, MaxNError, RequiredError
)
from stemlab.core.validators.validate import ValidateArgs

INTERPOLATION_METHODS = [
    'straight-line', 'lagrange', 'hermite',
    'newton-backward', 'newton-forward',
    'gauss-backward', 'gauss-forward',
    'newton-divided', 'neville', 'stirling', 'bessel', 'laplace-everett',
    'linear-splines', 'quadratic-splines', 'natural-cubic-splines', 
    'clamped-cubic-splines', 'not-a-knot-splines', 
    'linear-regression', 'polynomial', 
    'exponential', 'power', 'saturation', 'reciprocal'
]


def other_columns(ncols, letter='C'):
    other_columns = [f'{letter}{k + 1}' for k in range(ncols - 2)]
        
    return other_columns


def _coef_nan(lst: list | tuple) -> list:
    coefficients = lst
    if any(isnan(coefficient) for coefficient in coefficients):
        raise ValueError(
            "The specified method with the given 'p0' value was unsuccessful. "
            "Try using a different interpolation method or specify "
            "a different value for 'p0'."
        )
    coefficients = nan_to_num(coefficients, nan=0)
    return coefficients


def find_nearest_index(lst, target):
    index = min(range(len(lst)), key=lambda i: abs(lst[i] - target))
    return index


def _columns(ncols=0, variable="x", letter='C', no_blanks=True):
    """Generate column names. ncols=0 is a placeholder"""
    no_blank_rows = "_no_blank_rows" if no_blanks else ""
    xy_columns = [f'{variable}', f'f({variable}){no_blank_rows}']
    ck_columns = [f'{letter}{k + 1}' for k in range(ncols)]
    xy_ck_col_names = {'xy': xy_columns, 'other_cols': ck_columns}
    
    return xy_ck_col_names


class UnivariateInterpolation:
    """
    Univariate interpolation superclass
    
    Attributes
    ----------
    ...

    Methods
    -------
    solve(self):
        Interpolate the data.

    advance(self):
        Advance solution.
    
    _create_polynomials(self)
        Generate polynomial and its derivative if requested

    _plot_results(self):
        Plot the results.
    """
    
    def __init__(
            self,
            x: list[int | float],
            y: list[int | float],
            x0: int | float,
            p0: int | None = None,
            method: Literal[
                'straight-line', 'lagrange', 'hermite',
                'newton-backward', 'newton-forward',
                'gauss-backward', 'gauss-forward',
                'newton-divided', 'neville', 'stirling', 'bessel', 'laplace-everett',
                'linear-splines', 'quadratic-splines', 'natural-cubic-splines', 
                'clamped-cubic-splines', 'not-a-knot-splines', 
                'linear-regression', 'polynomial', 
                'exponential', 'power', 'saturation', 'reciprocal'
            ] = 'newton-divided',
            expr_variable: str = 'x',
            yprime: list[int | float] = None,
            poly_order: int = 1,
            qs_constraint: int | float = 0,
            end_points: list[int | float] = None,
            exp_type: Literal['b*exp(ax)', 'b*10^ax', 'ab^x'] = 'b*exp(ax)',
            sat_type: Literal['ax/(x+b)', 'x/(ax+b)'] = 'ax/(x+b)',
            plot_x0: bool = False,
            diff_order: int = 0,
            plot_deriv: bool = False,
            truncate_terms: float = 1e-16,
            auto_display: bool = True,
            decimal_points: int = 12
        ) -> dict:
        """Initializes the ODE solver with provided parameters."""

        self.x = x
        self.y = y
        self.x0 = x0
        self.p0 = p0
        self.method = method
        self.expr_variable = expr_variable
        self.yprime = yprime
        self.poly_order = poly_order
        self.qs_constraint = qs_constraint
        self.end_points = end_points
        self.exp_type = exp_type
        self.sat_type = sat_type
        self.plot_x0 = plot_x0
        self.diff_order = diff_order
        self.plot_deriv = plot_deriv
        self.truncate_terms = truncate_terms
        self.auto_display = auto_display
        self.decimal_points = decimal_points

        # x
        self.x = conv_to_arraylike(
            array_values=self.x, to_ndarray=True, par_name='x'
        )
        self.x_values = self.x.copy()
        self.n = len(self.x)
        if self.n > max_rows():
            raise MaxNError(
                par_name='len(x)', user_input=len(self.x), maxn=max_rows()
            )
        ncols = 0 if len(self.x.shape) == 1 else self.x.shape[1]
        
        # y
        self.y = conv_to_arraylike(
            array_values=self.y, to_ndarray=True, par_name='y'
        )
        _ = ValidateArgs.check_len_equal(x=self.x, y=self.y, par_name=['x', 'y'])
        
        # x0
        self.x0 = ValidateArgs.check_numeric(
            par_name='x0', limits=[min(self.x), max(self.x)], user_input=self.x0
        )
        
        # p0
        if p0 is not None:
            self.p0 = ValidateArgs.check_numeric(
                par_name='p0', limits=[0, len(self.x) - 1], user_input=self.p0
            )

        # method
        self.method = ValidateArgs.check_member(
            par_name='method',
            valid_items=INTERPOLATION_METHODS,
            user_input=self.method.lower()
        )
        
        # expr_variable
        self.expr_variable = ValidateArgs.check_string(
            par_name='expr_variable', user_input=self.expr_variable, default='x'
        )
        
        if self.method == 'hermite':
            if self.yprime is None:
                raise RequiredError(
                    par_name='yprime', required_when='method=hermite'
                )
                
            self.yprime = conv_to_arraylike(
                    array_values=self.yprime, 
                    to_ndarray=True, 
                    par_name='yprime'
                )
            _ = ValidateArgs.check_len_equal(
                x=self.x, y=self.yprime, par_name=['x', 'yprime']
            )
        
        if self.method == 'polynomial':
            self.poly_order = ValidateArgs.check_numeric(
                user_input=self.poly_order, 
                limits=[1, 9], 
                boundary='inclusive', 
                par_name='poly_order'
            )

        if self.method == 'quadratic-splines':
            self.qs_constraint = ValidateArgs.check_numeric(
                user_input=self.qs_constraint, par_name='qs_constraint'
            )
            self.qs_constraint = f'a0-{self.qs_constraint}'

        if self.method == 'clamped-cubic-splines':
            if self.end_points is None:
                raise RequiredError(
                    par_name='end_points', 
                    required_when='method=clamped-cubic-splines'
                )
            self.end_points = conv_to_arraylike(
                array_values=self.end_points,
                n=2,
                to_ndarray=True,
                par_name='end_points'
            )
        
        if self.method == 'exponential':
            self.exp_type = ValidateArgs.check_string(
                par_name='exp_type', 
                user_input=self.exp_type, 
                default='b*exp(ax)'
            )
            # check method
            self.exp_type = ValidateArgs.check_member(
                par_name='exp_type', 
                valid_items=['b*exp(ax)', 'b*10^ax', 'ab^x'], 
                user_input=self.exp_type
            )

        if self.method == 'saturation':
            self.sat_type = ValidateArgs.check_string(
                par_name='sat_type', 
                default='x/(ax+b)',
                user_input=self.sat_type
            )
            # check method
            self.sat_type = ValidateArgs.check_member(
                par_name='sat_type', 
                valid_items=['x/(ax+b)', 'ax/(x+b)'], 
                user_input=self.sat_type
            )
        
        # plot_x0
        self.plot_x0 = ValidateArgs.check_boolean(user_input=self.plot_x0, default=False)
        
        # diff_order
        self.diff_order = ValidateArgs.check_numeric(
            par_name='diff_order', 
            limits=[0, 9], 
            is_positive=True,
            is_integer=True, 
            user_input=self.diff_order
        )
        
        # plot_deriv
        self.plot_deriv = ValidateArgs.check_boolean(
            user_input=self.plot_deriv, default=False
        )
        
        # truncate_terms
        self.truncate_terms = ValidateArgs.check_numeric(
            user_input=self.truncate_terms, 
            limits=[0, .1], 
            par_name='truncate_terms'
        )
        
        self.auto_display = ValidateArgs.check_boolean(
            user_input=self.auto_display, default=False
        )
        self.decimal_points = ValidateArgs.check_decimals(x=self.decimal_points)
        
        self.N = full(shape=(self.n, self.n + 1), fill_value=nan)
        if len(self.x.shape) == 1:
            self.N[:, 0] = self.x
        else:
            self.N[:, :ncols] = self.x
        self.N[:, ncols + 1] = self.y

        # variables to be updated in functions
        self.col_names = ''
        self.poly_approx = 'x'
        self.style_indices = []
        self.create_polynomials = True
        self.poly_deriv = None
        self.p_col = None
            
            
    def compute(self):
        """
        Interpolate data.
        
        Returns
        -------
        table : pandas.DataFrame
            A DataFrame with the tabulated results.
        dframe : pandas.Styler
            Above table with values used for calculations highlighted.
        f: {sympy.Expr, str}
            Generated polynomial, f(x).
        fx: {float, Float}
            Value of f(x) at `x0`.
        df: {sympy.Expr, str}
            Nth derivative of f(x).
        dfx: {float, Float}
            Value of df(x) at `x0.
        plot: Image
            HTML code for the plotted f(x) and its nth derivative if specified.
        """
        self.advance() # this will call the respective interpolation class
        self.poly_approx = str(self.poly_approx).replace('1.0 * ', '').replace('1.0*', '')
        polynomials_dict = self._create_polynomials()
        if self.method in ['neville']:
            # Neville interpolation only gives the final result
            if self.auto_display:
                display_results({
                    'Answer': polynomials_dict['fx'],
                    'decimal_points': self.decimal_points
                })
            result = Result(fx = polynomials_dict['fx'])
        else:
            table, dframe_styled = self._dframe_table()
            f = polynomials_dict['f']
            f_latex = polynomials_dict['f_latex']
            fx = polynomials_dict['fx']
            fx_latex = polynomials_dict['fx_latex']
            df = polynomials_dict['df']
            df_latex = polynomials_dict['df_latex']
            dfx = polynomials_dict['dfx']
            dfx_latex = polynomials_dict['dfx_latex']
            plot = self._plot_results()
            if self.auto_display:
                display_results({
                    'dframe_styled': dframe_styled,
                    'f': f_latex,
                    'fx': fx_latex,
                    'df': df_latex,
                    'dfx': dfx_latex,
                    'decimal_points': self.decimal_points
                })
                
            result = Result(
                table=table,
                table_styled=dframe_styled,
                f=f,
                fx=fx,
                df=df,
                dfx=dfx,
                plot=plot
            )
            
        return result


    def advance(self):  
        """
        Advance solution.

        Returns
        -------
        numpy.ndarray
            The updated solution after advancing.
        """
        raise NotImplementedError
    
    def _create_polynomials(self):
        """
        Generate polynomial and its derivative if requested
        """
        if self.create_polynomials:
            x = symbols('x')
            poly_approx = sympify(self.poly_approx).expand()
            # self.poly_approx_x0 is used in _plot_results(self) function
            self.poly_approx_x0 = poly_approx.subs(x, self.x0)
            
            # for display
            poly_approx_exact = poly_approx.subs(x, self.expr_variable)
            poly_approx_eqtn_value = poly_approx_exact.subs(
                self.expr_variable, self.x0
            )
            
            # polynomials
            try:
                # will crush if not univariate polynomial
                poly_approx_eqtn = sym_remove_zeros(
                    fexpr=poly_approx_exact, threshold=self.truncate_terms
                )
            except Exception:
                pass
            if 'linear-splines' in self.method:
                poly_degree_fx, poly_degree_dfx = 1, 0
            elif 'quadratic' in self.method:
                poly_degree_fx, poly_degree_dfx = 2, 1
            elif 'cubic' in self.method:
                poly_degree_fx, poly_degree_dfx = 3, 2
            else:
                poly_degree_fx = degree(poly_approx_eqtn)
            
            poly_approx_eqtn_fx = poly_approx_eqtn
            poly_approx_eqtn_fx_latex = tex_display_latex(
                lhs=[f'P{poly_degree_fx}{f"({self.expr_variable})" if poly_degree_fx > 0 else ""}'], 
                rhs=[fround(poly_approx_eqtn, self.decimal_points)], 
                auto_display=False
            )
            
            poly_approx_eqtn_value_x0 = poly_approx_eqtn_value
            poly_approx_eqtn_value_x0_latex = tex_display_latex(
                lhs=[f'P{poly_degree_fx}{f"({fround(self.x0, self.decimal_points)})" if poly_degree_fx > 0 else ""}'], 
                rhs=[fround(poly_approx_eqtn_value, self.decimal_points)], 
                auto_display=False
            )

            if self.diff_order != 0:
                self.poly_deriv = poly_approx.diff(x, self.diff_order)
                poly_deriv_x0 = self.poly_deriv.subs(x, self.x0)
                poly_deriv_eqtn = self.poly_deriv.subs(x, self.expr_variable)
                poly_deriv_eqtn_value = poly_deriv_x0
                
                try:
                    poly_deriv_eqtn = sym_remove_zeros(
                        fexpr=poly_deriv_eqtn, threshold=self.truncate_terms
                    )
                except Exception:
                    pass
                
                try: 
                    # if `poly_degree_dfx` is not a polynomial, 
                    # then degree() function will crush, so just `pass` in `except`
                    poly_degree_dfx = degree(poly_deriv_eqtn)
                    is_derivative = True
                except Exception:
                    is_derivative = False
                
                order = None 
                if self.method == 'linear-splines':
                    order = 1 - self.diff_order
                elif self.method == 'quadratic-splines':
                    order = 2 - self.diff_order
                elif self.method in ['natural-cubic-splines', 'clamped-cubic-splines']:
                    order = 3 - self.diff_order
                else:
                    pass
                
                if order is not None:
                    poly_degree_dfx = order if order > 0 else 0
                
                # only proceed if differentiation was successful
                if is_derivative:
                    poly_deriv_eqtn_dfx = poly_deriv_eqtn
                    poly_deriv_eqtn_dfx_latex = tex_display_latex(
                        lhs=[f'dP{poly_degree_dfx}{f"({self.expr_variable})" if poly_degree_dfx > 0 else ""}'], 
                        rhs=[fround(poly_deriv_eqtn, self.decimal_points)], 
                        auto_display=False
                    )
                    
                    poly_deriv_eqtn_value_x0 = poly_deriv_eqtn_value
                    poly_deriv_eqtn_value_x0_latex = tex_display_latex(
                        lhs=[f'dP{poly_degree_dfx}{f"({fround(self.x0, self.decimal_points)})" if poly_degree_dfx > 0 else ""}'], 
                        rhs=[fround(poly_deriv_eqtn_value, self.decimal_points)], 
                        auto_display=False
                    )
                else:
                    poly_deriv_eqtn_dfx = poly_deriv_eqtn
                    poly_deriv_eqtn_dfx_latex = poly_deriv_eqtn_dfx
                    poly_deriv_eqtn_value_x0 = poly_deriv_eqtn_value
                    poly_deriv_eqtn_value_x0_latex = poly_deriv_eqtn_value_x0
            else:
                msg_deriv = "Computation of derivative was unsuccessful"
                poly_deriv_eqtn_dfx = None if self.diff_order == 0 else msg_deriv
                poly_deriv_eqtn_dfx_latex = poly_deriv_eqtn_dfx
                poly_deriv_eqtn_value_x0 = poly_deriv_eqtn_dfx
                poly_deriv_eqtn_value_x0_latex = poly_deriv_eqtn_value_x0
        else:
            poly_approx_eqtn_value_x0 = self.poly_approx
            poly_approx_eqtn_value_x0_latex = tex_display_latex(
                lhs=[f'P{len(self.x) - 1}{f"{({fround(self.x0, self.decimal_points)})}" if len(self.x) - 1 > 0 else ""}'], 
                rhs=[fround(self.poly_approx, self.decimal_points)],  auto_display=False
            )
        
        is_knot = self.method == 'not-a-knot-splines'
        result_dict = {
            'f': poly_approx_eqtn_fx if not is_knot else None,
            'f_latex': poly_approx_eqtn_fx_latex if not is_knot else None,
            'fx': poly_approx_eqtn_value_x0 if not is_knot else None,
            'fx_latex': poly_approx_eqtn_value_x0_latex if not is_knot else None,
            'df': poly_deriv_eqtn_dfx if not is_knot else None,
            'df_latex': poly_deriv_eqtn_dfx_latex if not is_knot else None,
            'dfx': poly_deriv_eqtn_value_x0 if not is_knot else None,
            'dfx_latex': poly_deriv_eqtn_value_x0_latex if not is_knot else None
        }
            
        return result_dict
    
    
    def _dframe_table(self):
        
        dframe = arr_table_blank_row(
            data=self.N,
            to_ndarray=True,
            convert_pd=True,
            col_names=self.col_names
        )
        
        if self.p_col is not None:
            col_p = 'p'
                
            dframe.insert(1, col_p, arr_table_blank_row(self.p_col))
        
        style_indices = array(self.style_indices).T
        if self.p_col is not None:
            style_indices[:, 1] += 1
        if self.style_indices:
            dframe_styled = sta_dframe_color(
                dframe=dframe,
                style_indices=style_indices.tolist(),
                decimal_points=self.decimal_points,
            )
        else:
            dframe_styled = dframe

        return dframe, dframe_styled
    

    def _plot_results(self):
        
        if self.method in ['neville']:
            return f'No plot for {self.method}.'
        
        (
            x, y, x_values, x0, method, poly_approx, poly_var, poly_deriv, 
            plot_deriv, diff_order, plot_x0, poly_approx_x0, decimal_points
        ) = (
            self.x, self.y, self.x_values, self.x0, self.method, self.poly_approx, 
            self.expr_variable, self.poly_deriv, self.plot_deriv, self.diff_order,
            self.plot_x0, self.poly_approx_x0, self.decimal_points
        )
        
        poly_linspace = linspace(
            float(min(x_values)), float(max(x_values)), 100
        )
        if method in ['not-a-knot-splines']:
            cubic_spline = CubicSpline(x, y, bc_type='not-a-knot')
            poly_approx_linspace = cubic_spline(poly_linspace)
        else:
            f = sym_lambdify_expr(
                fexpr=poly_approx, is_univariate=True, par_name='poly_approx'
            )
            poly_approx_linspace = f(poly_linspace)
        interp_method = f'{method.capitalize()} interpolation'

        clf()
        close()
        figure(figsize=(7, 5))
        scatter(x, y, color='blue', marker='D')
        plot(
            poly_linspace, 
            poly_approx_linspace, 
            color='blue', 
            label=f'$f\\,({poly_var})$'
        )
        if poly_deriv is not None and plot_deriv:
            if is_symexpr(poly_deriv):
                g = sym_lambdify_expr(
                    fexpr=poly_deriv, 
                    is_univariate=True, 
                    par_name='poly_deriv'
                )
                poly_deriv_values = g(poly_linspace)
            else:
                poly_deriv_values = [poly_deriv] * len(poly_linspace)
            deriv_order = "'" * diff_order
                
            plot(
                poly_linspace, 
                poly_deriv_values,
                color='orange', 
                linestyle='dashed',
                label=f'$f\\,{deriv_order}\\,({poly_var})$'
            )
            legend(
                loc='best', 
                ncol=2,
                prop={'family': 'monospace', 'weight': 'normal', 'size': 12}
            )  
        
        is_plot_x0_method = method not in ['not-a-knot-splines']
        if plot_x0 and x0 < x[len(x) - 1] and x0 > x[0] and is_plot_x0_method:
            try:
                plot(
                    x0, 
                    poly_approx_x0, 
                    color='red', 
                    marker='*', 
                    markersize=12
                )
                x = fround(float(x0), decimal_points)
                y = fround(float(poly_approx_x0), decimal_points)
                xy_text = str_remove_trailing_zeros(f'   ({x}, {y})')
                text(
                    x0, 
                    poly_approx_x0,
                    xy_text,
                    fontsize=12,
                    bbox = dict(facecolor='gray', alpha=0.075),
                    horizontalalignment='left'
                )
            except Exception:
                pass
        xlabel(poly_var)
        ylabel(f'$f\\,({poly_var})$')
        title(interp_method.capitalize().replace('-', ' '))
        tight_layout()
        figure_html = figure_encoded()
        
        return figure_html


    def _get_css_indices(self, data):
        """
        Get indices for css styling of the results table.
        """
        try:
            N = asfarray(data)
        except Exception:
            raise NumpifyError(par_name='data')
        nrows, ncols = arr_table_blank_row(N).shape
        style_indices = []
        if self.method in ['hermite', 'newton-forward', 'newton-divided']:
            if self.method == 'hermite':
                ncols = 2 * ncols
            for row in range(nrows):
                for col in range(1, ncols):
                    if row == col - 1:
                        style_indices.append([row, col])
            if self.method == 'hermite':
                style_indices.append([nrows, ncols - 2])
        elif self.method in ['newton-backward']:
            for col in range(1, ncols):
                style_indices.append([nrows - col, col])
        else:
            pass
        style_indices = asarray(style_indices).T.tolist()

        return style_indices
    
    
    def _coefs_and_indices(self, method, N, p, n):
        """Documentations"""
        
        def _gauss_indices(start, end=50):
            appended_list = [
                item for i in range(start, end + 1) for item in (i, i)
            ]
            return appended_list

        coefficients = []
        m = N.shape[0]
        rows = _gauss_indices(start=p, end=n - 1)
        is_gauss_forward = False
        if method in ['gauss-backward', 'gauss-forward']:
            if method == 'gauss-forward':
                rows = rows[1:]  # This is important
                is_gauss_forward = True

            try:
                for index in range(n - 1):
                    coefficients.append(N[rows[index], index + 1])
            except Exception:
                raise IndexError(
                    "The specified method with the given 'p0' value was unsuccessful. "
                    "Try using a different interpolation method or specify a different value "
                    "for 'p0'."
                )
            except Exception as e:
                raise ValueError(
                    f'An error occurred while attempting to compute polynomial '
                    f'coefficients: {e}'
                )
            coefficients = _coef_nan(lst=coefficients)
            rows = [2 * p, 2 * p + (1 if is_gauss_forward else -1)] * m
        style_indices = array([rows[:m], list(range(1, m + 1))]).T.tolist()

        return coefficients, style_indices


    def _interpolation_terms(self, X0, p):
        """
        Interpolation terms.

        Parameters
        ----------
        N : DataFrame
            A DataFrame with the values.
        x : array_like
            An array_like with the values of `x`.
        x0 : float
            The initial value.
        method : str
            Method of interpolation.
        p : float
            The point
        
        Returns
        ------
        style_indices : list
            A 2D list with the array indices.
        poly_approx : str
            A string with the interpolated polynomial.
        """
        N, x, method = self.N, self.x,self.method
        n = N.shape[1]
        terms = interpolation_p_terms(method=method)
        if method == 'gauss-backward':
            coefficients, style_indices = self._coefs_and_indices(
                method='gauss-backward', N=N, p=p, n=n
            )
        elif method == 'gauss-forward':
            coefficients, style_indices = self._coefs_and_indices(
                method='gauss-forward', N=N, p=p, n=n
            )
        elif method == 'stirling':
            coefs_forward, style_indices_forward = self._coefs_and_indices(
                method='gauss-forward', N=N, p=p, n=n
            )
            coefs_backward, style_indices_backward = self._coefs_and_indices(
                method='gauss-backward', N=N, p=p, n=n
            )
                
            def construct_expression(p0, forward_backward, middle):
                
                forward_backward = array(forward_backward).T.tolist()
                forward_backward = [
                    f'({x} + {y}) / 2' for x, y in forward_backward
                ]
                forward_backward_single = [
                    item for pair in zip(forward_backward, middle) for item in pair
                ]
                terms = p0 + forward_backward_single
                
                return terms
            
            coefficients = construct_expression(
                p0=[coefs_forward[0]],
                forward_backward=[coefs_backward[1::2], coefs_forward[1::2]], 
                middle=coefs_backward[2::2]
            )
            coefficients = [
                fround(sympify(coefficient), self.decimal_points) for coefficient in coefficients
            ]
            style_indices = style_indices_forward + style_indices_backward
        elif method == 'bessel':
            coefs_forward, style_indices_forward = self._coefs_and_indices(
                method='gauss-forward', N=N, p=p, n=n
            )
            coefs_backward, style_indices_backward = self._coefs_and_indices(
                method='gauss-backward', N=N, p=p + 1, n=n # note the p + 1
            )
            
            def construct_expression(forward, backward, middle):
                terms = []
                for i in range(len(forward)):
                    terms += [f'({forward[i]} + {backward[i]}) / 2', middle[i]]
                
                return terms
            
            coefficients = construct_expression(
                forward=coefs_forward[0::2].tolist(), 
                backward=coefs_backward[0::2].tolist(), 
                middle=coefs_backward[1::2]
            )
            coefficients = [
                fround(sympify(coefficient), self.decimal_points) for coefficient in coefficients
            ]
            style_indices = style_indices_forward + style_indices_backward       
        elif method == 'laplace-everett':
            terms_q, terms_p = terms
            coefs_forward, style_indices_forward = self._coefs_and_indices(
                method='gauss-forward', N=N, p=p, n=n
            )
            coefs_backward, style_indices_backward = self._coefs_and_indices(
                method='gauss-backward', N=N, p=p + 1, n=n # note the p + 1
            )
            coefficients = coefs_forward[::2], coefs_backward[::2]
            style_indices = style_indices_forward + style_indices_backward
            # remove the coloring for middle row 
            # (between above and below coloring)
            style_indices = array(style_indices)[::2, :].tolist()
        else:
            pass
        
        if method == 'laplace-everett':
            coefs_q, coefs_p = coefficients
            m = len(coefs_q)
            terms_q, terms_p = terms_q[:m], terms_p[:m]
            terms_q = [f'{coefs_q[k]} * {terms_q[k]}' for k in range(m)]
            terms_p = [f'{coefs_p[k]} * {terms_p[k]}' for k in range(m)]
            poly_approx = ' + '.join(terms_q + terms_p)
        else:
            m = len(coefficients)
            terms = terms[:m + 1]
            terms = [f'{coefficients[k]} * {terms[k]}' for k in range(m)]
            poly_approx = ' + '.join(terms)

        p = f'((x {str_plus_minus(x=-X0)} {abs(X0)}) / {x[1] - x[0]})'
        q = f'(1 - {p})'
        # q applies to laplace-everett only
        poly_approx = poly_approx.replace('p', p).replace('q', q)
        style_indices = array(style_indices).T.tolist()

        return style_indices, poly_approx


    def _interpolation_p(self):
        """
        Calculates column indices for p.

        Parameters
        ----------
        None
        
        Returns
        -------
        X0: float
            The reference value.
        p : int
            The positional index of a value in self.x.
        """
        if self.p0 is None:
            p = find_nearest_index(lst=self.x, target=self.x0)
        else:
            p = self.p0

        return self.x[p], p


class StraightLine(UnivariateInterpolation):
    """Straight line interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        columns = _columns(variable=expr_variable, no_blanks=True)
        self.col_names = columns['xy']
        x, y = N[:, 0], N[:, 1]
        parameters = {'n': 2, 'to_ndarray': True, 'label': 'exactly'}
        x = conv_to_arraylike(array_values=x, **parameters, par_name='x')
        y = conv_to_arraylike(array_values=y, **parameters, par_name='y')
        m = (y[1] - y[0]) / (x[1] - x[0])
        c = y[0] - m * x[0]
        self.poly_approx = f'{m} * x + {c}'

        self.style_indices = []
        self.create_polynomials = True
    

class Lagrange(UnivariateInterpolation):
    """Lagrange interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        columns = _columns(variable=expr_variable, no_blanks=True)
        self.col_names = columns['xy']
        x, y = N[:, 0], N[:, 1]
        coefs = asfarray(lagrange(x, y))[::-1]
        xs = [f'x**{k}' for k in range(len(coefs) + 1)]
        coefs_xs = [f'{coefs[k]} * {xs[k]}' for k in range(len(coefs))]
        self.poly_approx = sympify(' + '.join(coefs_xs))
        
        self.style_indices = []
        self.create_polynomials = True


class Hermite(UnivariateInterpolation):
    """Hermite interpolation"""
    def advance(self):
        
        N, expr_variable, yp = self.N, self.expr_variable, self.yprime
        nrows, ncols = N.shape
        columns = _columns(
            ncols=2 * ncols - 3,
            variable=expr_variable, 
            letter='C', 
            no_blanks=False
        )
        self.col_names = columns['xy'] + columns['other_cols']
        x, y = N[:, 0], N[:, 1]
        z = repeat(a=nan, repeats=2 * nrows)
        Q = full(shape=(2 * nrows, 2 * nrows), fill_value = nan)
        for i in range(nrows):
            z[2 * i] = x[i]
            z[2 * i + 1] = x[i]
            Q[2 * i, 0] = y[i]
            Q[2 * i + 1, 0] = y[i]
            Q[2 * i + 1, 1] = yp[i]
            if i != 0:
                Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / (z[2 * i] - z[2 * i - 1])
        for i in range(2, 2 * nrows):
            for j in range(2, i + 1):
                Q[i, j] = (Q[i, j - 1] - Q[i - 1, j - 1]) / (z[i] - z[i - j])
        
        # prepare the polynomial
        diag_elements = diag(v=Q)
        Q11 = diag_elements[0]
        Qii = diag_elements[1:]
        n = len(Qii) + 1
        M = [f'(x - {z[0]})']
        for i in range(1, n):
            for j in range(1, i + 1):
                fx = f'(x - {z[j]})'
            M.append(f'{M[i - 1]} * {fx}')
        M[0] = f'(x - {z[0]})'
        M.insert(0, Q11)
        ff = [Q11]
        for item in range(1, n):
            ff.append(f'{diag_elements[item]} * {M[item]}')
        self.poly_approx = ' + '.join(map(str, ff)).replace('+ -', '- ')
        self.N = hstack(tup=(z.reshape(-1, 1), Q))
        
        self.style_indices = self._get_css_indices(data=N)
        self.create_polynomials = True
    

class NewtonBackward(UnivariateInterpolation):
    """Newton backward interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        n, ncols = N.shape
        columns = _columns(variable=expr_variable, no_blanks=False)
        other_cols = other_columns(ncols=ncols, letter='C')
        self.col_names = columns['xy'] + other_cols
        ff = str(N[n - 1, 1])
        fi = '* p'
        x = N[:, 0]
        for j in range(1, n):
            for k in range(j, n):
                N[k, j + 1] = N[k, j] - N[k - 1, j]
            fi = fi if j == 1 else f'{fi} * (p + {j - 1})'
            ff = f'{ff} {str_plus_minus(x=N[n - 1, j + 1])} {abs(N[n - 1, j + 1])} {fi} / {factorial(j)}'
        self.N = N
        p = f'((x {str_plus_minus(x=-x[n - 1])} {abs(x[n - 1])}) / {x[1] - x[0]})'
        self.poly_approx = ff.replace('p', p)
        
        self.style_indices = self._get_css_indices(data=N)
        self.create_polynomials = True
    

class NewtonForward(UnivariateInterpolation):
    """Newton forward interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        n, ncols = N.shape
        columns = _columns(variable=expr_variable, no_blanks=False)
        other_cols = other_columns(ncols=ncols, letter='C')
        self.col_names = columns['xy'] + other_cols
        ff = str(N[0, 1])
        fi = '* p'
        x = N[:, 0] 
        for j in range(1, n):
            for k in range(j, n):
                N[k, j + 1] = N[k, j] - N[k - 1, j]
            fi = fi if j == 1 else f'{fi} * (p - {j - 1})'
            ff = f'{ff} {str_plus_minus(x=N[j, j + 1])} {abs(N[j, j + 1])} {fi} / {factorial(j)}'
        self.N = N
        p = f'((x {str_plus_minus(x=-x[0])} {abs(x[0])}) / {x[1] - x[0]})'
        self.poly_approx = ff.replace('p', p)
        
        self.style_indices = self._get_css_indices(data=N)
        self.create_polynomials = True
    

class GaussBackward(UnivariateInterpolation):
    """Gauss backward interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        nrows, ncols = N.shape
        columns = _columns(variable=expr_variable, no_blanks=False)
        other_cols = other_columns(ncols=ncols, letter='C')
        self.col_names = columns['xy'] + other_cols
        for j in range(1, nrows):
            for k in range(j, nrows):
                N[k, j + 1] = N[k, j] - N[k - 1, j]
        self.N = N
        X0, p = self._interpolation_p()
        self.p_col = arange(nrows) - p
        self.style_indices, self.poly_approx = self._interpolation_terms(X0, p)
        self.create_polynomials = True
    

class GaussForward(UnivariateInterpolation):
    """Gauss forward interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        nrows, ncols = N.shape
        columns = _columns(variable=expr_variable, no_blanks=False)
        other_cols = other_columns(ncols=ncols, letter='C')
        self.col_names = columns['xy'] + other_cols
        for j in range(1, nrows):
            for k in range(j, nrows):
                N[k, j + 1] = N[k, j] - N[k - 1, j]
        self.N = N
        X0, p = self._interpolation_p()
        self.p_col = arange(nrows) - p
        self.style_indices, self.poly_approx = self._interpolation_terms(X0, p)
        self.create_polynomials = True
    

class Neville(UnivariateInterpolation):
    """Neville interpolation"""
    def advance(self):
        
        x0, N = self.x0, self.N  
        n, _ = N.shape
        self.col_names = ""
        x, y = N[:, 0], N[:, 1]
        Q = full(shape=(n, n - 1), fill_value = nan)
        Q = concatenate((y.reshape(-1, 1), Q), axis=1)
        for i in range(1, n):
            for j in range(1, i + 1):
                Q[i, j] = ((x0 - x[i-j]) * Q[i, j - 1] - (x0 - x[i]) * Q[i - 1, j - 1]) / (x[i] - x[i-j])
        self.N = concatenate((x.reshape(-1, 1), Q), axis=1)
        self.poly_approx = Q[n - 1, n - 1]
        
        self.style_indices = [[n - 1], [n]]
        self.create_polynomials = True
    

class NewtonDivided(UnivariateInterpolation):
    """Newton divided differences interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        n, ncols = N.shape
        columns = _columns(variable=expr_variable, no_blanks=False)
        other_cols = other_columns(ncols=ncols, letter='C')
        self.col_names = columns['xy'] + other_cols
        ff = str(N[0, 1])
        fi = ''
        x = N[:, 0]
        for j in range(1, n):
            for k in range(j, n):
                N[k, j + 1] = (N[k, j] - N[k - 1, j]) / (x[k] - x[k-j])
            fi = f'{fi} * (x {str_plus_minus(x=-x[j - 1])} {abs(x[j - 1])})'
            ff = f'{ff} {str_plus_minus(x=N[j, j + 1])} {abs(N[j, j + 1])} {fi}'
        self.N = N
        self.poly_approx = ff
        self.style_indices = self._get_css_indices(data=N)
        self.create_polynomials = True
    

class Stirling(UnivariateInterpolation):
    """Stirling interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        n, ncols = N.shape
        if n % 2 == 0:
            raise ValueError(
                f'Expected number of elements to be odd but got: {n}'
            )
        columns = _columns(
            ncols=ncols - 2,
            variable=expr_variable,
            letter='C',
            no_blanks=False
        )
        self.col_names = columns['xy'] + columns['other_cols']
        for j in range(1, n):
            for k in range(j, n):
                N[k, j + 1] = N[k, j] - N[k - 1, j]
        self.N = N
        X0, p = self._interpolation_p()
        self.p_col = arange(n) - p
        self.style_indices, self.poly_approx = self._interpolation_terms(X0, p)
        self.create_polynomials = True
    

class Bessel(UnivariateInterpolation):
    """Bessel interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        n, ncols = N.shape
        if n % 2 != 0:
            raise ValueError(
                f'Expected number of elements to be even but got: {n}'
            )
        columns = _columns(
            ncols=ncols - 2,
            variable=expr_variable,
            letter='C',
            no_blanks=False
        )
        self.col_names = columns['xy'] + columns['other_cols']
        for j in range(1, n):
            for k in range(j, n):
                N[k, j + 1] = N[k, j] - N[k - 1, j]
        self.N = N
        X0, p = self._interpolation_p()
        self.p_col = arange(n) - p
        self.style_indices, self.poly_approx = self._interpolation_terms(X0, p)
        self.create_polynomials = True
    

class LaplaceEverett(UnivariateInterpolation):
    """Laplace Everett interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        n, ncols = N.shape
        if len(self.x) % 2 != 0:
            raise ValueError(
                f'Expected number of elements to be even but got: {n}'
            )
        columns = _columns(
            ncols=ncols - 2,
            variable=expr_variable,
            letter='C',
            no_blanks=False
        )
        self.col_names = columns['xy'] + columns['other_cols']
        for j in range(1, n):
            for k in range(j, n):
                N[k, j + 1] = N[k, j] - N[k - 1, j]
        self.N = N
        X0, p = self._interpolation_p()
        self.p_col = arange(n) - p
        self.style_indices, self.poly_approx = self._interpolation_terms(X0, p)
        self.create_polynomials = True
    

class LinearSplines(UnivariateInterpolation):
    """Linear splines interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        columns = _columns(variable=expr_variable, no_blanks=True)
        self.col_names = columns['xy'] + columns['other_cols']
        equations = []
        x, y = (N[:, 0], N[:, 1])
        for i in range(len(x) - 1):
            fx = f'{y[i] / (x[i] - x[i + 1])} * (x - {x[i + 1]}) + {y[i + 1] / (x[i + 1] - x[i])} * (x - {x[i]})'
            equations.append(sympify(fx))
        x_mid_values = x[1:-1] # exclude the first and last
        x = symbols('x')
        piece_values = [(equations[k], x < x_mid_values[k]) for k in range(len(x_mid_values))] + [(equations[-1], True)]
        self.poly_approx = Piecewise(*piece_values)
        
        self.style_indices = []
        self.create_polynomials = True
    

class QuadraticSplines(UnivariateInterpolation):
    """Quadratic splines interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        eqtn_syms = [
            'a0', 'b0', 'c0',
            'a1', 'b1', 'c1', 
            'a2', 'b2', 'c2', 
            'a3', 'b3', 'c3', 
            'a4', 'b4', 'c4', 
            'a5', 'b5', 'c5', 
            'a6', 'b6', 'c6', 
            'a7', 'b7', 'c7', 
            'a8', 'b8', 'c8', 
            'a9', 'b9', 'c9', 
            'a10', 'b10', 'c10', 
            'a11', 'b11', 'c11', 
            'a12', 'b12', 'c12', 
            'a13', 'b13', 'c13', 
            'a14', 'b14', 'c14', 
            'a15', 'b15', 'c15'
        ] 
        n, ncols = N.shape
        columns = _columns(variable=expr_variable, no_blanks=True)
        self.col_names = columns['xy'] + ['a', 'b', 'c']
        equations = []
        x, y = N[:, 0], N[:, 1]
        for i in range(n - 1):
            # move rhs to the left
            fx = f'{x[i] ** 2} * a{i} + {x[i]} * b{i} + c{i} - {y[i]}'
            equations.append(sympify(fx))
            # move rhs to the left
            fx = f'{x[i + 1] ** 2} * a{i} + {x[i + 1]} * b{i} + c{i} - {y[i + 1]}'
            equations.append(sympify(fx))         
        for i in range(n - 2):
            # rhs is zero
            fx = f'{2 * x[i + 1]} * a{i} + b{i} - {2 * x[i + 1]} * a{i + 1} - b{i + 1}'
            equations.append(sympify(fx))
        equations.append(sympify(self.qs_constraint))
        # convert equations to matrix form
        # * 3 since they are three variables: a, b, c
        A, b = linear_eq_to_matrix(equations, sympify(eqtn_syms[:((n - 1) * 3)]))
        A, b = asfarray(A), asfarray(b)
        solution_Ab = reshape(solve(A, b), ((n - 1), 3))
        x_mid_values = x[1:-1] # exclude the first and last
        x = symbols('x')
        nrows, ncols = solution_Ab.shape
        sol_equations = []
        x_vars = [f'x**{n}' for n in range(ncols - 1, -1, -1)]
        for row in range(nrows):
            sol = [f'{solution_Ab[row, col]} * {x_vars[col]}' for col in range(ncols)]
            sol_equations.append(sympify(' + '.join(sol))) 
        piece_values = [
            (sym_remove_zeros(sol_equations[k], sym_remove_zeros), x < x_mid_values[k]) for k in range(len(x_mid_values))
        ] + [(sol_equations[-1], True)]
        self.poly_approx = Piecewise(*piece_values)
        na_1column = repeat(nan, ncols).reshape(1, -1)
        solution_Ab_na = vstack([na_1column, solution_Ab])
        self.N = hstack([N[:, :2], solution_Ab_na])
        
        self.style_indices = []
        self.create_polynomials = True
    

class NaturalCubicSplines(UnivariateInterpolation):
    """Natural cubic spline interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        n, ncols = N.shape
        columns = _columns(variable=expr_variable, no_blanks=True)
        self.col_names = columns['xy'] + ['b', 'c', 'd']
        x, y = N[:, 0], N[:, 1]
        num_points = asfarray(repeat(0, n))
        h = num_points.copy()
        for i in range(n - 1):
            h[i] = x[i + 1] - x[i]
                
        alpha = num_points.copy()
        for i in range(1, n - 1):
            alpha[i] = (3/h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1])   
        l = num_points.copy(); l[0] = 1
        u = num_points.copy(); u[0] = 0
        z = num_points.copy(); z[0] = 0
            
        for i in range(1, n - 1):
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * u[i - 1]
            u[i] = h[i]/l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1])/l[i]
            
        l[-1] = 1
        z[-1] = 0
            
        c = num_points.copy(); c[-1] = 0
        b = num_points.copy()
        d = num_points.copy()
        for j in range(n - 2, -1, -1):
            c[j] = z[j] - u[j] * c[j + 1]
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
            d[j] = (c[j + 1] - c[j]) / (3 * h[j])
            
        self.N = vstack([x, y, b, c, d]).T
        M = self.N[:-1, 1:]
        x_copy = x.copy()
        x_mid_values = x[1:-1]
        x = symbols('x')
        nrows, ncols = M.shape
        sol_equations = []
        for row in range(nrows):
            x_vars = [f'(x - {x_copy[row]})**{n}' for n in range(ncols)]
            fx_list = [f'{M[row, col]} * {x_vars[col]}' for col in range(ncols)]
            fx = ' + '.join(fx_list)
            sol_equations.append(sympify(fx)) 
        piece_values = [
            (sym_remove_zeros(sol_equations[k], sym_remove_zeros), x < x_mid_values[k])
            for k in range(len(x_mid_values))
        ] + [(sol_equations[-1], True)]
        self.poly_approx = Piecewise(*piece_values)

        self.style_indices = []
        self.create_polynomials = True
    

class ClampedCubicSplines(UnivariateInterpolation):
    """Clamped cubic spline interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        n, ncols = N.shape
        columns = _columns(variable=expr_variable, no_blanks=True)
        self.col_names = columns['xy'] + ['b', 'c', 'd']
        x, y = N[:, 0], N[:, 1]
        end_points = conv_to_arraylike(
            array_values=self.end_points,
            n=2,
            to_ndarray=True,
            par_name='end_points'
        )
        fp_0, fp_n = end_points
        num_points = repeat(0, n).astype(float64)
        h = num_points.copy()
        for i in range(n - 1):
            h[i] = x[i + 1] - x[i]
        alpha = num_points.copy()
        alpha[0] = 3 * (y[1] - y[0]) / h[0] - 3 * fp_0
        alpha[-1] = 3 * fp_n - 3 * (y[-1] - y[-2]) / h[-2]
        for i in range(1, n - 1):
            alpha[i] = (3/h[i]) * (y[i + 1] - y[i]) - (3/h[i - 1]) * (y[i] - y[i - 1])
        l = num_points.copy()
        l[0] = 2 * h[0]
        u = num_points.copy()
        u[0] = 0.5
        z = num_points.copy()
        z[0] = alpha[0]/l[0]
        for i in range(1, n - 1):
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * u[i - 1]
            u[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
        l[-1] = h[-2] * (2 - u[-2])
        z[-1] = (alpha[-1] - h[-2] * z[-2]) / l[-1] 
        c = num_points.copy()
        c[-1] = z[-1] 
        b, d = num_points.copy(), num_points.copy()
        for j in range(n - 2, -1, -1):
            c[j] = z[j] - u[j] * c[j + 1]
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
            d[j] = (c[j + 1] - c[j]) / (3 * h[j])  
        self.N = vstack([x, y, b, c, d]).T
        M = self.N[:-1, 1:]
        x_copy = x.copy()
        x_mid_values = x[1:-1]
        x = symbols('x')
        nrows, ncols = M.shape
        sol_equations = []
        for row in range(nrows):
            x_vars = [f'(x - {x_copy[row]})**{n}' for n in range(ncols)]
            fx_list = [f'{M[row, col]} * {x_vars[col]}' for col in range(ncols)]
            fx = ' + '.join(fx_list)
            sol_equations.append(sympify(fx))
        piece_values = [
            (sym_remove_zeros(sol_equations[k], sym_remove_zeros), x < x_mid_values[k]) for k in range(len(x_mid_values))
        ] + [(sol_equations[-1], True)]
        self.poly_approx = Piecewise(*piece_values)
            
        self.style_indices = []
        self.create_polynomials = True
        

class NotAKnotSplines(UnivariateInterpolation):
    """Not-a-knot spline interpolation"""
    def advance(self):
        
        self.N, expr_variable = self.N[:, :2], self.expr_variable
        columns = _columns(variable=expr_variable, no_blanks=True)
        self.col_names = columns['xy'] + columns['other_cols']
        self.poly_approx = 'NA'
        
        self.style_indices = []
        self.create_polynomials = True
    

class LinearRegression(UnivariateInterpolation):
    """Linear regression interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        # columns with all nan --> index = isnan(arr).any(axis=0) any 
        #  with nan
        index = isnan(N).all(axis=0)
        N = delete(N, index, axis=1) # delete the columns with all nan
        x, y = N[:, :N.shape[1] - 1], N[:, -1].reshape(-1, 1)

        columns = _columns(variable=expr_variable, no_blanks=True)
        self.col_names = columns['xy']

        model.fit(x, y)
        bi = model.intercept_.tolist() + model.coef_.flatten().tolist()
        xi = [f'x{n}' for n in range(len(bi))]
        bixi = ' + '.join([f'{bi[n]} * {xi[n]}' for n in range(len(bi))])
        bixi = bixi.replace(' ', '').replace('*x0', '')
        bixi = bixi.replace('x1', 'x') if x.shape[1] == 1 else bixi
        self.poly_approx = sympify(bixi)
        
        self.style_indices = []
        self.create_polynomials = True
    

class Polynomial(UnivariateInterpolation):
    """Polynomial interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        columns = _columns(variable=expr_variable, no_blanks=True)
        self.col_names = columns['xy']
        x, y = N[:, 0], N[:, 1]
        poly_order = ValidateArgs.check_numeric(
            user_input=self.poly_order, 
            to_float=False, 
            limits=[1, len(x) - 1], 
            boundary='inclusive', 
            par_name='poly_order'
        )
        array_coefs = poly1d(polyfit(x, y, abs(int(poly_order))))
        self.poly_approx = str(Poly(array_coefs, sympify('x')).as_expr())
        
        self.style_indices = []
        self.create_polynomials = True
    

class Exponential(UnivariateInterpolation):
    """Exponential splines interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        columns = _columns(variable=expr_variable, no_blanks=True)
        self.col_names = columns['xy']
        x, y = N[:, 0].reshape(-1, 1), N[:, 1].reshape(-1, 1)
        y = log(y) if self.exp_type in ['b*exp(ax)', 'ab^x'] else log10(y)
        model.fit(x, y)
        b0, b1 = model.intercept_[0], model.coef_[0, 0] # a
        if self.exp_type == 'b*exp(ax)': # y = b0 exp(b1*x) -->> ln y = ln b0 + b1*x
            bixi = f'{exp(b0)} * exp({b1} * x)'
        elif self.exp_type == 'b*10^ax': # y = b 10^(ax) -->> log10 y = log10 b0 + b1*x
            bixi = f'{10**b0} * 10**({b1} * x)'
        else:
            bixi = f'{exp(b0)} * ({exp(b1)})^x'
        self.poly_approx = sympify(bixi)
        
        self.style_indices = []
        self.create_polynomials = True
    

class Power(UnivariateInterpolation):
    """Power interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        columns = _columns(variable=expr_variable, no_blanks=True)
        self.col_names = columns['xy']
        x, y = log(N[:, 0].reshape(-1, 1)), log(N[:, 1].reshape(-1, 1))
        model.fit(x, y) # y = b^(ax) --> ln y = ln b0 + b1 ln x ,
        b0, b1 = exp(model.intercept_[0]), model.coef_[0, 0]
        bixi = f'{b0} * x ** ({b1})'
        self.poly_approx = sympify(bixi)
        
        self.style_indices = []
        self.create_polynomials = True
    

class Saturation(UnivariateInterpolation):
    """Saturation interpolation"""
    def advance(self):
        """
        References
        ----------
        https://web.engr.oregonstate.edu/~webbky/MAE4020_5020_files/Section%207%20Curve%20Fitting.pdf
        """
        N, expr_variable = self.N, self.expr_variable
        columns = _columns(variable=expr_variable, no_blanks=True)
        self.col_names = columns['xy']
        x, y = 1 / N[:, 0].reshape(-1, 1), 1 / N[:, 1].reshape(-1, 1)
        model.fit(x, y)
        b0, b1 = model.intercept_[0], model.coef_[0, 0]
        if self.sat_type == 'ax/(x+b)':
            bixi = f'{1 / b0} * x / (x + {b1 / b0})'
        else:
            bixi = f'x / ({b0} * x + {b1})'
        self.poly_approx = sympify(bixi)
        
        self.style_indices = []
        self.create_polynomials = True
    

class Reciprocal(UnivariateInterpolation):
    """Reciprocal interpolation"""
    def advance(self):
        
        N, expr_variable = self.N, self.expr_variable
        columns = _columns(variable=expr_variable, no_blanks=True)
        self.col_names = columns['xy']
        x, y = N[:, 0].reshape(-1, 1), 1 / N[:, 1].reshape(-1, 1)
        model.fit(x, y) # y = 1 / (ax + b) --> 1/y = b0 + b1 * x
        b0, b1 = model.intercept_[0], model.coef_[0, 0]
        bixi = f'{1} / ({b1} * x + {b0})'
        self.poly_approx = sympify(bixi)
        
        self.style_indices = []
        self.create_polynomials = True
    

LiteralMethods = Literal[
    'straight-line', 'lagrange', 'hermite',
    'newton-backward', 'newton-forward',
    'gauss-backward', 'gauss-forward',
    'newton-divided', 'neville', 'stirling', 'bessel', 'laplace-everett',
    'linear-splines', 'quadratic-splines', 'natural-cubic-splines', 
    'clamped-cubic-splines', 'not-a-knot-splines', 
    'linear-regression', 'polynomial', 
    'exponential', 'power', 'saturation', 'reciprocal'
]

def interp(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    yprime: list[int | float] = None,
    p0: int | None = None,
    method: LiteralMethods = 'newton-divided',
    expr_variable: str = 'x',
    poly_order: int = 1,
    qs_constraint: int | float = 0,
    end_points: list[int | float] = None,
    exp_type: Literal['b*exp(ax)', 'b*10^ax', 'ab^x'] = 'b*exp(ax)',
    sat_type: Literal['ax/(x+b)', 'x/(ax+b)'] = 'ax/(x+b)',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result | tuple:
    """
    Performs univariate interpolation using a specified interpolation 
    method.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    yprime : array_like
        Points of the first derivative, same length as `x`. Must be 
        provided if `method='hermite'`
    p0 : int, optional (default=None)
        The positional index of the reference value. This only applies 
        to the central formulas, and must be specified appropriately,
        otherwise, the syntax will crush.
    method : str, optional (default='newton-divided')
        Interpolation method to be implimented.
        =============================================================
        method                          Description  
        =============================================================
        Unequally spaced
        ----------------
        straight-line ................. Linear interpolation
        lagrange ...................... Lagrange interpolation
        hermite ....................... Hermite interpolation

        Equally spaced data, backward/forward
        -------------------------------------
        newton-backward ............... Newton backward differences
        newton-forward ................ Newton forward differences

        Equally spaced, central formulas
        --------------------------------
        newton-divided ................ Newton divided differences
        neville ....................... Neville interpolation
        gauss-backward ................ Gauss backward differences
        gauss-forward ................. Gauss forward differences
        stirling ...................... Stirling interpolation
        bessel ........................ Bessel interpolation
        laplace-everett ............... Laplace everett interpolation

        Splines
        -------
        linear-splines ................ Linear splines interpolation
        quadratic-splines ............. Quadratic splines 
        natural-cubic-splines ......... Natural cubic splines
        clamped-cubic-splines ......... Clamped cubic splines
        not-a-knot-splines ............ Not-a-knot splines

        Least square methods
        --------------------
        linear-regression ............. Linear regression
        polynomial .................... Polynomial regression

        Linearization
        -------------
        exponential ................... Exponential
        power ......................... Power
        saturation .................... Saturation
        reciprocal .................... Reciprocal
        =============================================================
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    poly_order : int
        Degree of the interpolating polynomial. Required for when 
        `method=polynomial`.
    qs_constraint : {int, float}
        Last equation for quadratic spline. E.g. when `qs_constraint=5`, 
        the last equation becomes `a0 - 5 = 0`.
    end_points : array_like
        Two endpoints for the clamped-cubic-splines.
    exp_type : {'b*exp(ax)', 'b*10^ax', 'ab^x'}, optional default='exp(ax)'
        Type of exponential function to be applied.
    sat_type : {'ax/(x+b)', 'x/(ax+b)'}, optional (default='ax/(x+b)')
        Type of saturation function to be applied.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.
    
    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
        
    Examples
    --------
    >>> import stemlab as stm
    
    Unequally spaced
    ----------------
    
    Linear interpolation
    --------------------
    
    >>> x = [2, 5]
    >>> y = [4, 1]
    >>> x0 = 2.75
    >>> result = stm.interp(x, y, x0, method='straight-line', plot_x0=True,
    ... diff_order=0, decimal_points=8)
         x  f(x)
    0  2.0   4.0
    1  5.0   1.0
                                               
    Pn(x) = 6 - x
    
    Pn(x0) = 3.25
    
    Lagrange
    --------
    
    >>> x = [2, 5]
    >>> y = [4, 1]
    >>> x0 = 2.75
    >>> result = stm.interp(x, y, x0, method='lagrange', plot_x0=True,
    ... diff_order=0, plot_deriv=False, decimal_points=8)
         x  f(x)
    0  2.0   4.0
    1  5.0   1.0
                                               
    Pn(x) = 6 - x
    
    Pn(x0) = 3.25
    
    >>> x = np.array([2, 2.75, 4])
    >>> y = 1 / x
    >>> x0 = 3
    >>> result = stm.interp(x, y, x0, method='lagrange', plot_x0=True,
    ... diff_order=1, decimal_points=8)
          x        f(x)
    1  2.00  0.50000000
    2  2.75  0.36363636
    3  4.00  0.25000000
                                               
    Pn(x) = 0.045454545*x ** 2 - 0.39772727*x + 1.1136364
    
    Pn(x0) = 0.32954545
                                                   
    dPn(x) = 0.090909091*x - 0.39772727

    dPn(x0) = -0.125
    
    >>> x = [1, 2, 3, 4, 5, 6]
    >>> y = [16, 18, 21, 17, 15, 12]
    >>> x0 = 3.5
    >>> result = stm.interp(x, y, x0, method='lagrange', plot_x0=True,
    ... diff_order=1, decimal_points=8)
         x  f(x)
    1  1.0  16.0
    2  2.0  18.0
    3  3.0  21.0
    4  4.0  17.0
    5  5.0  15.0
    6  6.0  12.0
                                               
    Pn(x) = -0.24166667*x ** 5 + 4.3333333*x ** 4 - 28.958333*x ** 3 + 87.666667*x ** 2 - 115.8*x + 69.0
    
    Pn(x0) = 19.371094
                                                   
    dPn(x) = -1.2083333*x ** 4 + 17.333333*x ** 3 - 86.875*x ** 2 + 175.33333*x - 115.8

    dPn(x0) = -4.5109375
    
    Hermite
    -------
    
    >>> x = [1.3, 1.6, 1.9]
    >>> y = [0.6200860, 0.4554022, 0.2818186]
    >>> yprime = [-0.5220232, -0.5698959, -0.5811571]
    >>> x0 = 1.5
    >>> result = stm.interp(x, y, yprime=yprime, x0=x0,
    ... method='hermite', plot_x0=True, diff_order=1, decimal_points=8)
          x       f(x)         C1          C2          C3          C4          C5
    1   1.3   0.620086                                                           
    2                  -0.5220232                                                
    3   1.3   0.620086            -0.08974267                                    
    4                   -0.548946              0.06636556                        
    5   1.6  0.4554022              -0.069833              0.00266667            
    6                  -0.5698959              0.06796556             -0.00277469
    7   1.6  0.4554022            -0.02905367              0.00100185            
    8                   -0.578612              0.06856667                        
    9   1.9  0.2818186            -0.00848367                                    
    10                 -0.5811571                                                
    11  1.9  0.2818186                                                           
                                               
    Pn(x) = -0.0027746914*x ** 5 + 0.02403179*x ** 4 - 0.01455608*x ** 3 - 0.23521617*x ** 2 - 0.0082292235*x + 1.0019441
    
    Pn(x0) = 0.5118277
                                                   
    dPn(x) = -0.013873457*x ** 4 + 0.09612716*x ** 3 - 0.043668241*x ** 2 - 0.47043234*x - 0.0082292235

    dPn(x0) = -0.55793648
    
    Equally spaced data, backward/forward
    -------------------------------------

    >>> x = [0, 0.78539816, 1.57079633, 2.35619449, 3.14159265]
    >>> y = [1, 0.91600365, 0.0872758 , 0.11687946, 0.66613092]
    >>> methods = ['newton-backward', 'newton-forward']
    >>> for method in methods:
    ...     print(f'\\nMETHOD: {method.upper()}\\n')
    ...     result = stm.interp(x, y, x0=1.15, method=method, poly_order=3,
    ...         diff_order=1, plot_x0=True, decimal_points=8)
    
    Equally spaced, central formulas
    --------------------------------
    
    >>> x = [1.0, 1.3, 1.6, 1.9, 2.2]
    >>> y = [0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623]
    >>> methods = ['newton-divided', 'gauss-backward', 'gauss-forward']
    >>> for method in methods:
    ...     print(f'\\nMETHOD: {method.upper()}\\n')
    ...     result = stm.interp(x, y, x0=1.5, p0=2, method=method,
    ...     poly_order=3, diff_order=1, plot_x0=True, decimal_points=8)

    stirling
    --------
    
    >>> x = [10, 11, 12, 13, 14]
    >>> y = [0.23967, 0.28060, 0.31788, 0.35209, 0.38368]
    >>> x0 = 12.2
    >>> result = stm.interp(x, y, x0, p0=2, method='stirling',
    ... plot_x0=True, diff_order=1, decimal_points=8)
          x    p     f(x)       C1       C2       C3       C4
    1  10.0 -2.0  0.23967                                    
    2                      0.04093                           
    3  11.0 -1.0   0.2806          -0.00365                  
    4                      0.03728           0.00058         
    5  12.0  0.0  0.31788          -0.00307          -0.00013
    6                      0.03421           0.00045         
    7  13.0  1.0  0.35209          -0.00262                  
    8                      0.03159                           
    9  14.0  2.0  0.38368                                    
                                               
    Pn(x) = -5.4166667e-6*x ** 4 + 0.00034583333*x ** 3 - 0.0092995833*x ** 2 + 0.14688917*x - 0.59093
    
    Pn(x0) = 0.32495133
                                                   
    dPn(x) = -2.1666667e-5*x ** 3 + 0.0010375*x ** 2 - 0.018599167*x + 0.14688917

    dPn(x0) = 0.03505746

    stirling
    --------
    
    >>> x = [0, 5, 10, 15, 20, 25, 30]
    >>> y = [0, 0.0875, 0.1763, 0.2679, 0.3640, 0.4663, 0.5774]
    >>> x0 = 16
    >>> result = stm.interp(x, y, x0, p0=3, method='stirling',
    ... plot_x0=True, diff_order=1, decimal_points=8)
           x    p    f(x)      C1      C2      C3      C4      C5      C6
    1    0.0 -3.0     0.0                                                
    2                      0.0875                                        
    3    5.0 -2.0  0.0875          0.0013                                
    4                      0.0888          0.0015                        
    5   10.0 -1.0  0.1763          0.0028          0.0002                
    6                      0.0916          0.0017         -0.0002        
    7   15.0  0.0  0.2679          0.0045             0.0          0.0011
    8                      0.0961          0.0017          0.0009        
    9   20.0  1.0   0.364          0.0062          0.0009                
    10                     0.1023          0.0026                        
    11  25.0  2.0  0.4663          0.0088                                
    12                     0.1111                                        
    13  30.0  3.0  0.5774                                                
                                               
    Pn(x) = 9.7777778e-11*x ** 6 - 7.8666667e-9*x ** 5 + 2.4777778e-7*x ** 4 - 1.6166667e-6*x ** 3 + 1.9744444e-5*x ** 2 + 0.017415333*x
    
    Pn(x0) = 0.28670805
                                                   
    dPn(x) = 5.8666667e-10 * x ** 5 - 3.9333333e-8*x ** 4 + 9.9111111e-7*x ** 3 - 4.85e-6*x ** 2 + 3.9488889e-5*x + 0.017415333

    dPn(x0) = 0.01890256

    bessel
    ------
    
    >>> x = [20, 24, 28, 32]
    >>> y = [24, 32, 35, 40]
    >>> x0 = 25
    >>> result = stm.interp(x, y, x0, p0=1, method='bessel',
    ... plot_x0=True, diff_order=1, decimal_points=8)
          x    p  f(x)   C1   C2   C3
    1  20.0 -1.0  24.0               
    2                   8.0          
    3  24.0  0.0  32.0      -5.0     
    4                   3.0       7.0
    5  28.0  1.0  35.0       2.0     
    6                   5.0          
    7  32.0  2.0  40.0                    
                                               
    Pn(x) = 0.018229167*x ** 3 - 1.46875*x ** 2 + 40.083333*x - 336
    
    Pn(x0) = 32.9453125
                                                   
    dPn(x) = 0.0546875*x ** 2 - 2.9375*x + 40.083333

    dPn(x0) = 0.82552083

    bessel
    ------
    
    >>> x = [25, 26, 27, 28, 29, 30]
    >>> y = [4000, 3846, 3704, 3571, 3448, 3333]
    >>> x0 = 27.4
    >>> result = stm.interp(x, y, x0, p0=2, method='bessel',
    ... plot_x0=True, diff_order=1, decimal_points=8)
           x    p    f(x)     C1    C2   C3   C4   C5
    1   25.0 -2.0  4000.0                            
    2                     -154.0                     
    3   26.0 -1.0  3846.0         12.0               
    4                     -142.0       -3.0          
    5   27.0  0.0  3704.0          9.0       4.0     
    6                     -133.0        1.0      -7.0
    7   28.0  1.0  3571.0         10.0      -3.0     
    8                     -123.0       -2.0          
    9   29.0  2.0  3448.0          8.0               
    10                    -115.0                     
    11  30.0  3.0  3333.0                            
                                               
    Pn(x) = -0.058333333*x ** 5 + 8.0416667*x ** 4 - 443.125*x ** 3 + 12204.958*x ** 2 - 168223.82*x + 933710
    
    Pn(x0) = 3649.678336
                                                   
    dPn(x) = -0.29166667*x ** 4 + 32.166667*x ** 3 - 1329.375*x ** 2 + 24409.917*x - 168223.82

    dPn(x0) = -134.0048
    
    laplace-everett
    ---------------
    
    >>> x = [310, 320, 330, 340, 350, 360]
    >>> y = [2.49136, 2.50515, 2.51851, 2.53148, 2.54407, 2.55630]
    >>> x0 = 337.5
    >>> result = stm.interp(x, y, x0, p0=2, method='laplace-everett',
    ... plot_x0=True, diff_order=1, decimal_points=8)
            x    p     f(x)       C1       C2       C3       C4       C5
    1   310.0 -2.0  2.49136                                             
    2                        0.01379                                    
    3   320.0 -1.0  2.50515          -0.00043                           
    4                        0.01336           0.00004                  
    5   330.0  0.0  2.51851          -0.00039          -0.00003         
    6                        0.01297           0.00001           0.00004
    7   340.0  1.0  2.53148          -0.00038           0.00001         
    8                        0.01259           0.00002                  
    9   350.0  2.0  2.54407          -0.00036                           
    10                       0.01223                                    
    11  360.0  3.0   2.5563                                             
                                               
    Pn(x) = 3.3333333e-12*x ** 5 - 5.625e-9*x ** 4 + 3.7975e-6*x ** 3 - 0.0012839875*x ** 2 + 0.21903372*x - 12.74421
    
    Pn(x0) = 2.52827338
                                                   
    dPn(x) = 1.6666667e-11*x ** 4 - 2.25e-8*x ** 3 + 1.13925e-5*x ** 2 - 0.002567975*x + 0.21903372

    dPn(x0) = 0.00128742

    laplace-everett
    ---------------
    
    >>> x = [20, 28, 36, 44]
    >>> y = [2854, 3162, 7088, 7984]
    >>> x0 = 30
    >>> result = stm.interp(x, y, x0, p0=1,
    ... method='laplace-everett', plot_x0=True, diff_order=1,
    ... decimal_points=8)
          x    p    f(x)      C1      C2      C3
    1  20.0 -1.0  2854.0                        
    2                      308.0                
    3  28.0  0.0  3162.0          3618.0        
    4                     3926.0         -6648.0
    5  36.0  1.0  7088.0         -3030.0        
    6                      896.0                
    7  44.0  2.0  7984.0                        
                                               
    Pn(x) = -2.1640625*x ** 3 + 210.04688*x ** 2 - 6269.625*x + 61540.25
    
    Pn(x0) = 4064.0
                                                   
    dPn(x) = -6.4921875*x ** 2 + 420.09375*x - 6269.625

    dPn(x0) = 490.21875
    
    Splines
    -------
    
    >>> x = [1, 2, 3, 4, 5, 6]
    >>> y = [16, 18, 21, 17, 15, 12]
    >>> x0 = 3.5
    >>> methods = ['linear-splines', 'quadratic-splines',
    ... 'natural-cubic-splines', 'clamped-cubic-splines',
    ... 'not-a-knot-splines']
    >>> for method in methods:
    ...     print(f'\\nMETHOD: {method.upper()}\\n')
    ...     result = stm.interp(x, y, x0, method=method, poly_order=3,
    ...         plot_x0=True, diff_order=1, end_points=[-11.21666667, -13.3],
    ...         decimal_points=8)
    
    Least square and linearization methods
    --------------------------------------
    
    >>> x = [100, 150, 200, 250, 300, 350]
    >>> y = [10.63, 13.03, 15.04, 16.81, 18.42, 19.90]
    >>> x0 = 160
    >>> methods = ['linear-regression', 'polynomial', 'exponential',
    ... 'power', 'saturation', 'reciprocal']
    >>> for method in methods:
    ...     print(f'\\nMETHOD: {method.upper()}\\n')
    ...     result = stm.interp(x, y, x0, method=method, poly_order=3,
    ...         plot_x0=True, diff_order=1, decimal_points=8)
    
    References
    ----------
    https://s3.us-west-2.amazonaws.com/public.gamelab.fun/dataset/position_salaries.csv
    https://s3.us-west-2.amazonaws.com/public.gamelab.fun/dataset/salary_data.csv
    
    https://atozmath.com/example/CONM/NumeInterPola.aspx?q=F&q1=E1
    https://www.lkouniv.ac.in/site/writereaddata/siteContent/202004032250571912siddharth_bhatt_engg_Interpolation.pdf
    """
    method_class_map = {
        'straight-line': StraightLine,
        'lagrange': Lagrange,
        'hermite': Hermite,
        'newton-backward': NewtonBackward,
        'newton-forward': NewtonForward,
        'gauss-backward': GaussBackward,
        'gauss-forward': GaussForward,
        'newton-divided': NewtonDivided,
        'neville': Neville,
        'stirling': Stirling,
        'bessel': Bessel,
        'laplace-everett': LaplaceEverett,
        'linear-splines': LinearSplines,
        'quadratic-splines': QuadraticSplines,
        'natural-cubic-splines': NaturalCubicSplines,
        'clamped-cubic-splines': ClampedCubicSplines,
        'not-a-knot-splines': NotAKnotSplines,
        'linear-regression': LinearRegression,
        'polynomial': Polynomial,
        'exponential': Exponential,
        'power': Power,
        'saturation': Saturation,
        'reciprocal': Reciprocal
    }
    
    # validation of `method` must be repeated here
    method = ValidateArgs.check_member(
        par_name='method', 
        valid_items=INTERPOLATION_METHODS, 
        user_input=method.lower()
    )
    
    compute_class = method_class_map.get(method)
    solver = compute_class(
        x,
        y,
        x0,
        p0,
        method,
        expr_variable,
        yprime,
        poly_order,
        qs_constraint,
        end_points,
        exp_type,
        sat_type,
        plot_x0,
        diff_order,
        plot_deriv,
        truncate_terms,
        auto_display,
        decimal_points
    )

    return solver.compute()


def interp_straight_line(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    plot_x0: bool = True,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the straight-line method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [2, 5]
    >>> y = [4, 1]
    >>> x0 = 2.75
    >>> result = stm.interp_straight_line(x, y, x0, plot_x0=True,
    ... diff_order=0, decimal_points=8)
         x  f(x)
    1  2.0   4.0
    2  5.0   1.0
                                               
    P1(x) = 6.0 - x
    
    P1(2.75) = 3.25
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        method='straight-line',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_lagrange(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    plot_x0: bool = True,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Lagrange method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = np.array([2, 2.75, 4])
    >>> y = 1 / x
    >>> x0 = 3
    >>> result = stm.interp_lagrange(x, y, x0, plot_x0=True,
    ... diff_order=1, decimal_points=8)
        x        f(x)
    1  2.00  0.50000000
    2  2.75  0.36363636
    3  4.00  0.25000000
                                               
    P2(x) = 0.045454545*x**2 - 0.39772727*x + 1.1136364
    
    P2(3) = 0.32954545
                                                   
    dP1(x) = 0.090909091*x - 0.39772727

    dP1(3) = -0.125
    
    >>> x = [1, 2, 3, 4, 5, 6]
    >>> y = [16, 18, 21, 17, 15, 12]
    >>> x0 = 3.5
    >>> result = stm.interp_lagrange(x, y, x0, plot_x0=True,
    ... diff_order=1, decimal_points=8)
         x  f(x)
    1  1.0  16.0
    2  2.0  18.0
    3  3.0  21.0
    4  4.0  17.0
    5  5.0  15.0
    6  6.0  12.0
                                               
    P5(x) = -0.24166666666666678*x ** 5 + 4.333333333333332*x ** 4 - 28.958333333333343*x ** 3 + 87.66666666666669*x ** 2 - 115.79999999999998*x + 69.0
    
    P5(3.5) = 19.371093749999680
                                                   
    dP5(x) = -1.2083333333333339*x ** 4 + 17.33333333333333*x ** 3 - 86.875000000000029*x ** 2 + 175.3333333333334*x - 115.79999999999998

    dP5(3.5) = -4.5109375000004803
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        expr_variable=expr_variable,
        method='lagrange',
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_hermite(
    x: list[int | float],
    y: list[int | float],
    yprime: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Hermite method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    yprime : array_like
        Points of the first derivative, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [1.3, 1.6, 1.9]
    >>> y = [0.6200860, 0.4554022, 0.2818186]
    >>> yprime = [-0.5220232, -0.5698959, -0.5811571]
    >>> x0 = 1.5
    >>> result = stm.interp_hermite(x, y, yprime, x0, plot_x0=True,
    ... diff_order=1, decimal_points=8)
        x       f(x)         C1          C2          C3          C4          C5
    1   1.3   0.620086                                                           
    2                  -0.5220232                                                
    3   1.3   0.620086            -0.08974267                                    
    4                   -0.548946              0.06636556                        
    5   1.6  0.4554022              -0.069833              0.00266667            
    6                  -0.5698959              0.06796556             -0.00277469
    7   1.6  0.4554022            -0.02905367              0.00100185            
    8                   -0.578612              0.06856667                        
    9   1.9  0.2818186            -0.00848367                                    
    10                 -0.5811571                                                
    11  1.9  0.2818186                                                           
                                               
    P5(x) = -0.0027746914*x**5 + 0.02403179*x**4 - 0.01455608*x**3 - 0.23521617*x**2 - 0.0082292235*x + 1.0019441
    
    P5(1.5) = 0.5118277
                                                   
    dP4(x) = -0.013873457*x**4 + 0.09612716*x**3 - 0.043668241*x**2 - 0.47043234*x - 0.0082292235

    dP4(1.5) = -0.55793648
    """
    result = interp(
        x=x,
        y=y,
        yprime=yprime,
        x0=x0,
        method='hermite',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_newton_backward(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Newton backward 
    differences method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [0, 0.78539816, 1.57079633, 2.35619449, 3.14159265]
    >>> y = [1, 0.91600365, 0.0872758 , 0.11687946, 0.66613092]
    >>> x0 = 1.15
    >>> result = stm.interp_newton_backward(x, y, x0, diff_order=1,
    ... plot_x0=True, decimal_points=8)
                x        f(x)          C1          C2          C3          C4
    1         0.0         1.0                                                
    2                         -0.08399635                                    
    3  0.78539816  0.91600365              -0.7447315                        
    4                         -0.82872785              1.60306301            
    5  1.57079633   0.0872758              0.85833151             -1.94174672
    6                          0.02960366             -0.33868371            
    7  2.35619449  0.11687946               0.5196478                        
    8                          0.54925146                                    
    9  3.14159265  0.66613092                                                
                                               
    P4(x) = -0.21262867*x**4 + 1.5534689*x**3 - 3.3458112*x**2 + 1.6656016*x + 0.99999998
    
    P4(1.15) = 0.48134976
                                                   
    dP3(x) = -0.85051468*x**3 + 4.6604068*x**2 - 6.6916224*x + 1.6656016

    dP3(1.15) = -1.1599026
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        method='newton-backward',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_newton_forward(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Newton forward 
    differences method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [0, 0.78539816, 1.57079633, 2.35619449, 3.14159265]
    >>> y = [1, 0.91600365, 0.0872758 , 0.11687946, 0.66613092]
    >>> x0 = 1.15
    >>> result = stm.interp_newton_forward(x, y, x0, diff_order=1,
    ... plot_x0=True, decimal_points=8)
                x        f(x)          C1          C2          C3          C4
    1         0.0         1.0                                                
    2                         -0.08399635                                    
    3  0.78539816  0.91600365              -0.7447315                        
    4                         -0.82872785              1.60306301            
    5  1.57079633   0.0872758              0.85833151             -1.94174672
    6                          0.02960366             -0.33868371            
    7  2.35619449  0.11687946               0.5196478                        
    8                          0.54925146                                    
    9  3.14159265  0.66613092                                                
                                               
    P4(x) = -0.21262867*x**4 + 1.5534689*x**3 - 3.3458111*x**2 + 1.6656016*x + 1.0
    
    P4(1.15) = 0.48134975
                                                   
    dP3(x) = -0.85051468*x**3 + 4.6604067*x**2 - 6.6916223*x + 1.6656016

    dP3(1.15) = -1.1599026
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        method='newton-forward',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_gauss_backward(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    p0: int | None = None,
    expr_variable: str = 'x',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Gauss backward 
    differences method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    p0 : int, optional (default=None)
        The positional index of the reference value. This only applies 
        to the central formulas, and must be specified appropriately,
        otherwise, the syntax will crush.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [1.0, 1.3, 1.6, 1.9, 2.2]
    >>> y = [0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623]
    >>> x0 = 1.5
    >>> result = stm.interp_gauss_backward(x, y, x0, diff_order=1,
    ... plot_x0=True, decimal_points=8)
         x    p       f(x)         C1         C2         C3         C4
    1  1.0 -2.0  0.7651977                                            
    2                      -0.1451117                                 
    3  1.3 -1.0   0.620086            -0.0195721                      
    4                      -0.1646838             0.0106723           
    5  1.6  0.0  0.4554022            -0.0088998             0.0003548
    6                      -0.1735836             0.0110271           
    7  1.9  1.0  0.2818186             0.0021273                      
    8                      -0.1714563                                 
    9  2.2  2.0  0.1103623                                                                                            
                                               
    P4(x) = 0.0018251029*x**4 + 0.055292798*x**3 - 0.34304661*x**2 + 0.073391348*x + 0.97773506
    
    P4(1.5) = 0.51181999
                                                   
    dP3(x) = 0.0073004115*x**3 + 0.1658784*x**2 - 0.68609321*x + 0.073391348
    
    dP3(1.5) = -0.55788319
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        p0=p0,
        method='gauss-backward',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_gauss_forward(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    p0: int | None = None,
    expr_variable: str = 'x',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Gauss forward 
    differences method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    p0 : int, optional (default=None)
        The positional index of the reference value. This only applies 
        to the central formulas, and must be specified appropriately,
        otherwise, the syntax will crush.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [1.0, 1.3, 1.6, 1.9, 2.2]
    >>> y = [0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623]
    >>> x0 = 1.5
    >>> result = stm.interp_gauss_forward(x, y, x0, diff_order=1,
    ... plot_x0=True, decimal_points=8)
         x    p       f(x)         C1         C2         C3         C4
    1  1.0 -2.0  0.7651977                                            
    2                      -0.1451117                                 
    3  1.3 -1.0   0.620086            -0.0195721                      
    4                      -0.1646838             0.0106723           
    5  1.6  0.0  0.4554022            -0.0088998             0.0003548
    6                      -0.1735836             0.0110271           
    7  1.9  1.0  0.2818186             0.0021273                      
    8                      -0.1714563                                 
    9  2.2  2.0  0.1103623                                                                                            
                                               
    P4(x) = 0.0018251029*x**4 + 0.055292798*x**3 - 0.34304661*x**2 + 0.073391348*x + 0.97773506
    
    P4(1.5) = 0.51181999
                                                   
    dP3(x) = 0.0073004115*x**3 + 0.1658784*x**2 - 0.68609321*x + 0.073391348
    
    dP3(1.5) = -0.55788319
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        p0=p0,
        method='gauss-forward',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_newton_divided(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Newton divided 
    differences method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [1.0, 1.3, 1.6, 1.9, 2.2]
    >>> y = [0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623]
    >>> x0 = 1.5
    >>> result = stm.interp_newton_divided(x, y, x0, diff_order=1,
    ... plot_x0=True, decimal_points=8)
         x       f(x)          C1          C2          C3         C4
    1  1.0  0.7651977                                               
    2                 -0.48370567                                   
    3  1.3   0.620086             -0.10873389                       
    4                   -0.548946               0.0658784           
    5  1.6  0.4554022             -0.04944333              0.0018251
    6                   -0.578612              0.06806852           
    7  1.9  0.2818186              0.01181833                       
    8                   -0.571521                                   
    9  2.2  0.1103623                                                                                                                                  
                                               
    P4(x) = 0.0018251029*x**4 + 0.055292798*x**3 - 0.34304661*x**2 + 0.073391348*x + 0.97773506
    
    P4(1.5) = 0.51181999
                                                   
    dP3(x) = 0.0073004115*x**3 + 0.1658784*x**2 - 0.68609321*x + 0.073391348
    
    dP3(1.5) = -0.55788319
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        method='newton-divided',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_neville(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Neville method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [0, 0.78539816, 1.57079633, 2.35619449, 3.14159265]
    >>> y = [1, 0.91600365, 0.0872758 , 0.11687946, 0.66613092]
    >>> x0 = 1.15
    >>> result = stm.interp_neville(x, y, x0, plot_x0=True,
    ... decimal_points=8)
    
    Answer = 0.48134976
    
    (Note: Neville has no table, f(x), f'(x) and f'(x0). It just give 
    the value of the interpolated function at x0, i.e. f(x0))
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        method='neville',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_stirling(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    p0: int | None = None,
    expr_variable: str = 'x',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Stirling method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    p0 : int, optional (default=None)
        The positional index of the reference value. This only applies 
        to the central formulas, and must be specified appropriately,
        otherwise, the syntax will crush.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [0, 0.78539816, 1.57079633, 2.35619449, 3.14159265]
    >>> y = [1, 0.91600365, 0.0872758 , 0.11687946, 0.66613092]
    >>> x0 = 1.15
    >>> result = stm.interp_stirling(x, y, x0, p0=2, diff_order=1,
    ... plot_x0=True, decimal_points=8)
                x        f(x)          C1          C2          C3          C4
    1         0.0         1.0                                                
    2                         -0.08399635                                    
    3  0.78539816  0.91600365              -0.7447315                        
    4                         -0.82872785              1.60306301            
    5  1.57079633   0.0872758              0.85833151             -1.94174672
    6                          0.02960366             -0.33868371            
    7  2.35619449  0.11687946               0.5196478                        
    8                          0.54925146                                    
    9  3.14159265  0.66613092                                                
                                               
    P4(x) = -0.21262867*x**4 + 1.5534689*x**3 - 3.3458112*x**2 + 1.6656017*x + 0.99999997
    
    P4(1.15) = 0.48134976
                                                   
    dP3(x) = -0.85051468*x**3 + 4.6604068*x**2 - 6.6916224*x + 1.6656017
    
    dP3(1.15) = -1.15990263
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        p0=p0,
        method='stirling',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_bessel(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    p0: int | None = None,
    expr_variable: str = 'x',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Bessel method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    p0 : int, optional (default=None)
        The positional index of the reference value. This only applies 
        to the central formulas, and must be specified appropriately,
        otherwise, the syntax will crush.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [0, 0.78539816, 1.57079633, 2.35619449]
    >>> y = [1, 0.91600365, 0.0872758 , 0.11687946]
    >>> x0 = 1.15
    >>> result = stm.interp_bessel(x, y, x0, diff_order=1,
    ... plot_x0=True, decimal_points=8)
                x    p        f(x)          C1          C2          C3
    0         0.0 -1.0         1.0                                    
    1                              -0.08399635                        
    2  0.78539816  0.0  0.91600365              -0.7447315            
    3                              -0.82872785              1.60306301
    4  1.57079633  1.0   0.0872758              0.85833151            
    5                               0.02960366                        
    6  2.35619449  2.0  0.11687946                                                               
                                               
    P3(x) = 0.55147992*x**3 - 1.9030506*x**2 + 1.0475244*x + 1.0
    
    P3(1.15) = 0.52660071
                                                   
    dP2(x) = 1.6544398*x**2 - 3.8061011*x + 1.0475244

    dP2(1.15) = -1.1414953
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        p0=p0,
        method='bessel',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_laplace_everett(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    p0: int | None = None,
    expr_variable: str = 'x',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Laplace-Everett method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    p0 : int, optional (default=None)
        The positional index of the reference value. This only applies 
        to the central formulas, and must be specified appropriately,
        otherwise, the syntax will crush.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [0, 0.78539816, 1.57079633, 2.35619449]
    >>> y = [1, 0.91600365, 0.0872758 , 0.11687946]
    >>> x0 = 1.15
    >>> result = stm.interp_laplace_everett(x, y, x0, diff_order=1,
    ... plot_x0=True, decimal_points=8)
                x    p        f(x)          C1          C2          C3
    0         0.0 -1.0         1.0                                    
    1                              -0.08399635                        
    2  0.78539816  0.0  0.91600365              -0.7447315            
    3                              -0.82872785              1.60306301
    4  1.57079633  1.0   0.0872758              0.85833151            
    5                               0.02960366                        
    6  2.35619449  2.0  0.11687946                                    
                                               
    P3(x) = 0.55147992*x**3 - 1.9030506*x**2 + 1.0475244*x + 1.0
    
    P3(x0) = 0.52660071
                                                   
    dP2(x) = 1.6544398*x**2 - 3.8061011*x + 1.0475244
    
    dP2(x0) = -1.1414953
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        p0=p0,
        method='laplace-everett',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_linear_splines(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Linear splines method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [1.0, 1.3, 1.6, 1.9, 2.2]
    >>> y = [0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623]
    >>> x0 = 1.5
    >>> result = stm.interp_linear_splines(x, y, x0, diff_order=1,
    ... plot_x0=True, decimal_points=8)
         x       f(x)
    0  1.0  0.7651977
    1  1.3  0.6200860
    2  1.6  0.4554022
    3  1.9  0.2818186
    4  2.2  0.1103623                                               
                                               
    P1(x) = Piecewise((1.2489034 - 0.48370567*x, x < 1.3), (1.3337158 - 0.548946*x, x < 1.6), (1.3811814 - 0.578612*x, x < 1.9), (1.3677085 - 0.571521*x, True))
    
    P1(1.5) = 0.51029680
                                                   
    dP0(x) = Piecewise((-0.48370567, x < 1.3), (-0.548946, x < 1.6), (-0.578612, x < 1.9), (-0.571521, True))
    
    dP0(1.5) = -0.54894600
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        method='linear-splines',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_quadratic_splines(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    qs_constraint: int | float = 0,
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Quadratic splines 
    method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    qs_constraint : {int, float}
        Last equation for quadratic spline. E.g. when `qs_constraint=5`, 
        the last equation becomes `a0 - 5 = 0`.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [0, 0.78539816, 1.57079633, 2.35619449, 3.14159265]
    >>> y = [1, 0.91600365, 0.0872758 , 0.11687946, 0.66613092]
    >>> x0 = 1.15
    >>> result = stm.interp_quadratic_splines(x, y, x0, diff_order=1,
    ... plot_x0=True, decimal_points=8)
                x        f(x)            a             b             c
    0  0.00000000  1.00000000                                         
    1  0.78539816  0.91600365  -0.00000000   -0.10694747    1.00000000
    2  1.57079633  0.08727580  -1.20731321    1.78949567    0.25526852
    3  2.35619449  0.11687946   2.59878785  -10.16772350    9.64644652
    4  3.14159265  0.66613092  -1.75636656   10.35545819  -14.53185729
                                               
    P2(x) = Piecewise((1.0 - 0.10694747*x, x < 0.78539816), (-1.2073132*x**2 + 1.7894957*x + 0.25526852, x < 1.57079633), (2.5987879*x**2 - 10.167723*x + 9.6464465, x < 2.35619449), (-1.7563666*x**2 + 10.355458*x - 14.531857, True))
    
    P2(1.15) = 0.71651682
                                                   
    dP1(x) = Piecewise((-1.7669748e-16*x - 0.10694747, x < 0.78539816), (1.7894957 - 2.4146264*x, x < 1.57079633), (5.1975757*x - 10.167723, x < 2.35619449), (10.355458 - 3.5127331*x, True))
    
    dP1(1.15) = -0.98732471
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        method='quadratic-splines',
        expr_variable=expr_variable,
        qs_constraint=qs_constraint,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_natural_cubic_splines(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Natural cubic splines  
    method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm
    >>> x = [0, 1, 2, 3]
    >>> y = [1, np.exp(1), np.exp(2), np.exp(3)] # i.e. y = exp(x)
    >>> x0 = 1.15
    >>> result = stm.interp_natural_cubic_splines(x, y, x0,
    ... diff_order=1, plot_x0=True, decimal_points=8)
         x         f(x)           b           c           d
    0  0.0   1.00000000  1.46599761  0.00000000  0.25228421
    1  1.0   2.71828183  2.22285026  0.75685264  1.69107137
    2  2.0   7.38905610  8.80976965  5.83006675 -1.94335558
    3  3.0  20.08553692  0.00000000  0.00000000  0.00000000
                                               
    P3(x) = Piecewise((0.25228421*x**3 + 1.4659976*x + 1.0, x < 1.0), (1.6910714*x**3 - 4.3163615*x**2 + 5.7823591*x - 0.43878716, x < 2.0), (-1.9433556*x**3 + 17.4902*x**2 - 37.830764*x + 28.636628, True))
    
    P3(1.15) = 3.07444592
                                                   
    dP2(x) = Piecewise((0.75685264*x**2 + 1.4659976, x < 1.0), (5.0732141*x**2 - 8.6327229*x + 5.7823591, x < 2.0), (-5.8300668*x**2 + 34.980401*x - 37.830764, True))

    dP2(1.15) = 2.56405337
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        method='natural-cubic-splines',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_clamped_cubic_splines(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    end_points: list[int | float] = None,
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Clamped cubic splines  
    method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    end_points : array_like
        Two endpoints for the clamped-cubic-splines.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import numpy as np
    >>> import stemlab as stm
    >>> x = [0, 1, 2, 3]
    >>> y = [1, np.exp(1), np.exp(2), np.exp(3)] # i.e. y = exp(x)
    >>> end_points = [1, np.exp(3)] # i.e. [f'(0), f'(3)]
    >>> x0 = 1.15
    >>> result = stm.interp_clamped_cubic_splines(x, y, x0,
    ... end_points=end_points, diff_order=1, plot_x0=True,
    ... decimal_points=8)
         x         f(x)           b           c           d
    0  0.0   1.00000000  1.00000000  0.44468250  0.27359933
    1  1.0   2.71828183  2.71016299  1.26548049  0.69513079
    2  2.0   7.38905610  7.32651634  3.35087286  2.01909162
    3  3.0  20.08553692  0.00000000  9.40814772  0.00000000
                                               
    P3(x) = Piecewise((0.27359933*x**3 + 0.4446825*x**2 + 1.0*x + 1.0, x < 1.0), (0.69513079*x**3 - 0.81991188*x**2 + 2.2645944*x + 0.57846854, x < 2.0), (2.0190916*x**3 - 8.7636768*x**2 + 18.152124*x - 10.013218, True))
    
    P3(1.15) = 3.1556257
                                                   
    dP2(x) = Piecewise((0.82079799*x**2 + 0.88936499*x + 1.0, x < 1.0), (2.0853924*x**2 - 1.6398238*x + 2.2645944, x < 2.0), (6.0572749*x**2 - 17.527354*x + 18.152124, True))

    dP2(1.15) = 3.1367285
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        method='clamped-cubic-splines',
        expr_variable=expr_variable,
        end_points=end_points,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_not_a_knot_splines(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Not-a-knot splines  
    method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [1, 2, 3, 4, 5, 6]
    >>> y = [16, 18, 21, 17, 15, 12]
    >>> x0 = 3.5
    >>> result = stm.interp_not_a_knot_splines(x, y, x0, plot_x0=True,
    ... decimal_points=8)
         x  f(x)
    0  1.0  16.0
    1  2.0  18.0
    2  3.0  21.0
    3  4.0  17.0
    4  5.0  15.0
    5  6.0  12.0
                                               
    (Note that `interp_not_a_knot_splines` does not generate any other 
    results except the graph)
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        method='not-a-knot-splines',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_linear_regression(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Linear regression  
    method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [4.1953, 4.8608, 4.8111, 4.1795, 3.6946, 4.5836, 3.7503,
    ... 5.5671, 5.8547, 3.5338]
    >>> y = [28.0014, 32.2369, 29.5074, 26.9072, 25.7026, 30.0759,
    ... 26.7065, 34.6266, 36.4237, 24.0014]
    >>> x0 = 4.25
    >>> result = stm.interp_linear_regression(x, y, x0, diff_order=1,
    ... plot_x0=True, decimal_points=8)
            x     f(x)
    0  4.1953  28.0014
    1  4.8608  32.2369
    2  4.8111  29.5074
    3  4.1795  26.9072
    4  3.6946  25.7026
    5  4.5836  30.0759
    6  3.7503  26.7065
    7  5.5671  34.6266
    8  5.8547  36.4237
    9  3.5338  24.0014
                                               
    P1(x) = 4.9829948*x + 6.980136
    
    P1(4.25) = 28.15786369
                                                   
    dP0 = 4.9829948
    
    dP0 = 4.9829948
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        method='linear-regression',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_polynomial(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    poly_order: int = 1,
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Polynomial method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    poly_order : int
        Degree of the interpolating polynomial. Required for when 
        `method=polynomial`.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19,
    ... 21, 22]
    >>> y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79,
    ... 90, 99, 99, 100]
    >>> x0 = 12.5278
    >>> result = stm.interp_polynomial(x, y, x0, poly_order=3,
    ... diff_order=1, plot_x0=True, decimal_points=8)
           x   f(x)
    0    1.0  100.0
    1    2.0   90.0
    2    3.0   80.0
    3    5.0   60.0
    4    6.0   60.0
    5    7.0   55.0
    6    8.0   60.0
    7    9.0   65.0
    8   10.0   70.0
    9   12.0   70.0
    10  13.0   75.0
    11  14.0   76.0
    12  15.0   78.0
    13  16.0   79.0
    14  18.0   90.0
    15  19.0   99.0
    16  21.0   99.0
    17  22.0  100.0
                                               
    P3(x) = -0.03032088*x**3 + 1.3433319*x**2 - 15.538304*x + 113.76804
    
    P3(12.5278) = 70.321073
                                                   
    dP2(x) = -0.090962639*x**2 + 2.6866638*x - 15.538304
    
    dP2(12.5278) = 3.8434816
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        method='polynomial',
        expr_variable=expr_variable,
        poly_order=poly_order,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_exponential(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    exp_type: Literal['b*exp(ax)', 'b*10^ax', 'ab^x'] = 'b*exp(ax)',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Exponential method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    exp_type : {'b*exp(ax)', 'b*10^ax', 'ab^x'}, optional default='exp(ax)'
        Type of exponential function to be applied.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [1, 1.25, 1.5, 1.75, 2]
    >>> y = [5.1, 5.79, 6.53, 7.45, 8.46]
    >>> x0 = 1.6
    >>> result = stm.interp_exponential(x, y, x0, diff_order=1,
    ... plot_x0=True, decimal_points=8)
          x  f(x)
    0  1.00  5.10
    1  1.25  5.79
    2  1.50  6.53
    3  1.75  7.45
    4  2.00  8.46
                                               
    P(x) = 3.0724927*exp(0.5057196034329076*x)
    
    P(1.6) = 6.9008221
                                                   
    dP(x) = 1.5538198*exp(0.5057196034329076*x)

    dP(1.6) = 3.4898810
    
    >>> x = [1, 1.25, 1.5, 1.75, 2]
    >>> y = [5.1, 5.79, 6.53, 7.45, 8.46]
    >>> x0 = 1.6
    >>> exp_types = ['b*exp(ax)', 'b*10^ax', 'ab^x']
    >>> for exp_type in exp_types:
    ...     print(f'\\nMethod: {exp_type}\\n')
    ...     result = stm.interp_exponential(x, y, x0, diff_order=1,
    ...         exp_type=exp_type, plot_x0=True, decimal_points=8)
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        method='exponential',
        expr_variable=expr_variable,
        exp_type=exp_type,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_power(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Power method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [1, 1.25, 1.5, 1.75, 2]
    >>> y = [5.1, 5.79, 6.53, 7.45, 8.46]
    >>> x0 = 1.6
    >>> result = stm.interp_power(x, y, x0, diff_order=1,
    ... plot_x0=True, decimal_points=8)
          x  f(x)
    0  1.00  5.10
    1  1.25  5.79
    2  1.50  6.53
    3  1.75  7.45
    4  2.00  8.46                                              
                                               
    P(x) = 4.9928734*x**0.72568601
    
    P(1.6) = 7.0222708
                                                   
    dP(x) = 3.6232584/x**0.27431399
    
    dP(1.6) = 3.1849773
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        method='power',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_saturation(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    sat_type: Literal['ax/(x+b)', 'x/(ax+b)'] = 'ax/(x+b)',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Saturation method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    sat_type : {'ax/(x+b)', 'x/(ax+b)'}, optional (default='ax/(x+b)')
        Type of saturation function to be applied.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [1, 1.25, 1.5, 1.75, 2]
    >>> y = [5.1, 5.79, 6.53, 7.45, 8.46]
    >>> x0 = 1.6
    >>> result = stm.interp_saturation(x, y, x0, sat_type='ax/(x+b)',
    ... diff_order=1, plot_x0=True, decimal_points=8)
          x  f(x)
    0  1.00  5.10
    1  1.25  5.79
    2  1.50  6.53
    3  1.75  7.45
    4  2.00  8.46
                                               
    P(x) = 21.700827*x/(x + 3.3363351)
    
    P(1.6) = 7.0338263
                                                   
    dP(x) = -1.9495616*x/(0.29973009*x + 1.0)**2 + 21.700827/(x + 3.3363351)
    
    dP(1.6) = 2.9712329
    
    >>> result = stm.interp_saturation(x, y, x0, sat_type='x/(ax+b)',
    ... diff_order=1, plot_x0=True, decimal_points=8)
          x  f(x)
    0  1.00  5.10
    1  1.25  5.79
    2  1.50  6.53
    3  1.75  7.45
    4  2.00  8.46
                                               
    P(x) = x/(0.046081193*x + 0.1537423)
    
    P(1.6) = 7.0338263
                                                   
    dP(x) = -1.9495616*x/(0.29973009*x + 1.0)**2 + 1/(0.046081193*x + 0.1537423)
    
    dP(1.6) = 2.9712329
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        method='saturation',
        expr_variable=expr_variable,
        sat_type=sat_type,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def interp_reciprocal(
    x: list[int | float],
    y: list[int | float],
    x0: int | float,
    expr_variable: str = 'x',
    plot_x0: bool = False,
    diff_order: int = 0,
    plot_deriv: bool = False,
    truncate_terms: float = 1e-16,
    auto_display: bool = True,
    decimal_points: int = 12
) -> Result:
    """
    Performs univariate interpolation using the Reciprocal method.
    
    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    y : array_like
        The y-coordinates of the data points, same length as `x`.
    x0 : {int, float}
        The point at which the interpolation should be performed.
    expr_variable : str, optional (default='x')
        Variable to be used in the interpolated polynomial.
    plot_x0 : bool, optional (default=True)
        If `True`, the point `(x0, f(x0))` will be plotted on the graph.
    diff_order :  int, optional (default=0)
        Order of differentiation. Default 0 means no differentiation.
    plot_deriv : bool, optional (default=False)
        If `True`, derivative will be plotted. Will be disregarded 
        if derivative is `diff_order=0`.
    truncate_terms : float, optional (default=1e-16)
        Smallest value for which terms should be removed. Terms below 
        this value will not be shown in the polynomial to be displayed.
        This does not however affect precision, that is, truncation is 
        only applied to the display, and not the polynomial itself.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=12)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with values used for calculations highlighted.
    f : {sympy.Expr, str}
        The interpolation polynomial from the given data points.
    fx : {Float, float}
        The value of the interpolation polynomial at `x0`, i.e. `f(x0)`.
    df : {sympy.Expr, str}
        The nth derivative of the interpolation polynomial.
    dfx : {Float, float}
        Value of the nth derivative of the interpolation polynomial at `x0`.
    plot : Image
        Image with the plot of the interpolated polynomial.
    
    Examples
    --------
    >>> import stemlab as stm
    >>> x = [1, 1.25, 1.5, 1.75, 2]
    >>> y = [5.1, 5.79, 6.53, 7.45, 8.46]
    >>> x0 = 1.6
    >>> result = stm.interp_reciprocal(x, y, x0, diff_order=1,
    ... plot_x0=True, decimal_points=8)
          x  f(x)
    0  1.00  5.10
    1  1.25  5.79
    2  1.50  6.53
    3  1.75  7.45
    4  2.00  8.46
                                               
    P(x) = 1/(0.27141235 - 0.077693451*x)
    
    P(1.6) = 6.7979659
                                                   
    dP(x) = 1.0546909/(1.0 - 0.28625614*x)**2
    
    dP(1.6) = 3.5903962
    """
    result = interp(
        x=x,
        y=y,
        x0=x0,
        method='reciprocal',
        expr_variable=expr_variable,
        plot_x0=plot_x0,
        diff_order=diff_order,
        plot_deriv=plot_deriv,
        truncate_terms=truncate_terms,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result