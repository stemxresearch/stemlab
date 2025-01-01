from math import factorial
from typing import Literal, Callable

from numpy import array, zeros, linspace, vstack, asfarray
from pandas import DataFrame
from sympy import symbols, sympify, solve, flatten, Integer, Float, Expr
from matplotlib.pyplot import (
    plot, figure, clf, close, title, xlabel, ylabel, legend, tight_layout
)
from matplotlib.style import context, available
from stemlab.core.symbolic import (
    sym_lambdify_expr, lambdify_system, sym_sympify
)
from stemlab.core.validators.errors import (
    LowerGteUpperError, SympifyError, ComputationError
)
from stemlab.core.base.strings import str_singular_plural
from stemlab.core.arraylike import arr_abrange, conv_to_arraylike
from stemlab.core.htmlatex import sta_dframe_color, tex_to_latex
from stemlab.core.plotting import figure_encoded
from stemlab.core.display import Result, display_results
from stemlab.core.validators.validate import ValidateArgs

METHODS_DICT = {
    'taylor1': 'taylor1',
    'taylor2': 'taylor2',
    'taylor3': 'taylor3',
    'taylor4': 'taylor4',
    'taylor5': 'taylor5',
    'taylor6': 'taylor6',
    'taylor7': 'taylor7',
    'taylor8': 'taylor8',
    'taylor9': 'taylor9',
    'forward-euler': 'feuler',
    'modified-euler': 'meuler',
    'backward-euler': 'beuler',
    'midpoint-runge-kutta': 'rkmidpoint',
    'modified-euler-runge-kutta': 'rkmeuler',
    'second-order-ralston': 'ralston2',
    'third-order-heun': 'heun3',
    'third-order-nystrom': 'nystrom3',
    'third-order-runge-kutta': 'rk3',
    'fourth-order-runge-kutta': 'rk4',
    'fourth-order-runge-kutta - 38': 'rk38',
    'fourth-order-runge-kutta-mersen': 'rkmersen',
    'fifth-order-runge-kutta': 'rk5',
    'backward-euler': 'rkbeuler',
    'trapezoidal': 'rktrapezoidal',
    'one-stage-gauss-legendre': 'rk1stage',
    'two-stage-gauss-legendre': 'rk2stage',
    'three-stage-gauss-legendre': 'rk3stage',
    'adams-bashforth - 2-step': 'ab2',
    'adams-bashforth - 3-step': 'ab3',
    'adams-bashforth-4-step': 'ab4',
    'adams-bashforth-5-step': 'ab5',
    'adams-moulton - 2-step': 'am2',
    'adams-moulton - 3-step': 'am3',
    'adams-moulton - 4-step': 'am4',
    'euler-heun': 'eheun',
    'adams-bashforth-moulton - 2-step': 'abm2',
    'adams-bashforth-moulton - 3-step': 'abm3',
    'adams-bashforth-moulton - 4-step': 'abm4',
    'adams-bashforth-moulton - 5-step': 'abm5',
    'msimpson': 'ms',
    'modified-msimpson': 'mms',
    'hamming': 'hamming',
    'runge-kutta-fehlberg-45': 'rkf45',
    'runge-kutta-fehlberg-54': 'rkf54',
    'runge-kutta-verner': 'rkv',
    'adams-variable-step-size': 'adamsvariablestep',
    'extrapolation': 'extrapolation',
    'trapezoidal-with-newton-approximation': 'tnewton'
}

START_VALUES_DICT = {
    'explicit-euler': 'feuler',
    'modified-euler': 'meuler',
    'third-order-heun': 'heun3',
    'fourth-order-runge-kutta':'rk4'
}

TAYLOR_N = [f'taylor{order + 1}' for order in range(9)]
EULER_METHODS = ['feuler', 'meuler', 'beuler']
EXPLICIT_RK = [
    'rkmidpoint', 
    'rkmeuler', 
    'ralston2', 
    'heun3', 
    'nystrom3', 
    'rk3', 
    'rk4', 
    'rk38', 
    'rkmersen', 
    'rk5'
]
IMPLICIT_RK = ['rkbeuler', 'rktrapezoidal', 'rk1stage', 'rk2stage', 'rk3stage']
EXPLICIT_MULTISTEP = ['ab2', 'ab3', 'ab4', 'ab5']
IMPLICIT_MULTISTEP = ['am2', 'am3', 'am4']
PREDICTOR_CORRECTOR = [
    'eheun', 'abm2','abm3','abm4', 'abm5', 'hamming', 'msimpson', 'mmsimpson'
]
ADAPTIVE_VARIABLE_STEP = [
    'rkf45', 'rkf54', 'rkv', 'adamsvariablestep', 'extrapolation', 'tnewton'
]

VALID_IVP_METHODS = (
    TAYLOR_N
    + EULER_METHODS 
    + EXPLICIT_RK 
    + IMPLICIT_RK 
    + EXPLICIT_MULTISTEP 
    + IMPLICIT_MULTISTEP 
    + PREDICTOR_CORRECTOR 
    + ADAPTIVE_VARIABLE_STEP
)
START_VALUE_METHODS = (
    EXPLICIT_MULTISTEP + IMPLICIT_MULTISTEP + PREDICTOR_CORRECTOR
)

# system of equations is only allowed for the following method
SYSTEM_OF_EQUATIONS = TAYLOR_N + EULER_METHODS + EXPLICIT_MULTISTEP + PREDICTOR_CORRECTOR


class ODESolver:
    """
    ODE solver superclass

    Attributes
    ----------
    ...
    
    Methods
    -------
    solve(self):
        Integrate the ODE(s).

    advance(self):
        Advance solution one time step.

    _start_values(self)
        Calculate start values for the solver.

    _dframe_table(self):
        Generate DataFrame table for results.

    _plot_results(self):
        Plot the results.
    """
    
    def __init__(
            self,
            method: Literal[
                'taylor1', 'taylor2', 'taylor3', 'taylor4', 'taylor5', 'taylor6', 'taylor7', 'taylor8', 'taylor9',
                'feuler', 'meuler', 'beuler',
                'rkmidpoint', 'rkmeuler', 'ralston2', 'heun3', 'nystrom3', 'rk3', 'rk4', 'rk38', 'rkmersen', 'rk5',
                'rkbeuler', 'rktrapezoidal', 'rk1stage', 'rk2stage', 'rk3stage',
                'ab2', 'ab3', 'ab4', 'ab5',
                'am2', 'am3', 'am4',
                'eheun', 'abm2','abm3','abm4', 'abm5', 'hamming', 'msimpson', 'mmsimpson',
                'rkf45', 'rkf54', 'rkv', 'adamsvariablestep', 'extrapolation', 'tnewton'
            ] = 'rk4',
            odeqtn: str = '-1.2 * y + 7 * exp(-0.3 * t)',
            exactsol: str | Expr | Callable | None = None,
            vars: list[str] = ['t', 'y'],
            derivs: list[str] | None = None,
            start_values_or_method: list[float] | Literal['feuler', 'meuler', 'heun3', 'rk4'] = 'rk4',
            time_span: list[float] = [],
            y0: int | float = None,
            stepsize: float | list[float] | None = None,
            nsteps: int = 10,
            maxit: int = 10,
            tolerance: float = 1e-6,
            show_iters: int | None = None,
            auto_display: bool = True,
            decimal_points: int = 8
        ):
        """Initializes the ODE solver with provided parameters."""

        self.method = method
        self.odeqtn = odeqtn
        self.exactsol = exactsol
        self.vars = vars
        self.derivs = derivs
        self.start_values_or_method = start_values_or_method
        self.time_span = time_span
        self.y0 = y0
        self.stepsize = stepsize
        self.nsteps = nsteps
        self.maxit = maxit
        self.tolerance = tolerance
        self.show_iters = show_iters
        self.auto_display = auto_display
        self.decimal_points = decimal_points

        # for app
        dict_methods_app = {v: k for k, v in METHODS_DICT.items()}
        method_app = METHODS_DICT.get(self.method)
        if isinstance(start_values_or_method, str):
            start_method_app = START_VALUES_DICT.get(start_values_or_method)

        # get inputs
        # ----------
        # method
        self.method = ValidateArgs.check_member(
            par_name='method', 
            valid_items=VALID_IVP_METHODS, 
            user_input=self.method
        )

        # odeqtn
        try:
            try:
                self.fty = sym_sympify(self.odeqtn)
            except:
                self.fty = self.odeqtn
                self.fty_latex = 'f(x)'
            if isinstance(self.fty, (list, tuple)):
                if self.method not in SYSTEM_OF_EQUATIONS:
                    raise TypeError(
                        f'System of equations only applicable for the '
                        f'methods: {", ".join(SYSTEM_OF_EQUATIONS)}'
                    )
                self.number_of_odes = len(self.fty)
                self.fty_latex = 'f(x)'
                self.f_list = conv_to_arraylike(
                    array_values=self.odeqtn, 
                    to_ndarray=True, 
                    n=self.number_of_odes, 
                    label='exactly', 
                    par_name='odeqtn'
                )
                self.f = lambdify_system(
                    system_eqtns=self.f_list, 
                    is_univariate=False,
                    variables=self.vars,
                    par_name='odeqtn'
                )
            else:
                self.number_of_odes = 1
                try:
                    self.fty_latex = tex_to_latex(self.fty) # will crush if function
                except:
                    pass
                self.f = sym_lambdify_expr(
                    fexpr=self.odeqtn, 
                    is_univariate=False, 
                    variables=self.vars,
                    par_name='odeqtn'
                )
        except Exception:
            raise SympifyError(
                par_name='odeqtn', user_input=self.odeqtn
            )
        
        # exactsol
        if self.exactsol:
            try:
                self.ft = sym_lambdify_expr(
                    fexpr=self.exactsol, 
                    is_univariate=True, 
                    variables=self.vars[0],
                    par_name='exactsol'
                )
            except Exception:
                raise SympifyError(
                    par_name='exactsol', user_input=self.exactsol
                )
        else:
            self.ft = None
        
        # derivs (for taylor methods only)
        if self.method in TAYLOR_N:
            self.taylor_order = int(self.method[-1])
            if self.taylor_order < 2 or self.taylor_order > 9:
                if self.taylor_order == 1:
                    msg = (
                        'Expected Taylor order to bebetween 2 and 9 '
                        'inclusive. For order 1 as you have specified, use the '
                        'forward Euler function `feuler`'
                    )
                else:
                    msg = (
                        f'Expected Taylor order to bebetween 2 and 9 '
                        f'inclusive but got: {self.taylor_order}'
                    )
                raise ValueError(msg)
            if self.derivs:
                if isinstance(self.derivs, str):
                    try:
                        self.derivs = sympify(self.derivs.split(','))
                    except Exception:
                        raise SympifyError(
                            par_name='derivs', user_input=self.derivs
                        )
                self.derivs = conv_to_arraylike(
                    self.derivs,
                    n=self.taylor_order - 1,
                    par_name='derivs'
                )
            else:
                s = str_singular_plural(n=self.taylor_order)
                raise ValueError(
                    f"'{self.method}' expects {self.taylor_order} "
                    f"derivative{s} for the specified ODE equation "
                    f"but got None'"
                )
        
        # time_span
        time_span = conv_to_arraylike(
            array_values=time_span, n=2, to_ndarray=True, par_name='time_span'
        )

        self.t0 = ValidateArgs.check_numeric(
            par_name=f'time_span: {time_span[0]}', 
            to_float=True, 
            user_input=time_span[0]
        )
        
        self.tf = ValidateArgs.check_numeric(
            par_name=f'time_span: {time_span[1]}',
            to_float=True, 
            user_input=time_span[1]
        )
        
        if self.t0 >= self.tf:
            raise LowerGteUpperError(
                par_name='time_span', 
                lower_par_name='t0', 
                upper_par_name='tf', 
                user_input=[self.t0, self.tf]
            )
        
        # stepsize or nsteps
        # stepsize (h), number of steps (n), t, hmin, hmax
        index = ADAPTIVE_VARIABLE_STEP.index('adamsvariablestep')
        if self.method not in ADAPTIVE_VARIABLE_STEP[:index]:
            if self.stepsize is None:
                nsteps = ValidateArgs.check_numeric(
                    par_name='nsteps', 
                    is_integer=True, 
                    user_input=self.nsteps
                )
                self.n = abs(nsteps)
                self.h = float((self.tf - self.t0) / self.n)
                self.t = linspace(self.t0, self.tf, self.n + 1)
                self.n = len(self.t)
            else:
                stepsize = ValidateArgs.check_numeric(
                    par_name='stepsize', 
                    to_float=True, 
                    user_input=self.stepsize
                )
                self.h = stepsize
                self.t = arr_abrange(self.t0, self.tf, self.h)
                self.n = len(self.t)
        else: # adaptive methods
            stepsize = conv_to_arraylike(
                array_values=stepsize, n = 2, par_name='hmin_hmax'
            )
            
            # hmin
            self.hmin = ValidateArgs.check_numeric(
                par_name='- hmin_hmax[0]', to_float=True, user_input=stepsize[0]
            )
            
            # hmax
            self.hmax = ValidateArgs.check_numeric(
                par_name='- hmin_hmax[1]', to_float=True, user_input=stepsize[1]
            )
            
            if self.hmin >= self.hmax:
                raise LowerGteUpperError(
                    par_name='hmin_hmax', 
                    lower_par_name='hmin_hmax[0]', 
                    upper_par_name='hmin_hmax[1]', 
                    user_input=[self.hmin, self.hmax]
                )
            # these placeholders, otherwise, adaptive methods do not require them
            self.n = 10
            self.t = arr_abrange(self.t0, self.tf, self.hmin)
            self.n = len(self.t)
        
        # y0
        try:
            self.y0 = sympify(self.y0)
        except Exception:
            raise TypeError(
                f"Expected 'y0' to be a numeric value (for single ODE) "
                f"or array_like (for systems of ODES) but got: {self.y0}"
            )
        if isinstance(self.y0, (list, tuple)):
            self.y0 = conv_to_arraylike(
                array_values=self.y0, 
                n=len(self.y0), 
                to_ndarray=True, 
                par_name='y0'
            )
            self.number_of_odes = len(self.y0)
        else:
            self.y0 = ValidateArgs.check_numeric(
                par_name='y0', to_float=True, user_input=self.y0
            )
            self.number_of_odes = 1
  
        # tolerance
        if tolerance is not None:
            self.tolerance = ValidateArgs.check_numeric(
                par_name='tolerance', 
                boundary='exclusive', 
                to_float=True, 
                user_input=self.tolerance
            )

        maxit = ValidateArgs.check_numeric(
            par_name='maxit', 
            limits=[1, self.n], 
            is_integer=True, 
            user_input=maxit
        )
        # maximum iterations for Trapezoidal newton method
        self.N = (self.n + 1) if method == 'tnewton' else maxit

        # show_iters
        if self.show_iters is None:
            self.show_iters = self.n
        else:
            show_iters = ValidateArgs.check_numeric(
                par_name='show_iters', 
                limits=[1, self.n], 
                is_integer=True, 
                user_input=self.show_iters
            )
            self.n = self.show_iters if self.show_iters <= self.n else self.n

        self.auto_display = ValidateArgs.check_boolean(user_input=self.auto_display, default=True)
        self.decimal_points = ValidateArgs.check_decimals(x=self.decimal_points)

        # start values for multi-step methods
        self.methods_n_dict = {
            'ab2': 1, 'am2': 1, 'abm2': 1,
            'ab3': 2, 'am3': 2, 'abm3': 2,
            'ab4': 3, 'am4': 3, 'abm4': 3,
            'ab5': 4, 'am5': 4, 'abm5': 4,
            'hamming': 3, 'msimpson': 3, 'mmsimpson': 3
        }


    def _start_values(self):
        """
        Calculate start values for the multistep and predictor methods.

        Returns
        -------
        list: An array with start values for multistep and predictor
        methods.
        """
        
        if isinstance(self.start_values_or_method, str):
            self.start_method = self.start_values_or_method
            self.start_values_or_method = None
        else:
            self.start_method = None
            self.start_values_or_method = self.start_values_or_method 
                 
        if self.start_values_or_method is None:
            start_method = ValidateArgs.check_member(
                par_name='start_method', 
                valid_items=list(START_VALUES_DICT.values()), 
                user_input=self.start_method
            )
            if start_method == "feuler":
                self.y = self._forward_euler()
            elif start_method == "meuler":
                self.y = self._modified_euler()
            elif start_method == "heun3":
                self.y = self._heun3()
            elif start_method == "rk4":
                self.y = self._rk4()
        else:
            m = self.methods_n_dict.get(self.method, 0) + 1
            self.start_values_or_method = conv_to_arraylike(
                array_values = self.start_values_or_method,
                n = m,
                to_ndarray=True,
                par_name='start_values_or_method'
            )
            self.y[:m, :] = asfarray(self.start_values_or_method).reshape(-1, 1)

        return self.y


    def _forward_euler(self):
        """Forward Euler for start values"""
        f, t, y = self.f, self.t, self.y
        for i in range(5):
            h = t[i + 1] - t[i]
            y[i + 1, :] = y[i, :] + h * f(t[i], y[i, :])

        return y


    def _modified_euler(self):
        """Forward Euler for start values"""
        f, t, y = self.f, self.t, self.y
        for i in range(5):
            h = t[i + 1] - t[i]
            ynew = y[i, :] + h * f(t[i], y[i, :])
            y[i + 1, :] = y[i, :] + (h / 2) * (
                f(t[i], y[i, :]) + f(t[i + 1], ynew)
            )

        return y
        

    def _heun3(self):
        """Third Heun for start values"""
        f, t, y = self.f, self.t, self.y
        for i in range(5):
            h = t[i + 1] - t[i]
            k1 = h * f(t[i], y[i, :])
            k2 = h * f(t[i] + (1 / 3) * h, y[i, :] + (1 / 3) * k1)
            k3 = h * f(t[i] + (2 / 3) * h, y[i, :] + (2 / 3) * k2)
            y[i + 1, :] = y[i, :] + (1 / 4) * (k1 + 3 *  k3)

        return y


    def _rk4(self):
        """Fourth order Runge-Kutta for start values"""
        f, t, y = self.f, self.t, self.y
        for i in range(5):
            h = t[i + 1] - t[i]
            k1 = h * f(t[i], y[i, :])
            k2 = h * f(t[i] + h / 2, y[i, :] + k1 / 2)
            k3 = h * f(t[i] + h / 2, y[i, :] + k2 / 2)
            k4 = h * f(t[i] + h, y[i, :] + k3)
            y[i + 1, :] = y[i, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return y


    def compute(self):
        """
        Integrate the ODE(s).
        
        Returns
        -------
            DataFrame: DataFrame containing the solution.
            float: The computed answer.
        """
        self.y = zeros((self.n, self.number_of_odes))
        self.y[0, :] = self.y0
        if self.method in START_VALUE_METHODS:
            start_n = self.methods_n_dict.get(self.method, 0)
            self.y = self._start_values()
        else:
            start_n = 0 # for non-multiple step methods
        # perform iterations
        if self.method in ADAPTIVE_VARIABLE_STEP:
            self.t, self.y = self.advance()
        else:
            for i in range(start_n, self.n - 1):
                self.i = i
                self.y[i + 1, :] = self.advance()
        
        dframe, dframe_styled, answer = self._dframe_table()
        plot = self._plot_results()
        
        if self.auto_display:
            display_results({
                'table': dframe_styled,
                'Answer': answer,
                'decimal_points':self.decimal_points
            })

        result = Result(
            table=dframe, table_styled=dframe_styled, answer=answer, plot=plot
        )
        
        return result
    

    def advance(self):
        """
        Advance solution one time step.

        Returns
        -------
        numpy.ndarray
            The updated solution after advancing one time step.
        """
        raise NotImplementedError
    

    def _dframe_table(self):
        """
        Generate DataFrame table for results.

        Returns
        -------
        DataFrame:
            DataFrame containing the results.
        float:
            The computed answer.
        """
        # t, and y must be converted to numpy arrays
        t = asfarray(flatten(self.t))
        y = asfarray(flatten(self.y))
        show_iters = len(y)
        t = t[:show_iters]
        if self.exactsol:
            y_time_span = self.ft(t)
            absolute_error = abs(y_time_span - y)
            table_results = vstack([t, y, y_time_span, absolute_error]).T
            col_names = [
                'Time (t)', 
                'Approximated (yi)', 
                'Exact solution(y)', 
                'Error: | y - yi |'
            ]
        else:
            table_results = vstack([t, y]).T
            col_names = [
                'Time (t)', 
                'Approximated (yi)'
            ]

        answer = table_results[-1, 1]
        table_results = table_results[:show_iters, :]
        self.dframe = DataFrame(
            table_results, columns=col_names
        )

        # css styled
        dframe_styled = sta_dframe_color(
            dframe=self.dframe,
            style_indices=[[-1, 1]],
            decimal_points=self.decimal_points
        )
        
        return self.dframe, dframe_styled, answer

    
    def _plot_results(self):
        """
        Plot the results of the solver.

        This method plots the numerical solution obtained from the 
        solver alongside the exact solution (if available), using 
        various plot styles defined in `fig_styles`.

        Returns
        -------
        figure_code : str
            Encoded figure.

        Raises
        ------
        ValueError
            If the specified figure style is not valid.
        """
        valid_fig_styles = available
        fig_styles = {
            'fig_style': 'fast',
            'fig_width': 7,
            'fig_height': 5,
            'fig_marker': 'D',
            'fig_markerfacecolor': 'b',
            'fig_markersize': 6,
            'fig_linestyle': '-',
            'fig_color': 'b',
            'fig_linewidth': 1.5,
            'fig_title': f'$ f(t, y) = {self.fty_latex}$',
            'fig_xtitle': 'Time (t)',
            'fig_ytitle': 'Solution (y)',
            'fig_fontcolor': 'k',
            'fig_fontsize': 13
        }

        fig_style = fig_styles.get('fig_style')
        if fig_style not in valid_fig_styles:
            raise ValueError(f"'{fig_style}' is an invalid plot style.")

        clf()
        close('all')
        with context(fig_style):
            figure(figsize=(fig_styles['fig_width'], fig_styles['fig_height']))
            t_exact, y_exact = (0, 0)
            if self.exactsol:
                t_exact = linspace(self.t0, self.tf, 250)
                y_exact = self.ft(t_exact)
                plot(
                    t_exact,
                    y_exact, 
                    color='red', 
                    linewidth=2, 
                    label='Exact'
                )

            tab_results = self.dframe.values
            nrows = tab_results.shape[0]
            if self.number_of_odes > 1:
                tab_results = tab_results[:nrows, :]
                marker = fig_styles.get('fig_marker') if nrows <= 30 else None

                for i in range(tab_results.shape[1]):
                    plot(
                        tab_results[:, 0], 
                        tab_results[:, i], 
                        marker=marker, 
                        label=f'{i + 1}'
                    )
            else:
                plot_kwargs = {
                    'linestyle': fig_styles['fig_linestyle'],
                    'color': fig_styles['fig_color'],
                    'linewidth': fig_styles['fig_linewidth']
                }

                if nrows <= 30:
                    plot_kwargs.update({
                        'marker': fig_styles['fig_marker'],
                        'markersize': fig_styles['fig_markersize'],
                        'markerfacecolor': fig_styles['fig_markerfacecolor']
                    })

                plot(
                    tab_results[:, 0], 
                    tab_results[:, 1], 
                    **plot_kwargs, 
                    label='Solution'
                )
                legend(
                    loc='best', 
                    ncol=2, 
                    prop={'family': 'monospace', 'weight': 'normal', 'size': 12}
                )

            fig_fontdict = {
                'color': fig_styles['fig_fontcolor'],
                'size': fig_styles['fig_fontsize']
            }
            title(fig_styles['fig_title'], fontdict=fig_fontdict)
            xlabel(fig_styles['fig_xtitle'], fontdict=fig_fontdict)
            ylabel(fig_styles['fig_ytitle'], fontdict=fig_fontdict)
            tight_layout()
        figure_code = figure_encoded()

        return figure_code
    

class TaylorN(ODESolver):
    """Taylor order n method"""
    def advance(self):
        fty, derivs, taylor_order, t, y, i = (
            self.fty,  self.derivs, self.taylor_order, self.t, self.y, self.i
        )
        h = t[i + 1] - t[i]
        deriv_funcs = lambdify_system(
            system_eqtns=[fty] + derivs, 
            is_univariate=False,
            variables=self.vars,
            par_name=f'ode_derivative'
        )
        y_temp = 0
        # Calculate the derivative terms up to the specified order
        for j in range(1, taylor_order + 1):
            y_temp += (h**j / factorial(j)) * deriv_funcs[j - 1](t[i], y[i])

        return y[i] + y_temp


class ForwardEuler(ODESolver):
    """Forward Euler method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        y_step_i = y[i, :] + h * f(t[i], y[i, :])

        return y_step_i


class ModifiedEuler(ODESolver):
    """Modified Euler method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        ynew = y[i, :] + h * f(t[i], y[i, :])
        y_step_i = y[i, :] + (h / 2) * (f(t[i], y[i, :]) + f(t[i + 1], ynew))

        return y_step_i
    

class BackwardEuler(ODESolver):
    """Backward Euler method"""
    def advance(self):
        y, t, i = flatten(self.y), self.t, self.i
        h = t[i + 1] - t[i]
        fs_symbolic = maths_transform_case(ftn_str=str(self.fty), upper=True)
        fs_symbolic = fs_symbolic.replace('t', f'({t[i + 1]})').replace('y', f'y{i}')
        fs_symbolic = f'y{i} - ({y[i]} + {h} * ({fs_symbolic}))'
        y_step_i = _solve_symbolic(fs_symbolic, method='Backward Euler')

        return y_step_i
    
    
class MidpointRK(ODESolver):
    """Midpoint Runge-Kutta method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        k1 = h * f(t[i], y[i, :])
        k2 = h * f(t[i] + h / 2, y[i, :] + k1 / 2)
        y_step_i = y[i, :] + k2

        return y_step_i 

    
class ModifiedEulerRK(ODESolver):
    """Modified Euler Runge-Kutta method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        k1 = h * f(t[i], y[i, :])
        k2 = h * f(t[i] + h, y[i, :] + k1)
        y_step_i = y[i, :] + (1 / 2) * (k1 + k2)

        return y_step_i

    
class Ralston2(ODESolver):
    """Second order Ralston method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        k1 = h * f(t[i], y[i, :])
        k2 = h * f(t[i] + (3 / 4) * h, y[i, :] + (3 / 4) * k1)
        y_step_i = y[i, :] + (1 / 3) * (k1 + 2 * k2)

        return y_step_i

    
class Heun3(ODESolver):
    """Third order Heun method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        k1 = h * f(t[i], y[i, :])
        k2 = h * f(t[i] + (1 / 3) * h, y[i, :] + (1 / 3) * k1)
        k3 = h * f(t[i] + (2 / 3) * h, y[i, :] + (2 / 3) * k2)
        y_step_i = y[i, :] + (1 / 4) * (k1 + 3 *  k3)

        return y_step_i

    
class Nystrom3(ODESolver):
    """Third order Nystrom method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        k1 = h * f(t[i], y[i, :])
        k2 = h * f(t[i] + 2 / 3 * h, y[i, :] + 2 / 3 * k1)
        k3 = h * f(t[i] + 2 / 3 * h, y[i, :] + 2 / 3 * k2)
        y_step_i = y[i, :] + (1 / 8) * (2 * k1 + 3 * k2 + 3 * k3)

        return y_step_i

    
class RK3(ODESolver):
    """Third order Runge-Kutta method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        k1 = h * f(t[i], y[i, :])
        k2 = h * f(t[i] + h / 2, y[i, :] + 1 / 2 * k1)
        k3 = h * f(t[i] + h, y[i, :] - k1 + 2 * k2)
        y_step_i = y[i, :] + (1 / 6) * (k1 + 4 * k2 + k3)

        return y_step_i

    
class RK4(ODESolver):
    """Fourth order Runge-Kutta method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        k1 = h * f(t[i], y[i, :])
        k2 = h * f(t[i] + h / 2, y[i, :] + k1 / 2)
        k3 = h * f(t[i] + h / 2, y[i, :] + k2 / 2)
        k4 = h * f(t[i] + h, y[i, :] + k3)
        y_step_i = y[i, :] + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return y_step_i

    
class RK38(ODESolver):
    """Runge-Kutta 3/8 method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        k1 = h * f(t[i], y[i, :])
        k2 = h * f(t[i] + 1 / 3 * h, y[i, :] + 1 / 3 * k1)
        k3 = h * f(t[i] + 2 / 3 * h, y[i, :] - 1 / 3 * k1 + k2)
        k4 = h * f(t[i] + h, y[i, :] + k1 - k2 + k3)
        y_step_i = y[i, :] + (1 / 8) * (k1 + 3 * k2 + 3 * k3 + k4)

        return y_step_i
    

class RKMersen(ODESolver):
    """Runge-Kutta-Mersen method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        k1 = h * f(t[i], y[i, :])
        k2 = h * f(t[i] + h / 3, y[i, :] + k1 / 3)
        k3 = h * f(t[i] + h / 2, y[i, :] + k1 / 6 + k2 / 6)
        k4 = h * f(t[i] + h / 2, y[i, :] + k1 / 8 + 3 / 8 * k3)
        k5 = h * f(t[i] + h, y[i, :] + k1 / 2 - 3 / 2 * k3 + 2 * k4)
        y_step_i = y[i, :] + (1 / 6) * (k1 + 4 * k2 + k5)

        return y_step_i

    
class RK5(ODESolver):
    """Fifth order Runge-Kutta method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        k1 = h * f(t[i], y[i, :])
        k2 = h * f(t[i] + h / 4, y[i, :] + (1 / 4) * k1)
        k3 = h * f(t[i] + h / 4, y[i, :] + (1 / 8) * k1 + (1 / 8) * k2)
        k4 = h * f(t[i] + h / 2, y[i, :] - (1 / 2) * k2 + k3)
        k5 = h * f(t[i] + (3 * h) / 4, y[i, :] + (3 / 16) * k1 + (9 / 16) * k4)
        k6 = h * f(
            t[i] + h, 
            y[i, :] - (3 / 7) * k1 + (2 / 7) * k2 + (12 / 7) * k3 
            - (12 / 7) * k4 + (8 / 7) * k5
        )
        y_step_i = y[i, :] + (1 / 90) * (
            7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6
        )
        
        return y_step_i

    
class BackwardEulerRK(ODESolver):
    """Backward Euler Runge-Kutta method"""
    def advance(self):
        y, t, i = flatten(self.y), self.t, self.i
        h = t[i + 1] - t[i]
        fs_symbolic = maths_transform_case(ftn_str=str(self.fty), upper=True)
        fs_symbolic = fs_symbolic.replace('t', f'({t[i + 1]})').replace('y', f'y{i}')
        fs_symbolic = f'y{i} - ({y[i]} + {h} * ({fs_symbolic}))'
        y_step_i = _solve_symbolic(fs_symbolic, method='Backward Euler')
        
        return y_step_i

    
class TrapezoidalRK(ODESolver):
    """Trapezoidal Runge-Kutta method"""
    def advance(self):
        y, t, i = flatten(self.y), self.t, self.i
        h = t[i + 1] - t[i]
        fs_symbolic = maths_transform_case(ftn_str=str(self.fty), upper=True)
        fs_symbolic = fs_symbolic.replace('t', f'({t[i + 1]})').replace('y', f'y{i}')
        fs_symbolic = f'y{i} - ({y[i]} + {h / 2} * ({fs_symbolic} + {self.f(t[i], y[i])}))'
        y_step_i = _solve_symbolic(fs_symbolic, method='Trapezoidal Runge-Kutta method')
        
        return y_step_i

    
class GaussLegendre1Stage(ODESolver):
    """Gauss-Legendre 1 stage"""
    def advance(self):
        y, t, i = flatten(self.y), self.t, self.i
        h = t[i + 1] - t[i]
        k1 = symbols('k1')
        fty = maths_transform_case(ftn_str=str(self.fty), upper=True)
        fs_symbolic = fty.replace('t', f'{t[i] + h / 2}')
        fs_symbolic = fs_symbolic.replace('y', f'{y[i]} + {h / 2 * k1}')
        fs_symbolic = sympify(k1 - sympify(fs_symbolic))
        equation_solved = array(solve(fs_symbolic, k1, dict=True))
        k1 = float(equation_solved[0][k1])

        y_step_i = y[i] + h * k1
        
        return y_step_i
    

class GaussLegendre2Stage(ODESolver):
    """Gauss-Legendre 2 stage"""
    def advance(self):
        y, t, i = flatten(self.y), self.t, self.i
        h = t[i + 1] - t[i]
        k1, k2 = symbols('k1 k2')
        fty = maths_transform_case(ftn_str=str(self.fty), upper=True)
        fs_symbolic = fty.replace('t', f'{t[i] + (1 / 2 - (3 ** 0.5) / 6) * h}')
        y_new = y[i] + 1 / 4 * h * k1 + h * (1 / 4 - (3 ** 0.5) / 6) * k2
        fs_symbolic = fs_symbolic.replace('y', str(y_new))
        equation_k1 = sympify(k1 - sympify(fs_symbolic))
        
        fs_symbolic = fty.replace('t', f'{t[i] + (1 / 2 + (3 ** 0.5) / 6) * h}')
        y_new = y[i] + 1 / 4 * h * k1 + h * (1 / 4 + (3 ** 0.5) / 6) * k2
        fs_symbolic = fs_symbolic.replace('y', str(y_new))
        equation_k2 = sympify(k2 - sympify(fs_symbolic))
        
        equations_k1_k2 = [equation_k1, equation_k2]    
        solutions_k1_k2 = array(solve(equations_k1_k2, k1, k2, dict=True))
        k1 = float(solutions_k1_k2[0][k1])
        k2 = float(solutions_k1_k2[0][k2])
        
        y_step_i = y[i] + (h / 2) * (k1 + k2)
        
        return y_step_i
    
class GaussLegendre3Stage(ODESolver):
    """Gauss-Legendre 3 stage"""
    def advance(self):
        y, t, i = flatten(self.y), self.t, self.i
        h = t[i + 1] - t[i]
        k1, k2, k3 = symbols('k1 k2 k3')
        fty = maths_transform_case(ftn_str=str(self.fty), upper=True)
        fs_symbolic = fty.replace('t', f'{t[i] + (1 / 2 - (15 ** 0.5) / 10) * h}')
        y_new = (
            y[i] 
            + 5 / 26 * h * k1 
            + (2 / 9 - (15 ** 0.5) / 15) * h * k2 
            + (5 / 36 - (15 ** 0.5) / 30) * h * k3
        )
        fs_symbolic = fs_symbolic.replace('y', str(y_new))
        equation_k1 = sympify(k1 - sympify(fs_symbolic))
        
        fs_symbolic = fty.replace('t', f'{t[i] + (1 / 2) * h}')
        y_new = (
            y[i] 
            + (5 / 36 + (15 ** 0.5) / 24) * h * k1 
            + (2 / 9) * h * k2 
            + (5 / 36 - (15 ** 0.5) / 24) * h * k3
        )
        fs_symbolic = fs_symbolic.replace('y', str(y_new))
        equation_k2 = sympify(k2 - sympify(fs_symbolic))
        
        fs_symbolic = fty.replace('t', f'{t[i] + (1 / 2 + (15 ** 0.5) / 10) * h}')
        y_new = (
            y[i] 
            + (5 / 36 + (15 ** 0.5) / 30) * h * k1 
            + (2 / 5 + (15 ** 0.5) / 15) * h * k2 
            + (5 / 36) * h * k3
        )
        fs_symbolic = fs_symbolic.replace('y', str(y_new))
        equation_k3 = sympify(k3 - sympify(fs_symbolic))
        
        equations_k1_k2_k3 = [equation_k1, equation_k2, equation_k3]
        solutions_k1_k2_k3 = array(
            solve(equations_k1_k2_k3, k1, k2, k3, dict=True)
        )
        k1 = float(solutions_k1_k2_k3[0][k1])
        k2 = float(solutions_k1_k2_k3[0][k2])
        k3 = float(solutions_k1_k2_k3[0][k3])
        
        y_step_i = y[i] + (h / 18) * (5 * k1 + 8 * k2 + 5 * k3)
        
        return y_step_i

    
class AdamBashforth2Step(ODESolver):
    """Adam-Bashforth 2 step method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        y_step_i = y[i, :] + (h / 2) * (
            3 * f(t[i], y[i, :]) - f(t[i - 1], y[i - 1, :])
        )
        
        return y_step_i

    
class AdamBashforth3Step(ODESolver):
    """Adam-Bashforth 3 step method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        y_step_i = y[i, :] + (h / 12) * (
            23 * f(t[i], y[i, :]) - 16 * f(t[i - 1], y[i - 1, :]) 
            + 5 * f(t[i - 2], y[i - 2, :])
        )

        return y_step_i
    

class AdamBashforth4Step(ODESolver):
    """Adam-Bashforth 4 step method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        y_step_i = y[i, :] + (h / 24) * (
            55 * f(t[i], y[i, :]) - 59 * f(t[i - 1], y[i - 1]) 
            + 37 * f(t[i - 2], y[i - 2]) - 9 * f(t[i - 3], y[i - 3])
        )

        return y_step_i
    

class AdamBashforth5Step(ODESolver):
    """Adam-Bashforth 5 step method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        y_step_i = y[i, :] + (h / 720) * (
            1901 * f(t[i], y[i, :]) - 2774 * f(t[i - 1], y[i - 1]) 
            + 2616 * f(t[i - 2], y[i - 2]) - 1274 * f(t[i - 3], y[i - 3]) 
            + 251 * f(t[i - 4], y[i - 4])
        )
        
        return y_step_i

    
class AdamMoulton2Step(ODESolver):
    """Adam-Moulton 2 step method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        fs_symbolic = maths_transform_case(ftn_str=str(self.fty), upper=True)
        fs_symbolic = fs_symbolic.replace('t', f'({t[i + 1]})').replace('y', f'y{i}')
        # note the use of y[i][0] to extract the value of a list
        fty_symbolic = f'y{i} - ({y[i][0]} + {h / 12} * (5 * ({fs_symbolic}) + {8 * f(t[i], y[i][0]) - f(t[i - 1], y[i - 1][0])}))'
        y_step_i = _solve_symbolic(fty_symbolic, method='Adam-Moulton 2 step')
        
        return y_step_i
    

class AdamMoulton3Step(ODESolver):
    """Adam-Moulton 3 step method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        fs_symbolic = maths_transform_case(ftn_str=str(self.fty), upper=True)
        fs_symbolic = fs_symbolic.replace('t', f'({t[i + 1]})').replace('y', f'y{i}')
        # note the use of y[i][0] to extract the value of a list
        fty_symbolic = f'y{i} - ({y[i][0]} + {h / 24} * (9 * ({fs_symbolic}) + {19 * f(t[i], y[i][0]) - 5 * f(t[i - 1], y[i - 1][0]) + f(t[i - 2], y[i - 2][0])}))'
        y_step_i = _solve_symbolic(fty_symbolic, method='Adam-Moulton 3 step')

        return y_step_i

    
class AdamMoulton4Step(ODESolver):
    """Adam-Moulton 4 step method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        fs_symbolic = maths_transform_case(ftn_str=str(self.fty), upper=True)
        fs_symbolic = fs_symbolic.replace('t', f'({t[i + 1]})').replace('y', f'y{i}')
        fty_symbolic = f'y{i} - ({y[i][0]} + {h / 720} * (251 * ({fs_symbolic}) + {646 * f(t[i], y[i][0]) - 264 * f(t[i - 1], y[i - 1][0]) + 106 * f(t[i - 2], y[i - 2][0]) - 19 * f(t[i - 3], y[i - 3][0])}))'
        y_step_i = _solve_symbolic(fty_symbolic, method='Adam-Moulton 3 step')

        return y_step_i

    
class EulerHeun(ODESolver):
    """Euler-Heun method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        # Explicit Euler as predictor
        y[i + 1, :] = y[i, :] + h * f(t[i], y[i, :])
        # Heun as corrector
        y_step_i = y[i, :] + (h / 2) * (
            f(t[i + 1], y[i + 1, :]) + f(t[i], y[i, :])
        )

        return y_step_i

    
class AdamBashforthMoulton2Step(ODESolver):
    """Adam-Bashforth-Moulton 2 stage method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        # Adams-Bashforth 2-step as predictor
        y[i + 1, :] = y[i, :] + (h / 2) * (
            3 * f(t[i], y[i, :]) - f(t[i - 1], y[i - 1])
        )
        # Adams-Moulton 2-step as corrector
        y_step_i = y[i, :] + (h / 2) * (
            f(t[i + 1], y[i + 1, :]) + f(t[i], y[i, :])
        )

        return y_step_i

    
class AdamBashforthMoulton3Step(ODESolver):
    """Adam-Bashforth-Moulton 3 stage method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        # Adams-Bashforth 3-step as predictor
        y[i + 1, :] = y[i, :] + (h / 12) * (
            23 * f(t[i], y[i, :]) - 16 * f(t[i - 1], y[i - 1]) 
            + 5 * f(t[i - 2], y[i - 2])
        )
        # Adams-Moulton 2-step as corrector
        y_step_i = y[i, :] + (h / 12) * (
            5 * f(t[i + 1], y[i + 1, :]) + 8 * f(t[i], y[i, :]) 
            - f(t[i - 1], y[i - 1])
        )

        return y_step_i
    
    
class AdamBashforthMoulton4Step(ODESolver):
    """Adam-Bashforth-Moulton 4 stage method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        # Adams-Bashforth 4-step as predictor
        y[i + 1, :] = y[i, :] + (h / 24) * (
            55 * f(t[i], y[i, :]) - 59 * f(t[i - 1], y[i - 1]) 
            + 37 * f(t[i - 2], y[i - 2]) - 9 * f(t[i - 3], y[i - 3])
        )
        # Adams-Moulton 3-step as corrector
        y_step_i = y[i, :] + (h / 24) * (
            9 * f(t[i + 1], y[i + 1, :]) + 19 * f(t[i], y[i, :]) 
            - 5 * f(t[i - 1], y[i - 1]) + f(t[i - 2], y[i - 2])
        )

        return y_step_i

    
class AdamBashforthMoulton5Step(ODESolver):
    """Adam-Bashforth-Moulton 5 stage method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        # Adams-Bashforth 5-step as predictor
        y[i + 1, :] = y[i, :] + (h / 720) * (
            1901 * f(t[i], y[i, :]) - 2774 * f(t[i - 1], y[i - 1]) 
            + 2616 * f(t[i - 2], y[i - 2]) - 1274 * f(t[i - 3], y[i - 3]) 
            + 251 * f(t[i - 4], y[i - 4])
        )
        # Adams-Moulton 4-step as corrector
        y_step_i = y[i, :] + (h / 720) * (
            251 * f(t[i + 1], y[i + 1, :]) + 646 * f(t[i], y[i, :]) 
            - 264 * f(t[i - 1], y[i - 1]) + 106 * f(t[i - 2], y[i - 2]) 
            - 19 * f(t[i - 3], y[i - 3])
        )

        return y_step_i

    
class MilneSimpson(ODESolver):
    """MilneSimpson method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        # Milne as predictor
        y[i + 1, :] = y[i - 3] + (4 * h / 3) * (
            2 * f(t[i], y[i, :]) - f(t[i - 1], y[i - 1]) 
            + 2 * f(t[i - 2], y[i - 2])
        )
        # Hamming as corrector
        y_step_i = (9 * y[i, :] - y[i - 2]) / 8 + (3 * h / 8) * (
            f(t[i + 1], y[i + 1, :]) + 2 * f(t[i], y[i, :]) 
            - f(t[i - 1], y[i - 1])
        )

        return y_step_i
    

class ModifiedMilneSimpson(ODESolver):
    """Modified Milne-Simpson method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        # Milne as predictor
        y[i + 1, :] = y[i - 3] + (4 * h / 3) * (
            2 * f(t[i], y[i, :]) - f(t[i - 1], y[i - 1]) 
            + 2 * f(t[i - 2], y[i - 2])
        )
        # Simpson as corrector
        y_step_i = y[i - 1] + (h / 3) * (
            f(t[i + 1], y[i + 1, :]) + 4 * f(t[i], y[i, :]) 
            + f(t[i - 1], y[i - 1])
        )

        return y_step_i
    
class Hammings(ODESolver):
    """Hammings method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        pk = y[i - 3] + (4 * h / 3) * (
            2 * f(t[i], y[i]) - f(t[i - 1], y[i - 1]) + 2 * f(t[i - 2], y[i - 2])
        )
        y_step_i = (9 * y[i] - y[i - 2]) / 8 + (3 * h / 8) * (
            f(t[i + 1], pk) + 2 * f(t[i], y[i]) - f(t[i - 1], y[i - 1])
        )

        return y_step_i
    
class RKF45(ODESolver):
    """Fourth order Runge-Kutta-Fehlberg method"""
    def advance(self):
        f, t, y, hmin, hmax, tolerance = (
            self.f, self.t, self.y, self.hmin, self.hmax, self.tolerance
        )
        t, b = self.time_span
        y = y[0]
        h = hmax
        T, Y = [t], [y]
        a2, b2 = 1 / 4, 1 / 4
        a3, b3, c3 = 3 / 8, 3 / 32, 9 / 32
        a4, b4, c4, d4 = 12 / 13, 1932 / 2197, -7200 / 2197, 7296 / 2197
        a5, b5, c5, d5, e5 = 1, 439 / 216, -8, 3680 / 513, -845 / 4104
        a6, b6, c6, d6, e6, f6 = 1 / 2, -8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40
        r1, r3, r4, r5, r6 = 1 / 360, -128 / 4275, -2197 / 75240, 1 / 50, 2 / 55
        n1, n3, n4, n5 = 25 / 216, 1408 / 2565, 2197 / 4104, -1 / 5
        flag = True
        while flag:
            k1 = h * f(t, y)
            k2 = h * f(t + a2 * h, y + b2 * k1)
            k3 = h * f(t + a3 * h, y + b3 * k1 + c3 * k2)
            k4 = h * f(t + a4 * h, y + b4 * k1 + c4 * k2 + d4 * k3)
            k5 = h * f(t + a5 * h, y + b5 * k1 + c5 * k2 + d5 * k3 + e5 * k4)
            k6 = h * f(t + a6 * h, y + b6 * k1 + c6 * k2 + d6 * k3 + e6 * k4 + f6 * k5)

            R = (1 / h) * abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6)
            if R <= tolerance:
                t = t + h
                y = y + n1 * k1 + n3 * k3 + n4 * k4 + n5 * k5
                Y.append(y)
                T.append(t)

            delta = 0.84 * (tolerance / R) ** 0.25
            if delta <= 0.1:
                h = 0.1 * h
            elif delta >= 4:
                h = 4 * h
            else:
                h = delta * h

            if h > hmax:
                h = hmax

            if t >= b:
                flag = False
            elif t + h > b:
                h = b - t
            elif h < hmin:
                flag = False
        
        result = flatten(T), asfarray(Y).reshape(-1, 1)
            
        return result


class RKF54(ODESolver):
    """Runge-Kutta-Fehlberg-54 method"""
    def advance(self):
        f, t, y, hmin, hmax, tolerance = (
            self.f, self.t, self.y, self.hmin, self.hmax, self.tolerance
        )
        t, b = self.time_span
        j = 0
        T, Y = [t], [y[0]]
        a2, b2 = 1 / 4, 1 / 4
        a3, b3, c3 = 3 / 8, 3 / 32, 9 / 32
        a4, b4, c4, d4 = 12 / 13, 1932 / 2197, -7200 / 2197, 7296 / 2197
        a5, b5, c5, d5, e5 = 1, 439 / 216, -8, 3680 / 513, -845 / 4104
        a6, b6, c6, d6, e6, f6 = 1 / 2, -8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40
        r1, r3, r4, r5, r6 = 1 / 360, -128 / 4275, -2197 / 75240, 1 / 50, 2 / 55
        n1, n3, n4, n5 = 25 / 216, 1408 / 2565, 2197 / 4104, -1 / 5
        h = 0.8 * hmin + 0.2 * hmax
        br = b - tolerance * abs(b)
        while T[j] < b:
            if T[j] + h > br:
                h = b - T[j]
            tj = T[j]
            yj = Y[j]
            k1 = h * f(tj, yj)
            k2 = h * f(tj + a2 * h, yj + b2 * k1)
            k3 = h * f(tj + a3 * h, yj + b3 * k1 + c3 * k2)
            k4 = h * f(tj + a4 * h, yj + b4 * k1 + c4 * k2 + d4 * k3)
            k5 = h * f(tj + a5 * h, yj + b5 * k1 + c5 * k2 + d5 * k3 + e5 * k4)
            k6 = h * f(tj + a6 * h, yj + b6 * k1 + c6 * k2 + d6 * k3 + e6 * k4 + f6 * k5)
            err = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6)
            ynew = yj + n1 * k1 + n3 * k3 + n4 * k4 + n5 * k5
            if err < tolerance or h < 2 * hmin:
                Y.append(ynew)
                if tj + h > br:
                    T.append(b)
                else:
                    T.append(tj + h)
                j += 1
            if err == 0:
                s = 0
            else:
                s = (tolerance * h / (2 * err)) ** 0.25
                
            if s < 0.1:
                s = 0.1
            if s > 4:
                s = 4
            h = s * h
            if h > hmax:
                h = hmax
            
            if h < hmin:
                h = hmin
        
        result = flatten(T), asfarray(Y).reshape(-1, 1)
            
        return result
            
    
class RKV(ODESolver):
    """
    Fourth order Runge-Kutta-Verner method
    err = 3 / 40 * k1 + 875 / 2244 * k3 + 23 / 72 * k4 + 264 / 1955 * k5 + 125 / 11592 * k7 + 43 / 616 * k8
    y = 13 / 160 * k1 + 2375 / 5984 * k3 + 5 / 16 * k4 + 12 / 85 * k5 + 3 / 44 * k6
    """
    def advance(self):
        f, t, y, hmin, hmax, tolerance = (
            self.f, self.t, self.y, self.hmin, self.hmax, self.tolerance
        )
        t, b = self.time_span
        y = y[0]
        h = hmax
        T, Y = [t], [y]
        a2, b2 = 1 / 4, 1 / 4
        a3, b3, c3 = 3 / 8, 3 / 32, 9 / 32
        a4, b4, c4, d4 = 12 / 13, 1932 / 2197, -7200 / 2197, 7296 / 2197
        a5, b5, c5, d5, e5 = 1, 439 / 216, -8, 3680 / 513, -845 / 4104
        a6, b6, c6, d6, e6, f6 = 1 / 2, -8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40
        r1, r3, r4, r5, r6 = 1 / 360, -128 / 4275, -2197 / 75240, 1 / 50, 2 / 55
        n1, n3, n4, n5 = 25 / 216, 1408 / 2565, 2197 / 4104, -1 / 5
        flag = True
        while flag:
            k1 = h * f(t, y)
            k2 = h * f(t + a2 * h, y + b2 * k1)
            k3 = h * f(t + a3 * h, y + b3 * k1 + c3 * k2)
            k4 = h * f(t + a4 * h, y + b4 * k1 + c4 * k2 + d4 * k3)
            k5 = h * f(t + a5 * h, y + b5 * k1 + c5 * k2 + d5 * k3 + e5 * k4)
            k6 = h * f(t + a6 * h, y + b6 * k1 + c6 * k2 + d6 * k3 + e6 * k4 + f6 * k5)

            R = (1 / h) * abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6)
            if R <= tolerance:
                t = t + h
                y = y + n1 * k1 + n3 * k3 + n4 * k4 + n5 * k5
                Y.append(y)
                T.append(t)

            delta = 0.84 * (tolerance / R) ** 0.25
            if delta <= 0.1:
                h = 0.1 * h
            elif delta >= 4:
                h = 4 * h
            else:
                h = delta * h

            if h > hmax:
                h = hmax

            if t >= b:
                flag = False
            elif t + h > b:
                h = b - t
            elif h < hmin:
                flag = False
        
        result = flatten(T), asfarray(Y).reshape(-1, 1)
            
        return result

    
class AdamsVariableStep(ODESolver):
    """Adams variable step method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        raise TypeError('This function is still under development')


class Extrapolation(ODESolver):
    """Extrapolation method"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        raise TypeError('This function is still under development')

    
class TrapezoidalNewton(ODESolver):
    """Trapezoidal method with Newton approximation"""
    def advance(self):
        f, t, y, i = self.f, self.t, self.y, self.i
        h = t[i + 1] - t[i]
        raise TypeError('This function is still under development')


def maths_transform_case(ftn_str: str, upper: bool = True) -> str:
    """
    Change case of mathematics functions to facilitate replacement of 
    symbolic variables.

    Parameters
    ----------
    ftn_str : str
        The string containing mathematical functions.
    upper : bool, optional (default=True)
        If `True`, converts functions to uppercase; if False, converts functions 
        to lowercase. Defaults to True.

    Returns
    -------
    ftn_str : str
        The modified string with case-transformed mathematical functions.
    """
    ftns_lower = [
        'acos', 'acosh', 'acot', 'acoth', 'acsc', 'acsch', 'asec', 'asech',
        'asin', 'asinh', 'atan', 'atan2', 'atanh', 'cos', 'cosh', 'cot', 
        'coth', 'csc', 'csch', 'exp', 'ln', 'log', 'sec', 'sech', 'sin',
        'sinh', 'sqrt', 'tan', 'tanh'
    ]
    ftn_str = str(ftn_str)
    upper = ValidateArgs.check_boolean(user_input=upper, default=True)
    ftns_upper = [ftn_lower.upper() for ftn_lower in ftns_lower]
    if upper:
        for ftn in ftns_lower:
            ftn_str = ftn_str.replace(ftn, ftn.upper())
    else:
        for ftn in ftns_upper:
            ftn_str = ftn_str.replace(ftn, ftn.lower())

    return ftn_str


def _solve_symbolic(fs_symbolic: str, method: str) -> Float:
    """
    Solve a symbolic expression for a given method.

    Parameters
    ----------
    fs_symbolic : str 
        The symbolic expression to solve.
    method : str 
        The method used for solving the symbolic expression.

    Returns
    -------
    y_step_i : float
        The solution to the symbolic expression converted to float.

    Raises
    ------
    ComputationError
        If an error occurs during computation.
    Exception
        If the solution of the expression is non-numeric.
    """
    fs_symbolic = maths_transform_case(ftn_str=fs_symbolic, upper=False)
    try:
        y_step_i = float(solve(sympify(fs_symbolic))[0])
    except Exception as e:
        raise ComputationError(par_name=f'{method} ODE', error_message=e)
    
    if isinstance(y_step_i, (Integer, Float)): # it is correct this way
        raise Exception(
            f"Expected solution to be numeric but got a non-numeric "
            f"value: '{y_step_i}'"
        )
    
    return y_step_i


def ivps(
    method: Literal[
        'taylor1', 'taylor2', 'taylor3', 'taylor4', 'taylor5', 'taylor6', 'taylor7', 'taylor8', 'taylor9',
        'feuler', 'meuler', 'beuler',
        'rkmidpoint', 'rkmeuler', 'ralston2', 'heun3', 'nystrom3', 'rk3', 'rk4', 'rk38', 'rkmersen', 'rk5',
        'rkbeuler', 'rktrapezoidal', 'rk1stage', 'rk2stage', 'rk3stage',
        'ab2', 'ab3', 'ab4', 'ab5',
        'am2', 'am3', 'am4',
        'eheun', 'abm2','abm3','abm4', 'abm5', 'hamming', 'msimpson', 'mmsimpson',
        'rkf45', 'rkf54', 'rkv', 'adamsvariablestep', 'extrapolation', 'tnewton'
    ] = 'rk4',
    odeqtn: str | list | Callable = '-1.2 * y + 7 * exp(-0.3 * t)',
    exactsol: str | Expr | Callable | None = None,
    derivs: list[str] | None = None,
    vars: list[str] = ['t', 'y'],
    start_values_or_method: list[float] | Literal['feuler', 'meuler', 'heun3', 'rk4'] = 'rk4',
    time_span: list[float] = [],
    y0: int | float = None,
    stepsize: float | list[float] | None = None,
    nsteps: int = 10,
    tolerance: float = 1e-6,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the specified method.

    Parameters
    ----------
    method : str, (default='rk4')
        The method to be used to integrate the ODE numerically.
        ===============================================================
        method                 Description  
        ===============================================================
        'taylorn' ......... Taylor order n where n = 1, 2, 3, ..., 9
        'feuler' .......... Explicit Euler
        'meuler' .......... Modified Euler
        'beuler' .......... Implicit Euler
        'rkmidpoint' ...... Midpoint Runge-Kutta
        'rkmeuler' ........ Modified Euler Runge-Kutta
        'ralston2' ........ Second order Heun Runge-Kutta
        'heun3' ........... Third order Heun Runge-Kutta
        'nystrom3' ........ Third order Nystrom Runge-Kutta
        'rk3  ............. Classical third order Runge-Kutta
        'rk4' ............. Classical fourth order Runge-Kutta
        'rk38' ............ Fourth order Runge-Kutta 3/8
        'rkmersen' ........ Fourth order Runge-Kutta-Mersen
        'rk5' ............. Classical fifth order Runge-Kutta
        'rkbeuler' ........ Implicit backward Euler Runge-Kutta
        'rktrapezoidal' .......... Implicit implicit Trapezoidal Runge-Kutta
        'rk1stage' ........ Implicit 1-stage Gauss-Legendre Runge-Kutta
        'rk2stage' ........ Implicit 2-stage Gauss-Legendre Runge-Kutta
        'rk3stage' ........ Implicit 3-stage Gauss-Legendre Runge-Kutta
        'rkf45' ............ Adaptive Runge-Kutta-fehlberg 45
        'rkf54' ............ Adaptive Runge-Kutta-fehlberg 54
        'rkv' ............. Adaptive Runge-Kutta-Verner
        'ab2' ............. Explicit Adam-Bashforth 2 step
        'ab3' ............. Explicit Adam-Bashforth 3 step
        'ab4' ............. Explicit Adam-Bashforth 4 step
        'ab5' ............. Explicit Adam-Bashforth 5 step
        'am2' ............. Implicit Adam-Moulton 2 step
        'am3' ............. Implicit Adam-Moulton 3 step
        'am4' ............. Implicit Adam-Moulton 4 step
        'eheun' ........... Euler-Heun PC
        'abm2' ............ Adam-Bashforth-Moulton 2 step PC
        'abm3' ............ Adam-Bashforth-Moulton 3 step PC
        'abm4' ............ Adam-Bashforth-Moulton 4 step PC
        'abm5' ............ Adam-Bashforth-Moulton 5 step PC
        'ms' .............. Milne-Simpson PC
        'mms'.............. Modified Milne-Simpson PC
        'hamming' ......... Hammings PC
        ===============================================================
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    start_values_or_method : list_like | {'feuler', 'meuler', 'heun3', 'rk4'}, optional (default='rk4')
        Start values (as list-like) or IVP method to be used to
        approximate the initial values.
    stepsize : float | list-like, optional (default=None)
        A float representing the interval (difference between two
        consecutive time points) or list-like of length 2 for adaptive
        methods where:
        - stepsize[1]: Minimum value for RK-Fehlberg and RK-Verner
                       methods `(hmin)`.
        - stepsize[2]: Maximum value for RK-Fehlberg and RK-Verner
                       methods `(hmax)`.
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> dydx = ['y - t ** 2 - 2 * t + 1', 'y - t ** 2 - 2 * t - 1',
    ... 'y - t ** 2 - 2 * t - 1', 'y - t ** 2 - 2 * t - 1']
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5

    ## Taylor orders 2, 3, 4 and 5
    
    >>> table = []
    >>> for order in range(2, 6):
    ...     result = stm.ivps(method = f'taylor{order}', odeqtn=f,
    ...         derivs=dydx[:(order - 1)], exactsol=ft, time_span=[a, b],
    ...         y0=y0, decimal_points=14)
    
    ...     table.append(result.answer)
    >>> row_names = [f'Taylor order {n + 2}' for n in range(len(table))]
    >>> dframe = pd.DataFrame(table, index = row_names, columns = ['Solution'])
    >>> dframe

    ## Fourth order Runge-Kutta

    >>> result = stm.ivps(method='rk4', odeqtn=f,
    ... exactsol=ft, time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051270
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12720268494794   2.12722953575377   0.00002685080582
    5        1.0   2.64082269272875   2.64085908577048   0.00003639304173
    6        1.2   3.17989417023223   3.17994153863173   0.00004736839950
    7        1.4   3.73234007285498   3.73240001657766   0.00005994372268
    8        1.6   4.28340949831841   4.28348378780244   0.00007428948404
    9        1.8   4.81508569457943   4.81517626779353   0.00009057321409
    10       2.0   5.30536300069265   5.30547195053467   0.00010894984202
    
    Answer = 5.30536300069265

    ## Adams-Moulton PC with RK4 approximated start values

    >>> result = stm.ivps(method='am4', odeqtn=f,
    ... exactsol=ft, time_span=[a, b], y0=y0, start_values_or_method='rk4',
    ... decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051270
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12720569067661   2.12722953575377   0.00002384507716
    5        1.0   2.64082874144856   2.64085908577048   0.00003034432191
    6        1.2   3.17990290225611   3.17994153863173   0.00003863637561
    7        1.4   3.73235091672886   3.73240001657766   0.00004909984880
    8        1.6   4.28342148841619   4.28348378780244   0.00006229938625
    9        1.8   4.81509733035245   4.81517626779353   0.00007893744107
    10       2.0   5.30537206144074   5.30547195053467   0.00009988909394
    
    Answer = 5.30537206144074

    ## Adam-Bashforth PC with exact start values

    >>> start_values = (0.5, 0.829298620919915, 1.2140876511793646,
    ... 1.648940599804746)
    >>> result = stm.ivps(method='ab4', odeqtn=f,
    ... exactsol=ft, time_span=[a, b], y0=y0,
    ... start_values_or_method=start_values,decimal_points=12)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0     0.500000000000     0.500000000000     0.000000000000
    1        0.2     0.829298620920     0.829298620920     0.000000000000
    2        0.4     1.214087651179     1.214087651179     0.000000000000
    3        0.6     1.648940599805     1.648940599805     0.000000000000
    4        0.8     2.127312354336     2.127229535754     0.000082818582
    5        1.0     2.641081017714     2.640859085770     0.000221931943
    6        1.2     3.180348021052     3.179941538632     0.000406482420
    7        1.4     3.733060127926     3.732400016578     0.000660111349
    8        1.6     4.284493130095     4.283483787802     0.001009342293
    9        1.8     4.816657481988     4.815176267794     0.001481214194
    10       2.0     5.307583810133     5.305471950535     0.002111859599
    
    Answer = 5.307583810133
    
    ## Runge-Kutta-Felhberg

    >>> result = stm.ivps(method='rkf45', odeqtn=f,
    ... exactsol=ft, time_span=[a, b], y0=y0, tolerance=1e-5,
    ... stepsize=[0.01, 0.25], decimal_points=12)
    
               Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0  0.00000000000000   0.50000000000000   0.50000000000000   0.00000000000000
    1  0.25000000000000   0.92048860207582   0.92048729165613   0.00000131041969
    2  0.48655220228477   1.39649101428839   1.39648844281524   0.00000257147315
    3  0.72933319984230   1.95374878715415   1.95374461367237   0.00000417348178
    4  0.97933319984230   2.58642601474198   2.58641982885096   0.00000618589102
    5  1.22933319984230   3.26046051047871   3.26045200576709   0.00000850471163
    6  1.47933319984230   3.95209553728388   3.95208439562523   0.00001114165865
    7  1.72933319984230   4.63082681953757   4.63081272915408   0.00001409038349
    8  1.97933319984230   5.25748606455951   5.25746874929178   0.00001731526773
    9  2.00000000000000   5.30548962736878   5.30547195053467   0.00001767683411
    
    Answer = 5.30548962736878
    """
    method_class_map = {
        f'taylor{method[-1]}': TaylorN,
        'feuler': ForwardEuler,
        'meuler': ModifiedEuler,
        'beuler': BackwardEuler,
        'rkmidpoint': MidpointRK,
        'rkmeuler': ModifiedEulerRK,
        'ralston2': Ralston2,
        'heun3': Heun3,
        'nystrom3': Nystrom3,
        'rk3': RK3,
        'rk4': RK4,
        'rk38': RK38,
        'rkmersen': RKMersen,
        'rk5': RK5,
        'rkbeuler': BackwardEulerRK,
        'rktrapezoidal': TrapezoidalRK,
        'rk1stage': GaussLegendre1Stage,
        'rk2stage': GaussLegendre2Stage,
        'rk3stage': GaussLegendre3Stage,
        'ab2': AdamBashforth2Step,
        'ab3': AdamBashforth3Step,
        'ab4': AdamBashforth4Step,
        'ab5': AdamBashforth5Step,
        'am2': AdamMoulton2Step,
        'am3': AdamMoulton3Step,
        'am4': AdamMoulton4Step,
        'eheun': EulerHeun,
        'abm2': AdamBashforthMoulton2Step,
        'abm3': AdamBashforthMoulton3Step,
        'abm4': AdamBashforth4Step,
        'abm5': AdamBashforth5Step,
        'adamsvariablestep': AdamsVariableStep,
        'msimpson': MilneSimpson,
        'mmsimpson': ModifiedMilneSimpson,
        'hamming': Hammings,
        'rkf45': RKF45,
        'rkf54': RKF54,
        'rkv': RKV,
        'extrapolation': Extrapolation,
        'tnewton': TrapezoidalNewton
    }
    # validation of `method` must be repeated here
    method = ValidateArgs.check_member(
        par_name='method', 
        valid_items=VALID_IVP_METHODS, 
        user_input=method.lower()
    )
    compute_class = method_class_map.get(method)
    solver = compute_class(
        method,
        odeqtn,
        exactsol,
        vars,
        derivs,
        start_values_or_method,
        time_span,
        y0,
        stepsize,
        nsteps,
        maxit,
        tolerance,
        show_iters,
        auto_display,
        decimal_points
    )

    return solver.compute()

def ivps_taylor(
    order: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9],
    odeqtn: str | Expr | Callable,
    derivs: list[str] | None,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the Taylor method.

    Parameters
    ----------
    order : int
        An integer between 1 and 9 representing the Taylor order.
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    derivs : list_like
        A list of derivatives of the ODE equation.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    y0 : {int, float, list_like}
        Initial value(s) of the ODE or system of ODEs.
    exactsol : {None, str, symbolic, Callable}, optional (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> dydx = ['y - t^2 - 2*t + 1', 'y - t^2 - 2*t - 1']
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_taylor(order=3, odeqtn=f,
    ... derivs=dydx, exactsol=ft, time_span=[a, b], y0=y0,
    ... decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82933333333333   0.82929862091992   0.00003471241342
    2        0.4   1.21417244444444   1.21408765117936   0.00008479326508
    3        0.6   1.64909594548148   1.64894059980475   0.00015534567674
    4        0.8   2.12748251474805   2.12722953575377   0.00025297899428
    5        1.0   2.64124531134562   2.64085908577048   0.00038622557514
    6        1.2   3.18050760692345   3.17994153863173   0.00056606829172
    7        1.4   3.73320662392250   3.73240001657766   0.00080660734484
    8        1.6   4.28460969001735   4.28348378780244   0.00112590221491
    9        1.8   4.81672330140786   4.81517626779352   0.00154703361433
    10       2.0   5.30757139211947   5.30547195053468   0.00209944158479

    Answer = 5.30757139211947

    ## Taylor orders 2, 3, 4 and 5
    
    >>> dydx = ['y - t^2 - 2*t + 1', 'y - t^2 - 2*t - 1',
    ... 'y - t^2 - 2*t - 1', 'y - t^2 - 2*t - 1']
    >>> table = []
    >>> for order in range(2, 6):
    ...     result = stm.ivps_taylor(order=order, odeqtn=f,
    ...         derivs=dydx[:(order - 1)], exactsol=ft,
    ...         time_span=[a, b], y0=y0, decimal_points=14)
    ...     table.append(result.answer)
    >>> row_names = [f'Taylor order {n + 2}' for n in range(len(table))]
    >>> dframe = pd.DataFrame(table, index = row_names, columns = ['Solution'])
    >>> dframe
    
    (some output omitted)
    
                            Solution
    Taylor order 2  5.34768429228604
    Taylor order 3  5.30757139211947
    Taylor order 4  5.30555537917027
    Taylor order 5  5.30547471805094
    """
    result = ivps(
        method=f'taylor{order}',
        odeqtn=odeqtn,
        derivs=derivs,
        time_span=time_span,
        y0=y0,
        exactsol=exactsol,
        vars=vars,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_feuler(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
):
    """
    Solve an ordinary differential equation using the Forward Euler method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    y0 : {int, float, list_like}
        Initial value(s) of the ODE or system of ODEs.
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float | list-like, optional (default=None)
        A float representing the interval (difference between two
        consecutive time points) or list-like of length 2 for adaptive
        methods where:
        - stepsize[1]: Minimum value for RK-Fehlberg and RK-Verner
                       methods `(hmin)`.
        - stepsize[2]: Maximum value for RK-Fehlberg and RK-Verner
                       methods `(hmax)`.
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_feuler(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0      0.50000000000   0.50000000000000   0.00000000000000
    1        0.2      0.80000000000   0.82929862091992   0.02929862091991
    2        0.4      1.15200000000   1.21408765117936   0.06208765117936
    3        0.6      1.55040000000   1.64894059980475   0.09854059980475
    4        0.8      1.98848000000   2.12722953575377   0.13874953575377
    5        1.0      2.45817600000   2.64085908577048   0.18268308577048
    6        1.2      2.94981120000   3.17994153863173   0.23013033863173
    7        1.4      3.45177344000   3.73240001657766   0.28062657657766
    8        1.6      3.95012812800   4.28348378780244   0.33335565980244
    9        1.8      4.42815375360   4.81517626779352   0.38702251419352
    10       2.0      4.86578450432   5.30547195053468   0.43968744621467
    
    Answer = 4.86578450432
    """
    result = ivps(
        method='feuler',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_meuler(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
):
    """
    Solve an ordinary differential equation using the Modified Euler method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    y0 : {int, float, list_like}
        Initial value(s) of the ODE or system of ODEs.
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_meuler(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82600000000000   0.82929862091992   0.00329862091991
    2        0.4   1.20692000000000   1.21408765117936   0.00716765117936
    3        0.6   1.63724240000000   1.64894059980475   0.01169819980475
    4        0.8   2.11023572800000   2.12722953575377   0.01699380775377
    5        1.0   2.61768758816000   2.64085908577048   0.02317149761048
    6        1.2   3.14957885755520   3.17994153863173   0.03036268107653
    7        1.4   3.69368620621734   3.73240001657766   0.03871381036032
    8        1.6   4.23509717158516   4.28348378780244   0.04838661621728
    9        1.8   4.75561854933390   4.81517626779352   0.05955771845963
    10       2.0   5.23305463018735   5.30547195053468   0.07241732034732
    
    Answer = 5.23305463018735
    """
    result = ivps(
        method='meuler',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_beuler(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
):
    """
    Solve an ordinary differential equation using the Backward Euler method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    y0 : {int, float, list_like}
        Initial value(s) of the ODE or system of ODEs.
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_beuler(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.86500000000000   0.82929862091992   0.03570137908008
    2        0.4   1.29125000000000   1.21408765117936   0.07716234882064
    3        0.6   1.77406250000000   1.64894059980475   0.12512190019525
    4        0.8   2.30757812500000   2.12722953575377   0.18034858924623
    5        1.0   2.88447265625000   2.64085908577048   0.24361357047952
    6        1.2   3.49559082031250   3.17994153863173   0.31564928168077
    7        1.4   4.12948852539062   3.73240001657766   0.39708850881296
    8        1.6   4.77186065673828   4.28348378780244   0.48837686893584
    9        1.8   5.40482582092285   4.81517626779352   0.58964955312933
    10       2.0   6.00603227615356   5.30547195053468   0.70056032561889
    
    Answer = 6.00603227615356
    """
    result = ivps(
        method='beuler',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_rkmidpoint(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
):
    """
    Solve an ordinary differential equation using the Runge-Kutta 
    midpoint method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_rkmidpoint(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82800000000000   0.82929862091992   0.00129862091991
    2        0.4   1.21136000000000   1.21408765117936   0.00272765117936
    3        0.6   1.64465920000000   1.64894059980475   0.00428139980475
    4        0.8   2.12128422400000   2.12722953575377   0.00594531175377
    5        1.0   2.63316675328000   2.64085908577048   0.00769233249048
    6        1.2   3.17046343900160   3.17994153863173   0.00947809963013
    7        1.4   3.72116539558195   3.73240001657766   0.01123462099571
    8        1.6   4.27062178260998   4.28348378780244   0.01286200519246
    9        1.8   4.80095857478418   4.81517626779352   0.01421769300935
    10       2.0   5.29036946123670   5.30547195053468   0.01510248929798
    
    Answer = 5.2903694612367
    """
    result = ivps(
        method='rkmidpoint',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_rkmeuler(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the Runge-Kutta 
    modified Euler method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_rkmeuler(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82600000000000   0.82929862091992   0.00329862091991
    2        0.4   1.20692000000000   1.21408765117936   0.00716765117936
    3        0.6   1.63724240000000   1.64894059980475   0.01169819980475
    4        0.8   2.11023572800000   2.12722953575377   0.01699380775377
    5        1.0   2.61768758816000   2.64085908577048   0.02317149761048
    6        1.2   3.14957885755520   3.17994153863173   0.03036268107653
    7        1.4   3.69368620621734   3.73240001657766   0.03871381036032
    8        1.6   4.23509717158516   4.28348378780244   0.04838661621728
    9        1.8   4.75561854933390   4.81517626779352   0.05955771845963
    10       2.0   5.23305463018735   5.30547195053468   0.07241732034732

    Answer = 5.23305463018735
    """
    result = ivps(
        method='rkmeuler',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_ralston2(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the Second order 
    Ralston method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    y0 : {int, float, list_like}
        Initial value(s) of the ODE or system of ODEs.
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_ralston2(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82700000000000   0.82929862091992   0.00229862091992
    2        0.4   1.20914000000000   1.21408765117936   0.00494765117936
    3        0.6   1.64095080000000   1.64894059980475   0.00798979980475
    4        0.8   2.11575997600000   2.12722953575377   0.01146955975377
    5        1.0   2.62542717072000   2.64085908577048   0.01543191505048
    6        1.2   3.16002114827840   3.17994153863173   0.01992039035333
    7        1.4   3.70742580089965   3.73240001657766   0.02497421567802
    8        1.6   4.25285947709757   4.28348378780244   0.03062431070487
    9        1.8   4.77828856205904   4.81517626779352   0.03688770573449
    10       2.0   5.26171204571202   5.30547195053468   0.04375990482265
    
    Answer = 5.26171204571202
    """
    result = ivps(
        method='ralston2',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_heun3(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the Third order Heun 
    method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_heun3(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82924444444444   0.82929862091992   0.00005417647547
    2        0.4   1.21397499259259   1.21408765117936   0.00011265858677
    3        0.6   1.64876590206420   1.64894059980475   0.00017469774055
    4        0.8   2.12699053283218   2.12722953575377   0.00023900292158
    5        1.0   2.64055554854349   2.64085908577048   0.00030353722699
    6        1.2   3.17957628773222   3.17994153863173   0.00036525089950
    7        1.4   3.73198028386140   3.73240001657766   0.00041973271626
    8        1.6   4.28302303113383   4.28348378780244   0.00046075666861
    9        1.8   4.81469657313590   4.81517626779352   0.00047969465763
    10       2.0   5.30500719243442   5.30547195053468   0.00046475810025
    
    Answer = 5.30500719243442
    """
    result = ivps(
        method='heun3',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_nystrom3(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the Nystrom3 method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_nystrom3(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82915555555556   0.82929862091992   0.00014306536436
    2        0.4   1.21377754074074   1.21408765117936   0.00031011043862
    3        0.6   1.64843585864691   1.64894059980475   0.00050474115783
    4        0.8   2.12649855091632   2.12722953575377   0.00073098483745
    5        1.0   2.63986578574135   2.64085908577048   0.00099330002912
    6        1.2   3.17864496854100   3.17994153863173   0.00129657009073
    7        1.4   3.73075394380029   3.73240001657766   0.00164607277737
    8        1.6   4.28143637225031   4.28348378780244   0.00204741555213
    9        1.8   4.81266984486394   4.81517626779352   0.00250642292959
    10       2.0   5.30244299274938   5.30547195053468   0.00302895778530

    Answer = 5.30244299274938
    """
    result = ivps(
        method='nystrom3',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_rk3(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the Third order 
    Runge-Kutta method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_rk3(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82920000000000   0.82929862091992   0.00009862091991
    2        0.4   1.21387626666667   1.21408765117936   0.00021138451270
    3        0.6   1.64860088035556   1.64894059980475   0.00033971944919
    4        0.8   2.12674454187425   2.12722953575377   0.00048499387951
    5        1.0   2.64021066714242   2.64085908577048   0.00064841862806
    6        1.2   3.17911062813661   3.17994153863173   0.00083091049512
    7        1.4   3.73136711383084   3.73240001657766   0.00103290274682
    8        1.6   4.28222970169207   4.28348378780244   0.00125408611037
    9        1.8   4.81368320899992   4.81517626779352   0.00149305879361
    10       2.0   5.30372509259190   5.30547195053468   0.00174685794278
    
    Answer = 5.3037250925919
    """
    result = ivps(
        method='rk3',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_rk4(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the Fourth order 
    Runge-Kutta method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_rk4(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051270
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12720268494794   2.12722953575377   0.00002685080582
    5        1.0   2.64082269272875   2.64085908577048   0.00003639304173
    6        1.2   3.17989417023223   3.17994153863173   0.00004736839950
    7        1.4   3.73234007285498   3.73240001657766   0.00005994372268
    8        1.6   4.28340949831841   4.28348378780244   0.00007428948404
    9        1.8   4.81508569457943   4.81517626779352   0.00009057321409
    10       2.0   5.30536300069265   5.30547195053468   0.00010894984202
    
    Answer = 5.30536300069265
    """
    
    result = ivps(
        method='rk4',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_rk38(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the Fourth order 
    Runge-Kutta-38 method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_rk38(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929555555556   0.82929862091992   0.00000306536436
    2        0.4   1.21408114711111   1.21408765117936   0.00000650406825
    3        0.6   1.64893026863707   1.64894059980475   0.00001033116768
    4        0.8   2.12721498566887   2.12722953575377   0.00001455008490
    5        1.0   2.64083993905151   2.64085908577048   0.00001914671897
    6        1.2   3.17991745711307   3.17994153863173   0.00002408151865
    7        1.4   3.73237073767346   3.73240001657766   0.00002927890420
    8        1.6   4.28344917454992   4.28348378780244   0.00003461325252
    9        1.8   4.81513637735083   4.81517626779353   0.00003989044269
    10       2.0   5.30542712685186   5.30547195053467   0.00004482368281
    
    Answer = 5.30542712685186
    """
    result = ivps(
        method='rk38',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_rkmersen(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the  
    Fourth order Runge-Kutta-Mersen method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_rkmersen(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82243540740741   0.82929862091992   0.00686321351251
    2        0.4   1.19956525100214   1.21408765117936   0.01452240017722
    3        0.6   1.62591672461572   1.64894059980475   0.02302387518902
    4        0.8   2.09483342437052   2.12722953575377   0.03239611138324
    5        1.0   2.59821937331830   2.64085908577048   0.04263971245218
    6        1.2   3.12622768657172   3.17994153863173   0.05371385206000
    7        1.4   3.66688190438990   3.73240001657766   0.06551811218776
    8        1.6   4.20561543139093   4.28348378780244   0.07786835641151
    9        1.8   4.72471137079478   4.81517626779352   0.09046489699875
    10       2.0   5.20262121224022   5.30547195053468   0.10285073829445
    
    Answer = 5.20262121224022
    """
    result = ivps(
        method='rkmersen',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_rk5(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    Fifth order Runge-Kutta method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_rk5(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929867833333   0.82929862091992   0.00000005741342
    2        0.4   1.21408777777601   1.21408765117936   0.00000012659665
    3        0.6   1.64894080975180   1.64894059980475   0.00000020994706
    4        0.8   2.12722984610043   2.12722953575377   0.00000031034666
    5        1.0   2.64085951702964   2.64085908577048   0.00000043125916
    6        1.2   3.17994211547800   3.17994153863173   0.00000057684627
    7        1.4   3.73240076868468   3.73240001657766   0.00000075210701
    8        1.6   4.28348475084692   4.28348378780244   0.00000096304448
    9        1.8   4.81517748465891   4.81517626779352   0.00000121686538
    10       2.0   5.30547347275343   5.30547195053468   0.00000152221876

    Answer = 5.30547347275343
    """
    result = ivps(
        method='rk5',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_rkbeuler(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the implicit 
    Runge-Kutta Backward Euler method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_rkbeuler(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.86500000000000   0.82929862091992   0.03570137908008
    2        0.4   1.29125000000000   1.21408765117936   0.07716234882064
    3        0.6   1.77406250000000   1.64894059980475   0.12512190019525
    4        0.8   2.30757812500000   2.12722953575377   0.18034858924623
    5        1.0   2.88447265625000   2.64085908577048   0.24361357047952
    6        1.2   3.49559082031250   3.17994153863173   0.31564928168077
    7        1.4   4.12948852539062   3.73240001657766   0.39708850881296
    8        1.6   4.77186065673828   4.28348378780244   0.48837686893584
    9        1.8   5.40482582092285   4.81517626779352   0.58964955312933
    10       2.0   6.00603227615356   5.30547195053468   0.70056032561889
    
    Answer = 6.00603227615356
    """
    result = ivps(
        method='rkbeuler',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_rktrapezoidal(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    implicit Runge-Kutta Trapezoidal method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_rktrapezoidal(odeqtn=f,
    ... exactsol=ft, time_span=[a, b], y0=y0, decimal_points=14)
    
    (To update)
    """
    result = ivps(
        method='rktrapezoidal',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_rk1stage(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    implicit Runge-Kutta 1 stage method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_rk1stage(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.83111111111111   0.82929862091992   0.00181249019120
    2        0.4   1.21802469135802   1.21408765117936   0.00393704017866
    3        0.6   1.65536351165981   1.64894059980475   0.00642291185506
    4        0.8   2.13655540313977   2.12722953575377   0.00932586738600
    5        1.0   2.65356771494860   2.64085908577048   0.01270862917812
    6        1.2   3.19658276271496   3.17994153863173   0.01664122408323
    7        1.4   3.75360115442939   3.73240001657766   0.02120113785173
    8        1.6   4.30995696652481   4.28348378780244   0.02647317872237
    9        1.8   4.84772518130811   4.81517626779352   0.03254891351458
    10       2.0   5.34499744382102   5.30547195053468   0.03952549328634

    Answer = 5.34499744382102
    """
    result = ivps(
        method='rk1stage',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def ivps_rk2stage(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    implicit Runge-Kutta 2 stage method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_rk2stage(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.83037037037037   0.82929862091992   0.00107174945046
    2        0.4   1.21637860082305   1.21408765117936   0.00229094964368
    3        0.6   1.65261088248743   1.64894059980475   0.00367028268268
    4        0.8   2.13245033785500   2.12722953575377   0.00522080210124
    5        1.0   2.64780967219315   2.64085908577048   0.00695058642267
    6        1.2   3.18880441416200   3.17994153863173   0.00886287553027
    7        1.4   3.74335354323504   3.73240001657766   0.01095352665737
    8        1.6   4.29669136765764   4.28348378780244   0.01320757985520
    9        1.8   4.83077093084082   4.81517626779352   0.01559466304729
    10       2.0   5.32353484139804   5.30547195053468   0.01806289086336

    Answer = 5.32353484139804
    """
    result = ivps(
        method='rk2stage',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_rk3stage(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    implicit Runge-Kutta 3-stage method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_rk3stage(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.83372778650138   0.82929862091992   0.00442916558146
    2        0.4   1.22470655914018   1.21408765117936   0.01061890796082
    3        0.6   1.66783212836188   1.64894059980475   0.01889152855713
    4        0.8   2.15685507910165   2.12722953575377   0.02962554334789
    5        1.0   2.68412381687328   2.64085908577048   0.04326473110280
    6        1.2   3.24026996118441   3.17994153863173   0.06032842255269
    7        1.4   3.81382315076495   3.73240001657766   0.08142313418728
    8        1.6   4.39073942275768   4.28348378780244   0.10725563495523
    9        1.8   4.95382377448657   4.81517626779352   0.13864750669304
    10       2.0   5.48202316557866   5.30547195053468   0.17655121504398
    
    Answer = 5.48202316557866
    """
    result = ivps(
        method='rk3stage',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_ab2(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    start_values_or_method: list[float] | Literal['feuler', 'meuler', 'heun3', 'rk4'] = 'rk4',
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    Adam-Bashforth 2 step method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    y0 : {int, float, list_like}
        Initial value(s) of the ODE or system of ODEs.
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    start_values_or_method : list_like | {'feuler', 'meuler', 'heun3', 'rk4'}, optional (default='rk4')
        Start values (as list-like) or IVP method to be used to
        approximate the initial values.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_ab2(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], start_values_or_method='rk4', y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21608133333333   1.21408765117936   0.00199368215397
    3        0.6   1.65397640000000   1.64894059980475   0.00503580019525
    4        0.8   2.13656118666667   2.12722953575377   0.00933165091290
    5        1.0   2.65613190266667   2.64085908577048   0.01527281689619
    6        1.2   3.20331535480000   3.17994153863173   0.02337381616827
    7        1.4   3.76669677097333   3.73240001657766   0.03429675439567
    8        1.6   4.33237426678533   4.28348378780244   0.04889047898289
    9        1.8   4.88341686972360   4.81517626779352   0.06824060193007
    10       2.0   5.39920450396215   5.30547195053468   0.09373255342747
    
    Answer = 5.39920450396215
    
    >>> start_values = [0.5, 0.82929333333333]
    >>> result = stm.ivps_ab2(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, start_values_or_method=start_values,
    ... decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758659
    2        0.4   1.21608133333333   1.21408765117936   0.00199368215396
    3        0.6   1.65397640000000   1.64894059980475   0.00503580019525
    4        0.8   2.13656118666666   2.12722953575377   0.00933165091289
    5        1.0   2.65613190266666   2.64085908577048   0.01527281689618
    6        1.2   3.20331535479999   3.17994153863173   0.02337381616826
    7        1.4   3.76669677097332   3.73240001657766   0.03429675439566
    8        1.6   4.33237426678532   4.28348378780244   0.04889047898288
    9        1.8   4.88341686972358   4.81517626779352   0.06824060193006
    10       2.0   5.39920450396213   5.30547195053468   0.09373255342745
    
    Answer = 5.39920450396213
    """
    result = ivps(
        method='ab2',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=start_values_or_method,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_ab3(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    start_values_or_method: list[float] | Literal['feuler', 'meuler', 'heun3', 'rk4'] = 'rk4',
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
):
    """
    Solve an ordinary differential equation using the 
    Adam-Bashforth 3 step method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    start_values_or_method : list_like | {'feuler', 'meuler', 'heun3', 'rk4'}, optional (default='rk4')
        Start values (as list-like) or IVP method to be used to
        approximate the initial values.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_ab3(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], start_values_or_method='rk4', y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051270
    3        0.6   1.64932720253333   1.64894059980475   0.00038660272859
    4        0.8   2.12825675177111   2.12722953575377   0.00102721601735
    5        1.0   2.64277427016337   2.64085908577048   0.00191518439289
    6        1.2   3.18307987346481   3.17994153863173   0.00313833483308
    7        1.4   3.73720874889702   3.73240001657766   0.00480873231935
    8        1.6   4.29054865889720   4.28348378780244   0.00706487109476
    9        1.8   4.82525996789066   4.81517626779352   0.01008370009714
    10       2.0   5.31956404228424   5.30547195053468   0.01409209174957

    Answer = 5.31956404228424
    
    >>> start_values = [0.5, 0.82929333333333, 1.21407621066667]
    >>> result = stm.ivps_ab3(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, start_values_or_method=start_values, 
    ... decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758659
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051269
    3        0.6   1.64932720253334   1.64894059980475   0.00038660272859
    4        0.8   2.12825675177112   2.12722953575377   0.00102721601735
    5        1.0   2.64277427016338   2.64085908577048   0.00191518439290
    6        1.2   3.18307987346482   3.17994153863173   0.00313833483309
    7        1.4   3.73720874889703   3.73240001657766   0.00480873231936
    8        1.6   4.29054865889722   4.28348378780244   0.00706487109477
    9        1.8   4.82525996789068   4.81517626779352   0.01008370009715
    10       2.0   5.31956404228426   5.30547195053468   0.01409209174959
    
    Answer = 5.31956404228426
    """
    result = ivps(
        method='ab3',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=start_values_or_method,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_ab4(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    start_values_or_method: list[float] | Literal['feuler', 'meuler', 'heun3', 'rk4'] = 'rk4',
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    Adams-Bashforth step 4 method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    y0 : {int, float, list_like}
        Initial value(s) of the ODE or system of ODEs.
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    start_values : list_like
        Starting values.
    start_method : {'feuler', 'meuler', 'heun3', 'rk4'}, optional (default='rk4')
        Method to be used to approximate the starting values if 
        `start_values` is not specified.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_ab4(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], start_values_or_method='rk4', y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051270
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12728924905233   2.12722953575377   0.00005971329857
    5        1.0   2.64105332811142   2.64085908577048   0.00019424234094
    6        1.2   3.18031412883292   3.17994153863173   0.00037259020119
    7        1.4   3.73301858540624   3.73240001657766   0.00061856882858
    8        1.6   4.28444240619668   4.28348378780244   0.00095861839424
    9        1.8   4.81659556132722   4.81517626779352   0.00141929353369
    10       2.0   5.30750818139328   5.30547195053468   0.00203623085860

    Answer = 5.30750818139328
    
    >>> start_values = [0.5, 0.82929333333333, 1.21407621066667, 1.6489220170416]
    >>> result = stm.ivps_ab4(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, start_values_or_method=start_values, 
    ... decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758659
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051269
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12728924905233   2.12722953575377   0.00005971329856
    5        1.0   2.64105332811142   2.64085908577048   0.00019424234094
    6        1.2   3.18031412883292   3.17994153863173   0.00037259020119
    7        1.4   3.73301858540624   3.73240001657766   0.00061856882857
    8        1.6   4.28444240619668   4.28348378780244   0.00095861839423
    9        1.8   4.81659556132721   4.81517626779352   0.00141929353368
    10       2.0   5.30750818139327   5.30547195053468   0.00203623085860
    
    Answer = 5.30750818139327
    """
    result = ivps(
        method='ab4',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=start_values_or_method,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_ab5(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    start_values_or_method: list[float] | Literal['feuler', 'meuler', 'heun3', 'rk4'] = 'rk4',
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    Adams-Bashforth step 5 method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    start_values_or_method : list_like | {'feuler', 'meuler', 'heun3', 'rk4'}, optional (default='rk4')
        Start values (as list-like) or IVP method to be used to
        approximate the initial values.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_ab5(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], start_values_or_method='rk4', y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051270
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12720268494794   2.12722953575377   0.00002685080582
    5        1.0   2.64084332085071   2.64085908577048   0.00001576491976
    6        1.2   3.17994955300343   3.17994153863173   0.00000801437170
    7        1.4   3.73243661722837   3.73240001657766   0.00003660065071
    8        1.6   4.28356197342805   4.28348378780244   0.00007818562561
    9        1.8   4.81531576484758   4.81517626779352   0.00013949705406
    10       2.0   5.30569478939270   5.30547195053468   0.00022283885803
    
    Answer = 5.3056947893927
    
    >>> start_values = [0.5, 0.82929333333333, 1.21407621066667,
    ... 1.6489220170416, 2.12720268494794]
    >>> result = stm.ivps_ab5(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, start_values_or_method=start_values, 
    ... decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758659
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051269
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12720268494794   2.12722953575377   0.00002685080583
    5        1.0   2.64084332085071   2.64085908577048   0.00001576491976
    6        1.2   3.17994955300343   3.17994153863173   0.00000801437170
    7        1.4   3.73243661722837   3.73240001657766   0.00003660065070
    8        1.6   4.28356197342805   4.28348378780244   0.00007818562561
    9        1.8   4.81531576484758   4.81517626779352   0.00013949705406
    10       2.0   5.30569478939270   5.30547195053468   0.00022283885802
    
    Answer = 5.3056947893927
    """
    result = ivps(
        method='ab5',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=start_values_or_method,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_am2(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    start_values_or_method: list[float] | Literal['feuler', 'meuler', 'heun3', 'rk4'] = 'rk4',
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    Adams-Moulton 2 step method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    start_values_or_method : list_like | {'feuler', 'meuler', 'heun3', 'rk4'}, optional (default='rk4')
        Start values (as list-like) or IVP method to be used to
        approximate the initial values.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_am2(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], start_values_or_method='rk4', y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21403539393939   1.21408765117936   0.00005225723997
    3        0.6   1.64882024462810   1.64894059980475   0.00012035517665
    4        0.8   2.12701347710493   2.12722953575377   0.00021605864883
    5        1.0   2.64051083997286   2.64085908577048   0.00034824579762
    6        1.2   3.17941315710999   3.17994153863173   0.00052838152173
    7        1.4   3.73162879715467   3.73240001657766   0.00077121942300
    8        1.6   4.28238809180741   4.28348378780244   0.00109569599504
    9        1.8   4.81365020810453   4.81517626779352   0.00152605968900
    10       2.0   5.30337865562364   5.30547195053468   0.00209329491103
    
    Answer = 5.30337865562364
    
    >>> start_values = [0.5, 0.82929333333333]
    >>> result = stm.ivps_am2(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, start_values_or_method=start_values, 
    ... decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758659
    2        0.4   1.21403539393939   1.21408765117936   0.00005225723997
    3        0.6   1.64882024462809   1.64894059980475   0.00012035517665
    4        0.8   2.12701347710493   2.12722953575377   0.00021605864884
    5        1.0   2.64051083997285   2.64085908577048   0.00034824579762
    6        1.2   3.17941315710999   3.17994153863173   0.00052838152174
    7        1.4   3.73162879715466   3.73240001657766   0.00077121942301
    8        1.6   4.28238809180739   4.28348378780244   0.00109569599505
    9        1.8   4.81365020810451   4.81517626779352   0.00152605968901
    10       2.0   5.30337865562363   5.30547195053468   0.00209329491105
    
    Answer = 5.30337865562363
    """
    result = ivps(
        method='am2',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=start_values_or_method,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_am3(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    start_values_or_method: list[float] | Literal['feuler', 'meuler', 'heun3', 'rk4'] = 'rk4',
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    Adams-Moulton 3 step method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    start_values_or_method : list_like | {'feuler', 'meuler', 'heun3', 'rk4'}, optional (default='rk4')
        Start values (as list-like) or IVP method to be used to
        approximate the initial values.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_am3(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], start_values_or_method='rk4', y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051270
    3        0.6   1.64892005960360   1.64894059980475   0.00002054020114
    4        0.8   2.12719640148559   2.12722953575377   0.00003313426817
    5        1.0   2.64080879026258   2.64085908577048   0.00005029550790
    6        1.2   3.17986810719526   3.17994153863173   0.00007343143647
    7        1.4   3.73229566982265   3.73240001657766   0.00010434675501
    8        1.6   4.28333843567238   4.28348378780244   0.00014535213006
    9        1.8   4.81497686771660   4.81517626779352   0.00019940007692
    10       2.0   5.30520169463125   5.30547195053468   0.00027025590342
    
    Answer = 5.30520169463125
    
    >>> start_values = [0.5, 0.82929333333333, 1.21407621066667]
    >>> result = stm.ivps_am3(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, start_values_or_method=start_values, 
    ... decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758659
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051269
    3        0.6   1.64892005960361   1.64894059980475   0.00002054020114
    4        0.8   2.12719640148560   2.12722953575377   0.00003313426817
    5        1.0   2.64080879026258   2.64085908577048   0.00005029550789
    6        1.2   3.17986810719527   3.17994153863173   0.00007343143646
    7        1.4   3.73229566982266   3.73240001657766   0.00010434675500
    8        1.6   4.28333843567240   4.28348378780244   0.00014535213005
    9        1.8   4.81497686771662   4.81517626779352   0.00019940007690
    10       2.0   5.30520169463127   5.30547195053468   0.00027025590340
    
    Answer = 5.30520169463127
    """
    result = ivps(
        method='am3',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=start_values_or_method,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_am4(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    start_values_or_method: list[float] | Literal['feuler', 'meuler', 'heun3', 'rk4'] = 'rk4',
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    Adams-Moulton 4 step method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    start_values_or_method : list_like | {'feuler', 'meuler', 'heun3', 'rk4'}, optional (default='rk4')
        Start values (as list-like) or IVP method to be used to
        approximate the initial values.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_am4(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], start_values_or_method='rk4', y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051270
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12720569067661   2.12722953575377   0.00002384507716
    5        1.0   2.64082874144856   2.64085908577048   0.00003034432191
    6        1.2   3.17990290225611   3.17994153863173   0.00003863637561
    7        1.4   3.73235091672886   3.73240001657766   0.00004909984880
    8        1.6   4.28342148841619   4.28348378780244   0.00006229938625
    9        1.8   4.81509733035245   4.81517626779352   0.00007893744107
    10       2.0   5.30537206144074   5.30547195053468   0.00009988909394

    Answer = 5.30537206144074
    
    >>> start_values = [0.5, 0.82929333333333, 1.21407621066667,
    ... 1.6489220170416]
    >>> result = stm.ivps_am4(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, start_values_or_method=start_values, 
    ... decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758659
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051269
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12720569067661   2.12722953575377   0.00002384507716
    5        1.0   2.64082874144856   2.64085908577048   0.00003034432192
    6        1.2   3.17990290225611   3.17994153863173   0.00003863637562
    7        1.4   3.73235091672886   3.73240001657766   0.00004909984880
    8        1.6   4.28342148841619   4.28348378780244   0.00006229938625
    9        1.8   4.81509733035245   4.81517626779352   0.00007893744108
    10       2.0   5.30537206144073   5.30547195053468   0.00009988909394
    
    Answer = 5.30537206144073
    """
    result = ivps(
        method='am4',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=start_values_or_method,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result


def ivps_eheun(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the Euler-Heun method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_eheun(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82600000000000   0.82929862091992   0.00329862091991
    2        0.4   1.20692000000000   1.21408765117936   0.00716765117936
    3        0.6   1.63724240000000   1.64894059980475   0.01169819980475
    4        0.8   2.11023572800000   2.12722953575377   0.01699380775377
    5        1.0   2.61768758816000   2.64085908577048   0.02317149761048
    6        1.2   3.14957885755520   3.17994153863173   0.03036268107653
    7        1.4   3.69368620621734   3.73240001657766   0.03871381036032
    8        1.6   4.23509717158516   4.28348378780244   0.04838661621728
    9        1.8   4.75561854933390   4.81517626779352   0.05955771845963
    10       2.0   5.23305463018735   5.30547195053468   0.07241732034732
    
    Answer = 5.23305463018735
    """
    result = ivps(
        method='eheun',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_abm2(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    start_values_or_method: list[float] | Literal['feuler', 'meuler', 'heun3', 'rk4'] = 'rk4',
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    Adams-Bashforth 2 step method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    start_values_or_method : list_like | {'feuler', 'meuler', 'heun3', 'rk4'}, optional (default='rk4')
        Start values (as list-like) or IVP method to be used to
        approximate the initial values.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_abm2(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], start_values_or_method='rk4', y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21383080000000   1.21408765117936   0.00025685117936
    3        0.6   1.64831895066667   1.64894059980475   0.00062164913808
    4        0.8   2.12609400132000   2.12722953575377   0.00113553443377
    5        1.0   2.63901243211693   2.64085908577048   0.00184665365354
    6        1.2   3.17712435149063   3.17994153863173   0.00281718714110
    7        1.4   3.72827282801230   3.73240001657766   0.00412718856536
    8        1.6   4.27760433494023   4.28348378780244   0.00587945286222
    9        1.8   4.80697060369636   4.81517626779352   0.00820566409717
    10       2.0   5.29419779919712   5.30547195053468   0.01127415133756
    
    Answer = 5.29419779919712
    
    >>> start_values = [0.5, 0.82929333333333]
    >>> result = stm.ivps_abm2(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, start_values_or_method=start_values,
    ... decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758659
    2        0.4   1.21383080000000   1.21408765117936   0.00025685117937
    3        0.6   1.64831895066666   1.64894059980475   0.00062164913808
    4        0.8   2.12609400131999   2.12722953575377   0.00113553443377
    5        1.0   2.63901243211693   2.64085908577048   0.00184665365355
    6        1.2   3.17712435149062   3.17994153863173   0.00281718714111
    7        1.4   3.72827282801229   3.73240001657766   0.00412718856537
    8        1.6   4.27760433494021   4.28348378780244   0.00587945286223
    9        1.8   4.80697060369634   4.81517626779352   0.00820566409719
    10       2.0   5.29419779919710   5.30547195053468   0.01127415133758
    
    Answer = 5.2941977991971
    """
    result = ivps(
        method='abm2',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=start_values_or_method,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_abm3(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    start_values_or_method: list[float] | Literal['feuler', 'meuler', 'heun3', 'rk4'] = 'rk4',
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    Adams-Bashforth 3 step method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    start_values_or_method : list_like | {'feuler', 'meuler', 'heun3', 'rk4'}, optional (default='rk4')
        Start values (as list-like) or IVP method to be used to
        approximate the initial values.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_abm3(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], start_values_or_method='rk4', y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051270
    3        0.6   1.64890875007778   1.64894059980475   0.00003184972697
    4        0.8   2.12716847095545   2.12722953575377   0.00006106479832
    5        1.0   2.64075748811125   2.64085908577048   0.00010159765922
    6        1.2   3.17978445607730   3.17994153863173   0.00015708255443
    7        1.4   3.73216785930716   3.73240001657766   0.00023215727051
    8        1.6   4.28315101137156   4.28348378780244   0.00033277643088
    9        1.8   4.81470969644836   4.81517626779352   0.00046657134516
    10       2.0   5.30482865011835   5.30547195053468   0.00064330041633

    Answer = 5.30482865011835
    
    >>> start_values = [0.5, 0.82929333333333, 1.21407621066667]
    >>> result = stm.ivps_abm3(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, start_values_or_method=start_values,
    ... decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758659
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051269
    3        0.6   1.64890875007778   1.64894059980475   0.00003184972696
    4        0.8   2.12716847095545   2.12722953575377   0.00006106479831
    5        1.0   2.64075748811126   2.64085908577048   0.00010159765922
    6        1.2   3.17978445607731   3.17994153863173   0.00015708255442
    7        1.4   3.73216785930717   3.73240001657766   0.00023215727050
    8        1.6   4.28315101137158   4.28348378780244   0.00033277643087
    9        1.8   4.81470969644838   4.81517626779352   0.00046657134515
    10       2.0   5.30482865011836   5.30547195053468   0.00064330041631
    
    Answer = 5.30482865011836
    """
    result = ivps(
        method='abm3',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=start_values_or_method,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_abm4(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    start_values_or_method: list[float] | Literal['feuler', 'meuler', 'heun3', 'rk4'] = 'rk4',
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    Adams-Bashforth 4 step method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    start_values_or_method : list_like | {'feuler', 'meuler', 'heun3', 'rk4'}, optional (default='rk4')
        Start values (as list-like) or IVP method to be used to
        approximate the initial values.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_abm4(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], start_values_or_method='rk4', y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051270
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12728924905233   2.12722953575377   0.00005971329857
    5        1.0   2.64105332811142   2.64085908577048   0.00019424234094
    6        1.2   3.18031412883292   3.17994153863173   0.00037259020119
    7        1.4   3.73301858540624   3.73240001657766   0.00061856882858
    8        1.6   4.28444240619668   4.28348378780244   0.00095861839424
    9        1.8   4.81659556132722   4.81517626779352   0.00141929353369
    10       2.0   5.30750818139328   5.30547195053468   0.00203623085860
    
    Answer = 5.30750818139328
    
    >>> start_values = [0.5, 0.82929333333333, 1.21407621066667,
    ... 1.6489220170416]
    >>> result = stm.ivps_abm4(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, start_values_or_method=start_values,
    ... decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758659
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051269
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12728924905233   2.12722953575377   0.00005971329856
    5        1.0   2.64105332811142   2.64085908577048   0.00019424234094
    6        1.2   3.18031412883292   3.17994153863173   0.00037259020119
    7        1.4   3.73301858540624   3.73240001657766   0.00061856882857
    8        1.6   4.28444240619668   4.28348378780244   0.00095861839423
    9        1.8   4.81659556132721   4.81517626779352   0.00141929353368
    10       2.0   5.30750818139327   5.30547195053468   0.00203623085860
    
    Answer = 5.30750818139327
    
    """
    result = ivps(
        method='abm4',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=start_values_or_method,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_abm5(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    start_values_or_method: list[float] | Literal['feuler', 'meuler', 'heun3', 'rk4'] = 'rk4',
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    Adams-Bashforth 5 step method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, optional(default=None)
        The exact solution of the ODE.
    start_values_or_method : list_like | {'feuler', 'meuler', 'heun3', 'rk4'}, optional (default='rk4')
        Start values (as list-like) or IVP method to be used to
        approximate the initial values.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_abm5(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], start_values_or_method='rk4', y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051270
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12720268494794   2.12722953575377   0.00002685080582
    5        1.0   2.64084332085071   2.64085908577048   0.00001576491976
    6        1.2   3.17994955300343   3.17994153863173   0.00000801437170
    7        1.4   3.73243661722837   3.73240001657766   0.00003660065071
    8        1.6   4.28356197342805   4.28348378780244   0.00007818562561
    9        1.8   4.81531576484758   4.81517626779352   0.00013949705406
    10       2.0   5.30569478939270   5.30547195053468   0.00022283885803
    
    Answer = 5.3056947893927
    
    >>> start_values = [0.5, 0.82929333333333, 1.21407621066667,
    ... 1.6489220170416, 2.127202684948]
    >>> result = stm.ivps_abm5(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, start_values_or_method=start_values, 
    ... decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758659
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051269
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12720268494800   2.12722953575377   0.00002685080577
    5        1.0   2.64084332085080   2.64085908577048   0.00001576491967
    6        1.2   3.17994955300352   3.17994153863173   0.00000801437180
    7        1.4   3.73243661722848   3.73240001657766   0.00003660065082
    8        1.6   4.28356197342820   4.28348378780244   0.00007818562576
    9        1.8   4.81531576484776   4.81517626779352   0.00013949705424
    10       2.0   5.30569478939292   5.30547195053468   0.00022283885824
    
    Answer = 5.30569478939292
    """
    result = ivps(
        method='abm5',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=start_values_or_method,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_milne_simpson(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    start_values_or_method: list[float] | Literal['feuler', 'meuler', 'heun3', 'rk4'] = 'rk4',
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    Milne-Simpson step method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    start_values_or_method : list_like | {'feuler', 'meuler', 'heun3', 'rk4'}, optional (default='rk4')
        Start values (as list-like) or IVP method to be used to
        approximate the initial values.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_milne_simpson(odeqtn=f,
    ... exactsol=ft, time_span=[a, b],
    ... start_values_or_method='rk4', y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051270
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12720527906304   2.12722953575377   0.00002425669073
    5        1.0   2.64082787244227   2.64085908577048   0.00003121332820
    6        1.2   3.17990149510207   3.17994153863173   0.00004004352966
    7        1.4   3.73234852073496   3.73240001657766   0.00005149584270
    8        1.6   4.28341768950392   4.28348378780244   0.00006609829853
    9        1.8   4.81509161557732   4.81517626779353   0.00008465221620
    10       2.0   5.30536378185149   5.30547195053467   0.00010816868319
    
    Answer = 5.30536378185149
    
    >>> start_values = [0.5, 0.82929333333333, 1.21407621066667,
    ... 1.6489220170416]
    >>> result = stm.ivps_milne_simpson(odeqtn=f,
    ... exactsol=ft, time_span=[a, b], start_values_or_method=start_values,
    ... y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758659
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051269
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12720527906304   2.12722953575377   0.00002425669073
    5        1.0   2.64082787244227   2.64085908577048   0.00003121332820
    6        1.2   3.17990149510207   3.17994153863173   0.00004004352966
    7        1.4   3.73234852073496   3.73240001657766   0.00005149584270
    8        1.6   4.28341768950392   4.28348378780244   0.00006609829853
    9        1.8   4.81509161557732   4.81517626779353   0.00008465221620
    10       2.0   5.30536378185149   5.30547195053467   0.00010816868319
    
    Answer = 5.30536378185149
    """
    result = ivps(
        method='msimpson',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=start_values_or_method,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_milne_simpson_modified(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    start_values_or_method: list[float] | Literal['feuler', 'meuler', 'heun3', 'rk4'] = 'rk4',
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    modified Milne-Simpson method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    start_values_or_method : list_like | {'feuler', 'meuler', 'heun3', 'rk4'}, optional (default='rk4')
        Start values (as list-like) or IVP method to be used to
        approximate the initial values.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_milne_simpson_modified(odeqtn=f,
    ... exactsol=ft, time_span=[a, b],
    ... start_values_or_method='rk4', y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051270
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12721346463479   2.12722953575377   0.00001607111897
    5        1.0   2.64083609467696   2.64085908577048   0.00002299109351
    6        1.2   3.17991889505857   3.17994153863173   0.00002264357316
    7        1.4   3.73237040524644   3.73240001657766   0.00002961133123
    8        1.6   4.28345305563341   4.28348378780244   0.00003073216903
    9        1.8   4.81513829334157   4.81517626779353   0.00003797445195
    10       2.0   5.30543142997583   5.30547195053467   0.00004052055884
    
    Answer = 5.30543142997583
    
    >>> start_values = [0.5, 0.82929333333333, 1.21407621066667,
    ... 1.6489220170416]
    >>> result = stm.ivps_milne_simpson_modified(odeqtn=f,
    ... exactsol=ft, time_span=[a, b], start_values_or_method=start_values,
    ... y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758659
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051269
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12721346463480   2.12722953575377   0.00001607111897
    5        1.0   2.64083609467696   2.64085908577048   0.00002299109351
    6        1.2   3.17991889505857   3.17994153863173   0.00002264357315
    7        1.4   3.73237040524644   3.73240001657766   0.00002961133122
    8        1.6   4.28345305563342   4.28348378780244   0.00003073216903
    9        1.8   4.81513829334158   4.81517626779353   0.00003797445195
    10       2.0   5.30543142997584   5.30547195053467   0.00004052055884
    
    Answer = 5.30543142997584
    """
    result = ivps(
        method='mmsimpson',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=start_values_or_method,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_hamming(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    start_values_or_method: list[float] | Literal['feuler', 'meuler', 'heun3', 'rk4'] = 'rk4',
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the Hammings method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    start_values_or_method : list_like | {'feuler', 'meuler', 'heun3', 'rk4'}, optional (default='rk4')
        Start values (as list-like) or IVP method to be used to
        approximate the initial values.
    stepsize : float, optional (default=None)
        The interval (difference between two consecutive time points).
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_hamming(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], start_values_or_method='rk4', y0=y0, decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758658
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051270
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12720527906304   2.12722953575377   0.00002425669073
    5        1.0   2.64082787244227   2.64085908577048   0.00003121332820
    6        1.2   3.17990149510207   3.17994153863173   0.00004004352966
    7        1.4   3.73234852073496   3.73240001657766   0.00005149584270
    8        1.6   4.28341768950392   4.28348378780244   0.00006609829853
    9        1.8   4.81509161557732   4.81517626779353   0.00008465221620
    10       2.0   5.30536378185149   5.30547195053467   0.00010816868319
    
    Answer = 5.30536378185149
    
    >>> start_values = [0.5, 0.82929333333333, 1.21407621066667,
    ... 1.6489220170416]
    >>> result = stm.ivps_hamming(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, start_values_or_method=start_values, 
    ... decimal_points=14)
    
        Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0        0.0   0.50000000000000   0.50000000000000   0.00000000000000
    1        0.2   0.82929333333333   0.82929862091992   0.00000528758659
    2        0.4   1.21407621066667   1.21408765117936   0.00001144051269
    3        0.6   1.64892201704160   1.64894059980475   0.00001858276315
    4        0.8   2.12720527906304   2.12722953575377   0.00002425669073
    5        1.0   2.64082787244227   2.64085908577048   0.00003121332820
    6        1.2   3.17990149510207   3.17994153863173   0.00004004352966
    7        1.4   3.73234852073496   3.73240001657766   0.00005149584270
    8        1.6   4.28341768950392   4.28348378780244   0.00006609829853
    9        1.8   4.81509161557732   4.81517626779353   0.00008465221620
    10       2.0   5.30536378185149   5.30547195053467   0.00010816868319
    
    Answer = 5.30536378185149
    """
    result = ivps(
        method='hamming',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=start_values_or_method,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_rkf45(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    hmin_hmax: list[float] = [0.01, 0.25],
    nsteps: int = 10,
    tolerance: float = 1e-6,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    Runge-Kutta-Fehlberg 45 method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    hmin_hmax : list-like, optional (default=[0.01, 0.25])
        A list-like of length 2 representing `hmin` and `hmax`.
    nsteps : int, optional (default=10)
        Number of steps (time points).
    tolerance : float, optional (default=1e-6)
        The allowable tolerance.
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_rkf45(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, tolerance=1e-5, hmin_hmax=[0.01, 0.25],
    ... decimal_points=14)
    
               Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0  0.00000000000000   0.50000000000000   0.50000000000000   0.00000000000000
    1  0.25000000000000   0.92048860207582   0.92048729165613   0.00000131041969
    2  0.48655220228477   1.39649101428839   1.39648844281524   0.00000257147315
    3  0.72933319984230   1.95374878715415   1.95374461367237   0.00000417348178
    4  0.97933319984230   2.58642601474198   2.58641982885096   0.00000618589102
    5  1.22933319984230   3.26046051047871   3.26045200576709   0.00000850471163
    6  1.47933319984230   3.95209553728388   3.95208439562523   0.00001114165865
    7  1.72933319984230   4.63082681953757   4.63081272915408   0.00001409038349
    8  1.97933319984230   5.25748606455951   5.25746874929178   0.00001731526773
    9  2.00000000000000   5.30548962736878   5.30547195053468   0.00001767683411
    
    Answer = 5.30548962736878
    """
    result = ivps(
        method='rkf45',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=hmin_hmax,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=tolerance,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_rkf54(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    hmin_hmax: list[float] = [0.01, 0.25],
    nsteps: int = 10,
    tolerance: float = 1e-6,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    Runge-Kutta-Fehlberg 54 method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    hmin_hmax : list-like, optional (default=[0.01, 0.25])
        A list-like of length 2 representing `hmin` and `hmax`.
    nsteps : int, optional (default=10)
        Number of steps (time points).
    tolerance : float, optional (default=1e-6)
        The allowable tolerance.
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_rkf54(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, tolerance=1e-5, hmin_hmax=[0.01, 0.25],
    ... decimal_points=14)
    
               Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0  0.00000000000000   0.50000000000000   0.50000000000000   0.00000000000000
    1  0.05800000000000   0.58950650323704   0.58950650214486   0.00000000109219
    2  0.29000000000000   0.99588716028182   0.99588625598726   0.00000090429456
    3  0.52766921570860   1.48628675514116   1.48628469508866   0.00000206005250
    4  0.77205321996763   2.05807349844964   2.05806997210943   0.00000352634022
    5  1.02205321996763   2.69925724241511   2.69925192821842   0.00000531419669
    6  1.27205321996763   3.37814752405616   3.37814019094387   0.00000733311229
    7  1.52205321996763   4.06995070276124   4.06994113264830   0.00000957011294
    8  1.77205321996763   4.74283109148821   4.74281910527331   0.00001198621490
    9  2.00000000000000   5.30548641911408   5.30547195053468   0.00001446857940

    Answer = 5.30548641911408
    """
    result = ivps(
        method='rkf54',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=hmin_hmax,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=tolerance,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_rkv(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    hmin_hmax: list[float] = [0.01, 0.25],
    nsteps: int = 10,
    tolerance: float = 1e-6,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the 
    Runge-Kutta-Fehlberg-Verner method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    hmin_hmax : list-like, optional (default=[0.01, 0.25])
        A list-like of length 2 representing `hmin` and `hmax`.
    nsteps : int, optional (default=10)
        Number of steps (time points).
    tolerance : float, optional (default=1e-6)
        The allowable tolerance.
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_rkv(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, tolerance=1e-5, hmin_hmax=[0.01, 0.25],
    ... decimal_points=14)
    
               Time (t)  Approximated (yi)  Exact solution(y)  Error: | y - yi |
    0  0.00000000000000   0.50000000000000   0.50000000000000   0.00000000000000
    1  0.25000000000000   0.92048860207582   0.92048729165613   0.00000131041969
    2  0.48655220228477   1.39649101428839   1.39648844281524   0.00000257147315
    3  0.72933319984230   1.95374878715415   1.95374461367237   0.00000417348178
    4  0.97933319984230   2.58642601474198   2.58641982885096   0.00000618589102
    5  1.22933319984230   3.26046051047871   3.26045200576709   0.00000850471163
    6  1.47933319984230   3.95209553728388   3.95208439562523   0.00001114165865
    7  1.72933319984230   4.63082681953757   4.63081272915408   0.00001409038349
    8  1.97933319984230   5.25748606455951   5.25746874929178   0.00001731526773
    9  2.00000000000000   5.30548962736878   5.30547195053468   0.00001767683411
    
    Answer = 5.30548962736878
    """
    result = ivps(
        method='rkv',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        hmin_hmax=hmin_hmax,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=tolerance,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_adams_variable_step(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: list[float] | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the Adams variable 
    step method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    hmin_hmax : list-like, optional (default=[0.01, 0.25])
        A list-like of length 2 representing `hmin` and `hmax`.
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps_adams_variable_step(odeqtn=f,
    ... exactsol=ft, time_span=[a, b], y0=y0, decimal_points=14)
    
    (To update)
    
    """
    result = ivps(
        method='adamsvariablestep',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_extrapolation(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: list[float] | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the Extrapolation 
    method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    hmin_hmax : list-like, optional (default=[0.01, 0.25])
        A list-like of length 2 representing `hmin` and `hmax`.
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
    """
    result = ivps(
        method='extrapolation',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result
    

def ivps_tnewton(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    stepsize: float | None = None,
    nsteps: int = 10,
    maxit: int = 10,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Solve an ordinary differential equation using the Trapezoidal 
    with Newton approximation method.

    Parameters
    ----------
    odeqtn : {str, symbolic, list_like, callable}
        The ODE equation to be integrated numerically.
    time_span : list_like
        Start and end time points (lower and upper limits of 
        integration).
    exactsol : {None, str, symbolic, Callable}, (default=None)
        The exact solution of the ODE.
    start_values_or_method : list_like | {'feuler', 'meuler', 'heun3', 'rk4'}, optional (default='rk4')
        Start values (as list-like) or IVP method to be used to
        approximate the initial values.
    stepsize : float | list-like, optional (default=None)
        The interval. (difference between two consecutive time points)
    nsteps : int, optional (default=10)
        Number of steps (time points).
    maxit : int, optional (default=10)
        Maximum number of iterations.
    show_iters : int, optional (default=None)
        Integer representing the number of iterations to be displayed.
        Valid values: None or 1 <= x <= n. If None, then all 
        iterations are displayed.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=8)
        Number of decimal points or significant figures for symbolic expressions.

    Returns
    -------
    table : pandas.DataFrame
        A DataFrame with the tabulated results.
    dframe : pandas.Styler
        Above table with answer highlighted.
    float
        The numerical solution of the ODE.

    Examples
    --------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivps(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, decimal_points=14)
    
    """
    result = ivps(
        method='tnewton',
        odeqtn=odeqtn,
        time_span=time_span,
        y0=y0,
        derivs=None,
        exactsol=exactsol,
        vars=vars,
        start_values_or_method=None,
        stepsize=stepsize,
        nsteps=nsteps,
        maxit=maxit,
        tolerance=None,
        show_iters=show_iters,
        auto_display=auto_display,
        decimal_points=decimal_points
    )
    
    return result

def ivp_adams(
    odeqtn: str | Expr | Callable,
    time_span: list[float],
    y0: float,
    exactsol: str | Expr | Callable | None = None,
    vars: list[str] = ['t', 'y'],
    hmin_hmax: list[float] = [0.01, 0.25],
    tolerance=1e-6,
    show_iters: int | None = None,
    auto_display: bool = True,
    decimal_points: int = 8
) -> Result:
    """
    Example
    -------
    >>> import stemlab as stm
    >>> f = 'y - t^2 + 1'
    >>> ft = '(t + 1)^2 - 0.5 * exp(t)'
    >>> a, b = (0, 2)
    >>> y0 = 0.5
    >>> result = stm.ivp_adams(odeqtn=f, exactsol=ft,
    ... time_span=[a, b], y0=y0, hmin=0.01, hmax=0.20, decimal_points=14)
    
    """
    f = odeqtn
    def rk4(t, y, h, start):
        tj = [t]
        yj = [y]
        for i in range(start, 3):
            k1 = h * f(tj[i], yj[i])
            k2 = h * f(tj[i] + h / 2, yj[i] + k1 / 2)
            k3 = h * f(tj[i] + h / 2, yj[i] + k2 / 2)
            k4 = h * f(tj[i] + h, yj[i] + k3)
            y_new = yj[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            yj.append(y_new)
            tj.append(tj[i] + h)

        return tj, yj
        
    t0, b = time_span
    hmin = hmin_hmax[0]
    hmax = hmin_hmax[1]
    h = hmax
    flag = 1
    last = 0
    
    t, y = rk4(t=t0, y=y0, h=h, start=0)
    
    nflag = 1
    i = 3
    f = odeqtn
    while flag == 1:
        # Adams-Bashforth 4-step as predictor
        yp = y[i] + (h / 24) * (
            55 * f(t[i], y[i]) - 59 * f(t[i - 1], y[i - 1]) 
            + 37 * f(t[i - 2], y[i - 2]) - 9 * f(t[i - 3], y[i - 3])
        )
        # Adams-Moulton 3-step as corrector, used t[i] + h, not t[i+1]
        yc = y[i] + (h / 24) * (
            9 * f(t[i] + h, yp) + 19 * f(t[i], y[i]) 
            - 5 * f(t[i - 1], y[i - 1]) + f(t[i - 2], y[i - 2])
        )
        sigma = 19 * abs(yc - yp) / (270 * h)
        if sigma <= tolerance:
            y.append(yc) # result
            if flag == 1:
                for j in range(i - 2, i):
                    out = j, t[j], y[j], h
            else:
                out = j, t[i], y[i], h
                
            if last == 0:
                flag = 0
            else:
                i += 1
                nflag = 0
                if sigma <= 0.1 * tolerance or t[i] + h > b:
                    q = (tolerance / (2 * sigma)) ** 0.25
                    if q > 4:
                        h *= 4
                    else:
                        h *= q
                    if h > hmax:
                        h = hmax
                    if t[i] + 4 * h > b:
                        h = (b - t[i]) / 4
                        last = 1
                    t, y = rk4(t=t[i], y=y[i], h=h, start=i)
                    nflag = 1
                    i += 3
        else:
            q = (tolerance / (2 * sigma)) ** 0.25
            if q < 0.1:
                h *= 0.1
            else:
                h *= q
            if h < hmin:
                flag = 0
                print('hmin exceeded')
            else:
                if nflag == 1:
                    i -= 3
                    t, y = rk4(t=t[i], y=y[i], h=h, start=i)
                    i += 3
                    nflag = 1
        t.append(t[i] + h) 
        
    return t, y