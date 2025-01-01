'''
=======
stemlab
=======

stemlab is a Python library for performing mathematical computations.
It aims to become a first choice library for trainers and students
in Science, Technology, Engineering and Mathematics.

How to use the documentation
----------------------------
Documentation is available in two forms: 
    - Docstrings provided with the code
    - stemlab homepage <https://stemlab.stemxresearch.com>`

The docstring examples assume that `stemlab` has been imported as ``stm``::

>>> import stemlab as stm

Code snippets are indicated by three greater-than signs::

>>> x = 42
>>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

>>> help(stm.arr_abrange)
... # doctest: +SKIP

To search for documents containing a keyword, do::

>>> np.lookfor('keyword')
... # doctest: +SKIP

Viewing documentation using IPython
-----------------------------------

Start IPython and import `stemlab` 

>>> import stemlab as stm

- To see which functions are available in `stemlab`, enter ``stm.<TAB>`` 
(where ``<TAB>`` refers to the TAB key), or use
``stm.*rk*?<ENTER>`` (where ``<ENTER>`` refers to the ENTER key) to 
filter functions that contain the characters *rk*.

- To view the docstring for a specified function, use
``np.arr_abrange?<ENTER>`` (to view the docstring) and 
``np.cos??<ENTER>`` (to view the source code).

'''

import sys

if sys.version_info < (3, 6):
    raise ImportError('stemlab requires installation of Python version 3.6 or above.')
del sys

__version__ = '0.0.21a'
__author__ = 'John Indika'
__credits__ = 'STEM Research'


from .mathematics.calculus import (
    # differentiation_integration
    sym_diff_parametric, 
    
    # nonlinear nle_roots_regula_falsi_modified
    nle_roots, nle_roots_aitken, nle_roots_bisection, nle_roots_fixed_point, 
    nle_roots_newton_raphson_modified, nle_roots_regula_falsi_modified, nle_roots_newton_raphson, 
    nle_roots_regula_falsi, nle_roots_secant, nle_roots_steffensen, nle_roots_system,
    
    # numerical_differentiation
    diff_fd_first_derivative, diff_fd_second_derivative,
    diff_fd_third_derivative, diff_fd_fourth_derivative,
    diff_richardson,
    
    # numerical_integration
    int_cotes, int_cotes_data, int_composite, int_composite_data, int_romberg, 
    int_gauss_legendre,
    
    # odes
    # ivps_adams_variable_step, ivps_extrapolation, ivps_rktrapezoidal, ivps_tnewton, ivp_adams
    ivps, ivps_ab2, ivps_ab3, ivps_ab4, ivps_ab5, ivps_abm2, ivps_abm3, 
    ivps_abm4, ivps_abm5, ivps_am2, ivps_am3, ivps_am4, ivps_beuler,
    ivps_eheun, ivps_feuler, ivps_hamming, ivps_heun3, ivps_meuler,
    ivps_milne_simpson, ivps_milne_simpson_modified, ivps_nystrom3, ivps_ralston2,
    ivps_rk1stage, ivps_rk2stage, ivps_rk3, ivps_rk38, ivps_rk3stage, ivps_rk4,
    ivps_rk5, ivps_rkbeuler, ivps_rkf45, ivps_rkf54, ivps_rkmersen,
    ivps_rkmeuler, ivps_rkmidpoint, ivps_rkv, ivps_taylor
)

from .core import (
    
    # array_like
    conv_to_arraylike, arr_abrange, is_strictly_increasing, is_diff_constant, conv_list_to_dict, conv_list_to_string, 
    dict_subset, dict_sort, dict_merge, list_merge, tuple_merge, arr_table_blank_row, list_join,
    
    # arrays
    
    # data_types
    conv_structure_to_list, is_any_element_negative,
    
    # decimals
    fround,
 
    # htmlatex
    tex_aligned_latex, tex_array_to_latex, sta_dframe_color, sta_dframe_to_html, tex_display_latex, 
    tex_integral_to_latex, tex_list_to_latex, tex_matrix_to_latex_eqtns, tex_table_to_latex, tex_to_latex,
    
    # symbolic
    is_symexpr, sym_poly_constant, sym_simplify_expr, sym_poly_terms, sym_remove_zeros, 
    sym_sympify, sym_lambdify_expr, sym_get_expr_vars,
)

from .core.base import (
    # arrays
    arr_max_zeros, arr_contains_string, arr_get_cols, arr_get_rows, arr_swap,
    arr_swap_cols, arr_swap_rows, list_elements_not_in,
    
    # dictionaries
    Dictionary, dict_none_remove, dict_none_keys,
    
    # constants
    MathConstants,
    
    # strings
    str_args_to_dict, str_capitalize_nohiphen, str_change_case, 
    str_normal_case,
    str_partial_characters, str_plus_minus, str_replace_space,
    str_remove_special_symbols, str_replace_characters, str_singular_plural,
    str_random_string, str_strip_all, str_print, str_remove_trailing_zeros,
    str_replace_dot_zero
)

from .core.validators import ValidateArgs

from .datasets import dataset_read, datasets_show
from .mathematics.interpolation import (
    interp, interp_lagrange, interp_bessel, interp_clamped_cubic_splines,
    interp_exponential, interp_gauss_backward, interp_gauss_forward,
    interp_hermite, interp_laplace_everett, interp_linear_regression,
    interp_linear_splines, interp_natural_cubic_splines, interp_neville,
    interp_newton_backward, interp_newton_divided, interp_newton_forward,
    interp_not_a_knot_splines, interp_polynomial, interp_power,
    interp_quadratic_splines, interp_reciprocal, interp_saturation,
    interp_stirling, interp_straight_line
)
from .mathematics.linearalgebra import (
    # core
    la_norm_order, la_relax_parameter, la_inverse,
    
    # iterative techniques
    la_solve, la_solve_jacobi, la_solve_gauss_seidel, la_solve_sor, 
    la_solve_conjugate,
    
    # linear systems
    la_gauss_pivot, la_jacobian,
    
    # matrix forms
    la_echelon_form, la_ref, la_rref
)

from .mathematics.trigonometry import TrigFunctions

def __dir__():
    from stemlab.utils import get_public_names
    return get_public_names(globals_dict=globals())

#===========================================================================#
#                                                                           #
# STEM RESEARCH :: Technology for Innovation :: https://stemlab.org        #
#																			#
#===========================================================================#