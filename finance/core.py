from typing import Literal
import warnings

from numpy import nan
from pandas import DataFrame
from IPython.display import display

from stemlab.core.decimals import fround
from stemlab.core.base.computations import make_subject_and_solve
from stemlab.core.display import Result, display_results
from stemlab.graphics.common import gph_barchart_simple
from stemlab.core.base.functions import get_function_name
from stemlab.core.validators.validate import ValidateArgs


def simple_interest(
    p: float | int = None,
    r: float | int = None,
    t: int | float = None,
    i: int | float = None,
    decimal_points: int = 2
) -> float:
    """
    Calculate the principal (P), rate (R) or time (T) using the simple 
    interest formula.

    Parameters
    ----------
    p : {float, int}
        Principal amount (initial investment or loan).
    r : {float, int}
        Annual interest rate (in decimal form, e.g., 0.05 for 5%).
    t : {float, int}
        Time in years.
    i : {float, int}
        Accrued interest.
    decimal_points : int, optional (default=-1)
        Number of decimal points.

    Returns
    -------
    result : float
        Calculated value based on provided inputs.

    Examples
    --------
    >>> p = 10000
    >>> r = 0.045
    >>> t = 5
    >>> fin.simple_interest(p=p, r=r, t=t)
    2250.0
    """
    if p is not None:
        p = ValidateArgs.check_numeric(par_name='p', is_positive=True, user_input=p)
        
    if r is not None:
        r = ValidateArgs.check_numeric(
            par_name='r', limits=[0, 1], is_positive=True, user_input=r
        )
        
    if t is not None:
        t = ValidateArgs.check_numeric(par_name='t', is_positive=True, user_input=t)
        
    if i is not None:
        i = ValidateArgs.check_numeric(par_name='i', is_positive=True, user_input=i)
        
    decimal_points = ValidateArgs.check_decimals(x=decimal_points)
    
    args_dict = {'i': i, 'p': p, 'r': r, 't': t}
    ValidateArgs.check_args_count(ftn_name=get_function_name(), args_dict=args_dict)
    fexpr = 'i - p * r * t'
    result = make_subject_and_solve(
        dct=args_dict, fexpr=fexpr,  initial_guess=1
    )
    result = float(fround(x=result, decimal_points=decimal_points))
    
    return result


def compound_interest(
    p: float | int | None = None,
    r: float | int | None = None,
    n: int | None = None,
    t: float | int | None = None,
    a: float | int | None = None,
    x0: float | int = 1,
    decimal_points: int = 2
) -> float:
    """
    Calculate compound interest.

    Parameters
    ----------
    p : {float, int}
        Principal amount (initial investment or loan).
    r : {float, int}
        Annual interest rate (in decimal form, e.g., 0.05 for 5%).
    n : int
        Number of times that interest is compounded per year.
    t : {float, int}
        Time in years.
    a : {float, int}
        Total amount.
    x0 : {float, int}
        Initial guess for complex expression that require use of 
        numerical techniques.
    decimal_points : int, optional (default=-1)
        Number of decimal points.

    Returns
    -------
    result : float
        Calculated value based on provided inputs.
        
    Examples
    --------
    >>> p = 10000
    >>> r = 0.045
    >>> n = 4
    >>> t = 5
    >>> fin.compound_interest(p=p, r=r, n=n, t=t)
    12507.51
    """    
    if p is not None:
        p = ValidateArgs.check_numeric(par_name='p', is_positive=True, user_input=p)
    if r is not None:
        r = ValidateArgs.check_numeric(
            par_name='r', limits=[0, 1], is_positive=True, user_input=r
        )
    if n is not None:
        n = ValidateArgs.check_numeric(
            par_name='n', is_positive=True, is_integer=True, user_input=n
        )
    if t is not None:
        t = ValidateArgs.check_numeric(par_name='t', is_positive=True, user_input=t)
    if a is not None:
        a = ValidateArgs.check_numeric(par_name='a', is_positive=True, user_input=a)
    if x0 is not None:
        x0 = ValidateArgs.check_numeric(par_name='x0', user_input=x0)
    
    decimal_points = ValidateArgs.check_decimals(x=decimal_points)
    
    args_dict = {'a': a, 'p': p, 'r': r, 'n': n, 't': t}
    ValidateArgs.check_args_count(ftn_name=get_function_name(), args_dict=args_dict)
    fexpr = 'a - p * (1 + r / n) ** (n * t)'
    result = make_subject_and_solve(
        dct=args_dict, fexpr=fexpr,  initial_guess=x0
    )
    result = float(fround(x=result, decimal_points=decimal_points))
    
    return result


def amortization_schedule(
    principal: float | int,
    annual_rate: float,
    years: float | int,
    schedule_type: Literal['monthly', 'annually'] = 'monthly',
    auto_display: bool = True,
    decimal_points: int = 2
) -> Result:
    """
    Calculate the amortization schedule for a loan.

    Parameters
    ----------
    principal : {float, int}
        The loan amount.
    annual_rate : float
        The annual interest rate as a percentage.
    years : int
        The loan term in years or months.
    schedule_type : {'monthly', 'annually'}, optional (default='monthly')
        The type of amortization schedule to calculate.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=2)
        Number of decimal points for rounding the output values.

    Returns
    -------
    result : Result
        - dframe: pandas.DataFrame
            A DataFrame containing the amortization schedule with columns:
                - Number
                - Payment
                - Principal
                - Interest
                - Balance
        - number_of_installments : int
            Number of installments.
        - total_interest : float
            Total interest accrued on the principal.

    Raises
    ------
    ValueError
        If any of the input parameters are invalid, such as negative 
        values, non-numeric types, or unsupported schedule types.
        
    Examples
    --------
    >>> import stemlab.finance as fin
    >>> df = fin.amortization_schedule(principal=200000, annual_rate=6,
    ... years=15, schedule_type='monthly', decimal_points=2)
        Month  Payment Principal Interest    Balance
    0                                      200000.00
    1       1  1687.71    687.71   1000.0  199312.29
    2       2  1687.71    691.15   996.56  198621.13
    3       3  1687.71    694.61   993.11  197926.53
    4       4  1687.71    698.08   989.63  197228.45
    ..    ...      ...       ...      ...        ...
    176   176  1687.71   1646.15    41.57    6667.31
    177   177  1687.71   1654.38    33.34    5012.93
    178   178  1687.71   1662.65    25.06    3350.28
    179   179  1687.71   1670.96    16.75    1679.32
    180   180  1687.71   1679.32      8.4      -0.00

    [181 rows x 5 columns]
    
    Installments = 80
    
    Interest = 103788.46
    
    Amount = 303788.46

    >>> df = fin.amortization_schedule(principal=200000, annual_rate=6,
    ... years=1, schedule_type='annually', decimal_points=2)
       Month   Payment Principal Interest    Balance
    0                                      200000.00
    1      1  17213.29  16213.29   1000.0  183786.71
    2      2  17213.29  16294.35   918.93  167492.36
    3      3  17213.29  16375.82   837.46  151116.54
    4      4  17213.29   16457.7   755.58  134658.83
    5      5  17213.29  16539.99   673.29  118118.84
    6      6  17213.29  16622.69   590.59  101496.15
    7      7  17213.29  16705.81   507.48   84790.35
    8      8  17213.29  16789.33   423.95   68001.01
    9      9  17213.29  16873.28   340.01   51127.73
    10    10  17213.29  16957.65   255.64   34170.08
    11    11  17213.29  17042.44   170.85   17127.65
    12    12  17213.29  17127.65    85.64      -0.00
    
    Installments = 12
    Interest = 6559.43
    Amount = 206559.43
    """
    principal = ValidateArgs.check_numeric(
        par_name='principal', is_positive=True, user_input=principal
    )
    annual_rate = ValidateArgs.check_numeric(
        par_name='annual_rate',
        limits=[0, 100],
        is_positive=True,
        user_input=annual_rate
    )
    if annual_rate <= 1: # possibly given as a proportion and not percentage
        warnings.warn(
            f"The rate of {annual_rate} was converted to a percentage "
            f"({annual_rate})."
        )
        annual_rate *= 100
    years = ValidateArgs.check_numeric(
        par_name='years', is_positive=True, user_input=years
    )
    schedule_type = ValidateArgs.check_member(
        par_name='schedule_type',
        valid_items=['monthly', 'annual', 'annually', 'yearly'],
        user_input=schedule_type
    )
    auto_display = ValidateArgs.check_boolean(user_input=auto_display, default=True)
    
    if schedule_type == 'monthly':
        # Calculate monthly interest rate and number of payments
        interest_rate = annual_rate / 100 / 12
        total_payments = years * 12
        # Calculate monthly payment
        if interest_rate == 0:
            payment = principal / total_payments
        else:
            payment = (principal * interest_rate) / (1 - (1 + interest_rate) ** -total_payments)

        # Initialize schedule list
        schedule = [[nan, nan, nan, nan, principal]]
        remaining_balance = principal

        for payment_number in range(1, total_payments + 1):
            interest_payment = remaining_balance * interest_rate
            principal_payment = payment - interest_payment
            remaining_balance -= principal_payment
            
            # Append the data for this payment to the schedule list
            schedule.append([
                payment_number,
                payment,
                principal_payment,
                interest_payment,
                remaining_balance
            ])
    else: # annual
        # Calculate annual interest rate and number of payments
        interest_rate = annual_rate / 100
        total_payments = years
        # Calculate annual payment
        if interest_rate == 0:
            payment = principal / total_payments
        else:
            payment = (principal * interest_rate) / (1 - (1 + interest_rate) ** -total_payments)

        # Initialize schedule list
        schedule = [[nan, nan, nan, nan, principal]]
        remaining_balance = principal

        for payment_number in range(1, total_payments + 1):
            interest_payment = remaining_balance * interest_rate
            principal_payment = payment - interest_payment
            remaining_balance -= principal_payment
            
            # Append the data for this payment to the schedule list
            schedule.append([
                payment_number,
                payment,
                principal_payment,
                interest_payment,
                remaining_balance
            ])
    columns = [
        'Month' if schedule_type == 'monthly' else 'Year',
        'Payment',
        'Principal',
        'Interest',
        'Balance'
    ]
    dframe = DataFrame(schedule, columns=columns)
    dframe[columns[0]] = dframe[columns[0]].fillna(0).astype('int')
    
    # must be here
    payment_installments = dframe.shape[0] - 1
    total_interest = dframe['Interest'][1:].sum()
    total_amount = principal + total_interest
    lst = [principal, total_amount, total_interest]
    
    dframe = fround(x=dframe, decimal_points=decimal_points)
    dframe.iloc[0, 0] = nan
    dframe = dframe.fillna('')
    
    plot_html = gph_barchart_simple(values=lst)
    if auto_display:
        display_results({
            'dframe': dframe,
            'Installments': payment_installments,
            'Interest': total_interest,
            'Amount': total_amount,
            'decimal_points': decimal_points
        })
        display(plot_html)
        
    result = Result(
        dframe=dframe,
        installments=payment_installments,
        principal=principal,
        interest=total_interest,
        amount=total_amount,
        plot=plot_html
    )

    return result