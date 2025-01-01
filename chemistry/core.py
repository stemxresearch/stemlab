from stemlab.core.decimals import fround
from stemlab.core.base.computations import make_subject_and_solve
from stemlab.core.base.functions import get_function_name
from stemlab.core.validators.validate import ValidateArgs


def ideal_gas_law(
    pressure: float | None = None,
    moles: float | None = None,
    volume: float | None = None,
    temperature: float | None = None,
    gas_constant: float = 0.0821,
    decimal_points: int = -1
) -> float:
    """
    References
    ----------
    https://www.chemteam.info/GasLaw/Gas-Ideal.html
    https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Physical_Properties_of_Matter/States_of_Matter/Properties_of_Gases/Gas_Laws/The_Ideal_Gas_Law
    
    Examples
    --------
    >>> import stemlab.chemistry as che
    
    **Example 1:** A sample of gas at 25.0 °C has a volume of 11.0 L 
    and exerts a pressure of 660.0 mmHg. How many moles of gas are in 
    the sample?
    
    >>> t = 25 + 273 # temperature from Celcius to Kelvin
    >>> p = 660 / 760 # pressure from mmHg to atm
    >>> v = 11
    >>> che.ideal_gas_law(pressure=p, volume=v, temperature=t)
    0.39044836379548
    
    **Example 2:** Calculate the approximate volume of a 0.400 mol 
    sample of gas at 11.0 °C and a pressure of 2.43 atm.
    
    >>> t = 11 + 273 # temperature from Celcius to Kelvin
    >>> p = 2.43
    >>> n = 0.4
    >>> che.ideal_gas_law(pressure=p, moles=n, temperature=t)
    3.83809053497942
    
    **Example 3:** Calculate the approximate temperature of a 0.300 mol 
    sample of gas at 780 mmHg and a volume of 6.00 L.
    
    >>> p = 780 / 760
    >>> n = 0.3
    >>> v = 6
    >>> che.ideal_gas_law(pressure=p, moles=n, volume=v)
    250.0160266683762
    
    **Example 4:** What is the pressure exerted by 2.3 mol of a gas 
    with a temperature of 40 °C and a volume of 3.5 L?
    
    >>> t = 40 + 273
    >>> n = 2.3
    >>> v = 3.5
    >>> che.ideal_gas_law(moles=n, volume=v, temperature=t)
    16.8867971428571
    """
    if pressure is not None:
        pressure = ValidateArgs.check_numeric(
            par_name='pressure', is_positive=True, user_input=pressure
        )
    if volume is not None:
        volume = ValidateArgs.check_numeric(
            par_name='volume', is_positive=True, user_input=volume
        )
    if moles is not None:
        moles = ValidateArgs.check_numeric(
            par_name='moles', is_positive=True, user_input=moles
        )
    if gas_constant is not None:
        gas_constant = ValidateArgs.check_numeric(
            par_name='gas_constant', is_positive=True, user_input=gas_constant
        )
    if temperature is not None:
        temperature = ValidateArgs.check_numeric(
            par_name='temperature', is_positive=True, user_input=temperature
        )
    
    p, n, r, t, v = pressure, moles, gas_constant, temperature, volume
    args_dict = {'p': p, 'n': n, 'r': r, 't': t, 'v': v}
    ValidateArgs.check_args_count(ftn_name=get_function_name(), args_dict=args_dict)
    fexpr = 'p - (n * r * t) / v'
    result = make_subject_and_solve(
        dct=args_dict, fexpr=fexpr,  initial_guess=1
    )
    result = float(fround(x=result, decimal_points=decimal_points))
    
    return result