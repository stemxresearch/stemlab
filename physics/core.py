from stemlab.core.decimals import fround
from stemlab.core.base.computations import make_subject_and_solve
from stemlab.core.base.functions import get_function_name
from stemlab.core.validators.validate import ValidateArgs


def newtons_second_law(
    force: float | None = None,
    mass: float | None = None,
    acceleration: float | None = None,
    decimal_points=-1
):
    """
    Calculate force (F), mass (m) or acceleration (a) using 
    Newton's Second Law.

    Parameters
    ----------
    force : float
        The force (F).
    mass : float
        Mass (m) of the object in kilograms.
    acceleration : float
        Acceleration (a) of the object in meters per second squared.
    decimal_points : int, optional (default=-1)
        Number of decimal points to round the result to.

    Returns
    -------
    result : float
        Calculated value based on provided inputs.
        
    References
    ----------
    https://www.varsitytutors.com/ap_physics_1-help/newton-s-second-law

    Examples
    --------
    >>> import stemlab.physics as phy
    
    **Example 1:** How much horizontal net force is required to 
    accelerate a 1,000 kg car at 4 m/s^2?
    
    >>> m = 1000
    >>> a = 4
    >>> phy.newtons_second_law(mass=m, acceleration=a)
    4000.0
    
    **Example 2:** If there is a block of mass 2 kg, and a force of 
    20 N is acting on it in the positive x-direction, and a force of 
    30 N in the negative x-direction, then what would be its 
    acceleration?

    >>> f = 20 - 30
    >>> a = 2
    >>> phy.newtons_second_law(force=f, acceleration=a)
    -5.0

    **Example 3:** What net force is required to accelerate a car at 
    a rate of 2 m/s^2 if the car has a mass of 3,000 kg?
    
    >>> m = 3000
    >>> a = 2
    >>> phy.newtons_second_law(mass=m, acceleration=a)
    6000.0
    
    **Example 4:** What is the mass of a truck if it produces a force 
    of 14,000 N while accelerating at a rate of 5 m/s^2 ?
    
    >>> f = 14000
    >>> a = 5
    >>> phy.newtons_second_law(force=f, acceleration=a)
    2800.0
    
    **Example 5:** Your own car has a mass of 2,000 kg. If your car 
    produces a force of 5,000 N, how fast will it accelerate?
    
    >>> f = 5000
    >>> m = 2000
    >>> phy.newtons_second_law(force=f, mass=m)
    2.5
    
    **Example 6:** Even though she is way ahead of you, Sally switches 
    her car to run on nitrous oxide fuel. The nitrous oxide allows her 
    car to develop 10,000 N of force. What is Sally's acceleration if 
    her car has a mass of 500 kg?
    
    >>> f = 10000
    >>> m = 500
    >>> phy.newtons_second_law(force=f, mass=m)
    20.0
    
    **Example 7:** A man of mass 50 kg on the top floor of a skyscraper 
    steps into an elevator. What is the man's weight as the elevator 
    accelerates downward at a rate of 1.5 m/s^2?
    
    >>> m = 50
    >>> a = 9.81 - 1.5 # (g - a)
    >>> phy.newtons_second_law(mass=m, acceleration=a)
    415.5
    """
    if force is not None:
        force = ValidateArgs.check_numeric(par_name='force', user_input=force)
    if mass is not None:
        mass = ValidateArgs.check_numeric(par_name='mass', is_positive=True, user_input=mass)
    if acceleration is not None:
        acceleration = ValidateArgs.check_numeric(
            par_name='acceleration', user_input=acceleration
        )
    
    f, m, a = force, mass, acceleration
    args_dict = {'f': f, 'm': m, 'a': a}
    ValidateArgs.check_args_count(ftn_name=get_function_name(), args_dict=args_dict)
    fexpr = 'f - m * a'
    result = make_subject_and_solve(
        dct=args_dict, fexpr=fexpr,  initial_guess=1
    )
    result = float(fround(x=result, decimal_points=decimal_points))
    
    return result


def ohms_law(
    voltage: float | None = None,
    current: float | None = None,
    resistance: float | None = None,
    decimal_points: int = -1
):
    """
    Calculate voltage (V), current (I), or resistance (R) using Ohm's Law.

    Parameters
    ----------
    voltage : float, optional (default=None)
        Voltage in volts.
    current : float, optional (default=None)
        Current in amperes.
    resistance : float, optional (default=None)
        Resistance in ohms.
    decimal_points : int, optional (default=-1)
        Number of decimal points to round the result to.

    Returns
    -------
    result : float
        Calculated value based on provided inputs.

    Examples
    --------
    >>> import stemlab.physics as phy
    
    **Example 1:** If the resistance of an electric iron is 50 Ω and 
    a current of 3.2 A flows through the resistance. Find the voltage 
    between two points.
    
    >>> i = 50
    >>> r = 3.2
    >>> phy.ohms_law(current=i, resistance=r)
    160.0
    
    **Example 2:** An EMF source of 8.0 V is connected to a purely 
    resistive electrical appliance (a light bulb). An electric current 
    of 2.0 A flows through it. Consider the conducting wires to be 
    resistance-free. Calculate the resistance offered by the electrical 
    appliance.

    >>> v = 8
    >>> i = 2
    >>> phy.ohms_law(voltage=v, current=i)
    4.0
    """
    if voltage is not None:
        voltage = ValidateArgs.check_numeric(
            par_name='voltage', is_positive=True, user_input=voltage
        )
    if current is not None:
        current = ValidateArgs.check_numeric(
            par_name='current', is_positive=True, user_input=current
        )
    if resistance is not None:
        resistance = ValidateArgs.check_numeric(
            par_name='resistance', is_positive=True, user_input=resistance
        )
    
    v, i, r = voltage, current, resistance
    args_dict = {'v': v, 'i': i, 'r': r}
    ValidateArgs.check_args_count(
        ftn_name=get_function_name(), args_dict=args_dict
    )
    fexpr = 'v - i * r'
    result = make_subject_and_solve(
        dct=args_dict, fexpr=fexpr,  initial_guess=1
    )
    result = float(fround(x=result, decimal_points=decimal_points))
    
    return result