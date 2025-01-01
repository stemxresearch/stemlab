
from numpy import log

from stemlab.core import fround
from stemlab.core.display import Result, display_results
from stemlab.core.base.computations import make_subject_and_solve
from stemlab.core.base.functions import get_function_name
from stemlab.core.validators.validate import ValidateArgs


def logistic_growth(
    population: int = None,
    initial_pop: int = None,
    rate: float = None,
    carrying_capacity: int = None,
    time: float | int = None,
    decimal_points: int = -1
) -> float | int:
    """
    Calculate population size at time t (N), initial population (N0),
    Intrinsic rate of increase (r), carrying capacity (K), time (t).

    Parameters
    ----------
    population : int
        Population size at time t (N).
    initial_pop : int
        Initial population size (N0).
    rate : float
        Intrinsic rate of increase (r).
    carrying_capacity : int
        Carrying capacity (K).
    time : {float, int}
        Time period (t).
    decimal_points : int, optional (default=-1)
        Number of decimal points.

    Returns
    -------
    result : {float, int}
        Calculated value based on provided inputs.

    Examples
    --------
    import stemlab.biology as bio
    
    **Example 1:** What is the population at time t=6 in a logistic 
    growth model with an initial population of 50, growth rate of 0.1, 
    and carrying capacity of 500?
     
    >>> N0 = 50
    >>> r = 0.1
    >>> K = 500
    >>> t = 6
    >>> bio.logistic_growth(initial_pop=N0, rate=r,
    ... carrying_capacity=K, time=t)
    84.1849379959107 # you can round this to an integer
    
    **Example 2:** A lake is currently inhabited by 725 fish. The lake 
    has a carrying capacity of 3000 fish. If left unchecked, the fish 
    population would increase by 18% per year. How long will it take 
    for the population to reach 2000 fish using the logistic growth
    model?
    
    >>> N0 = 725
    >>> r = 0.18
    >>> K = 3000
    >>> N = 2000
    >>> bio.logistic_growth(population=N, initial_pop=N0, rate=r,
    ... carrying_capacity=K, time=None)
    10.20394920605733
    
    **Example 3:** A wildlife reserve currently has 450 deer. The 
    reserve can support up to 4000 deer. After 10 years, the 
    population is expected to grow to 2500 deers. What is the annual 
    growth rate of the deer population using the logistic growth model?
    
    >>> N0 = 1450
    >>> K = 4000
    >>> t = 8
    >>> N = 2500
    >>> bio.logistic_growth(population=N, initial_pop=N0, rate=None,
    carrying_capacity=K, time=t)
    0.13441942831298
    
    **Example 4:** A forest is currently home to a population of 200 rabbits. 
    The forest is estimated to be able to sustain a population of 2000 
    rabbits. Absent any restrictions, the rabbits would grow by 50% 
    per year. Predict the future population using the logistic growth 
    model for a period 10 years.
    
    >>> from stemlab.graphics import gph_scatter
    >>> N0 = 200
    >>> r = 0.5
    >>> K = 2000
    >>> time = np.arange(0, 11)
    >>> pop = []
    >>> for t in time:
    ...     p = bio.logistic_growth(initial_pop=N0, rate=r, carrying_capacity=K, time=t)
    ...     pop.append(p)
    >>> dframe = pd.DataFrame([time, pop], index=['Time', 'Population']).T
    >>> display(dframe)
    >>> gph_scatter(time, pop, xlabel='Time', ylabel='Population')
    """
    if population is not None:
        population = ValidateArgs.check_numeric(
            par_name='population', is_positive=True, user_input=population
        )
    if initial_pop is not None:
        initial_pop = ValidateArgs.check_numeric(
            par_name='initial_pop', is_positive=True, user_input=initial_pop
        )
    if rate is not None:
        rate = ValidateArgs.check_numeric(
            par_name='rate', is_positive=True, user_input=rate
        )
    if carrying_capacity is not None:
        carrying_capacity = ValidateArgs.check_numeric(
            par_name='carrying_capacity',
            is_positive=True,
            user_input=carrying_capacity
        )
    if time is not None:
        time = ValidateArgs.check_numeric(
            par_name='time', is_positive=True, user_input=time
        )
        
    N, N0, r, K, t = population, initial_pop, rate, carrying_capacity, time
    args_dict = {'Nt': N, 'N0': N0, 'r': r, 'K': K, 't': t}
    ValidateArgs.check_args_count(ftn_name=get_function_name(), args_dict=args_dict)
    fexpr = 'Nt - (N0 * K * exp(r * t) / ((K - N0) + N0 * exp(r * t)))'
    result = make_subject_and_solve(
        dct=args_dict, fexpr=fexpr,  initial_guess=1
    )
    result = float(fround(x=result, decimal_points=decimal_points))
    
    return result


def michaelis_menten(
    v0: float = None,
    v_max: float = None,
    k_m: float = None,
    s: float = None,
    decimal_points:int = -1
) -> float:
    """
    Calculate the reaction rate using Michaelis-Menten kinetics.

    Parameters
    ----------
    v0 : float
        Rate of the reaction.
    v_max : float
        Maximum rate of the reaction.
    k_m : float
        Michaelis constant.
    s : float
        Substrate concentration.
    decimal_points : int, optional (default=-1)
        Number of decimal points.

    Returns
    -------
    result : float
        Calculated value based on provided inputs.

    Examples
    --------
    >>> import stemlab.biology as bio
    
    **Example 1:** Given a maximum reaction velocity of 100 units, 
    a Michaelis constant of 7 units, and a substrate concentration 
    of 10 units, calculate the reaction velocity using the 
    Michaelis-Menten equation.
    
    >>> v_max = 100
    >>> k_m = 7
    >>> s = 10
    >>> bio.michaelis_menten(v_max=v_max, k_m=k_m, s=s)
    58.8235294117647
    
    **Example 2:** An enzyme hydrolyzed a substrate concentration of 0.03 mmol/L. 
    The initial velocity was 1.5 x 10^(-3) mmol/L·min(-1), and the 
    maximum velocity was 4.5 x 10^(-3) mmol/L·min^(-1). Calculate 
    the k_m value.

    >>> v0 = 1.5e-3
    >>> v_max = 4.5e-3
    >>> s = 0.03
    >>> bio.michaelis_menten(v0=v0, v_max=v_max, s=s)
    0.06
    """
    if v0 is not None:
        v0 = ValidateArgs.check_numeric(par_name='v0', is_positive=True, user_input=v0)
    if v_max is not None:
        v_max = ValidateArgs.check_numeric(
            par_name='v_max', is_positive=True, user_input=v_max
        )
    if s is not None:
        s = ValidateArgs.check_numeric(par_name='s', is_positive=True, user_input=s)
    if k_m is not None:
        k_m = ValidateArgs.check_numeric(par_name='k_m', is_positive=True, user_input=k_m)
    
    args_dict = {'v0': v0, 'v_max': v_max, 's': s, 'k_m': k_m}
    ValidateArgs.check_args_count(ftn_name=get_function_name(), args_dict=args_dict)
    fexpr = 'v0 - v_max * s / (k_m + s)'
    result = make_subject_and_solve(
        dct=args_dict, fexpr=fexpr,  initial_guess=1
    )
    try:
        result = float(fround(x=result, decimal_points=decimal_points))
    except:
        pass
    
    return result


def lotka_volterra(
    pop_prey_x: int,
    pop_preditor_y: int,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
    auto_display: bool = True,
    decimal_points: int = -1
) -> Result:
    """
    Calculate the rate of change of prey and predator populations
    using Lotka-Volterra equations.

    Parameters
    ----------
    pop_prey : int
        Prey population (x).
    pop_preditor : int
        Predator population (y).
    alpha : float
        Growth rate of prey.
    beta : float
        Rate at which predators destroy prey.
    delta : float
        Rate at which predators increase by consuming prey.
    gamma : float
        Death rate of predators.
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=-1)
        Number of decimal points.

    Returns
    -------
    result : Result
        - prey: rate of change of prey population (dx/dt)
        - preditor: rate of change of predator population (dy/dt)

    Examples
    --------
    >>> x = 40
    >>> y = 9
    >>> alpha = 0.1
    >>> beta = 0.02
    >>> delta = 0.01
    >>> gamma = 0.1
    >>> result = bio.lotka_volterra(pop_prey_x=x, pop_preditor_y=y,
    ... alpha=alpha, beta=beta, delta=delta, gamma=gamma)
    """
    x, y = pop_prey_x, pop_preditor_y
    dx_dt = fround(x=alpha * x - beta * x * y, decimal_points=decimal_points)
    dy_dt = fround(x=delta * x * y - gamma * y, decimal_points=decimal_points)
    if auto_display:
        display_results({
            'Prey': dx_dt,
            'Preditor': dy_dt,
            'decimal_points': decimal_points
        })
    result = Result(Prey=dx_dt, Preditor=dy_dt)
    
    return result


def pop_doubling_time(
    growth_rate: float, decimal_point: int = -1
) -> float:
    """
    Calculate the population doubling time.

    Parameters
    ----------
    growth_rate : float
        Growth rate.
    decimal_points : int, optional (default=-1)
        Number of decimal points.

    Returns
    -------
    result : float
        Doubling time.

    Examples
    --------
    >>> r = 0.2
    >>> bio.pop_doubling_time(growth_rate=r)
    3.46573590279973
    """
    result = fround(x=log(2) / growth_rate, decimal_points=decimal_point)
    
    return result 