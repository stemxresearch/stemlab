from typing import Literal

from numpy import sqrt
from scipy.stats import t, norm
from stemlab.core.display import Result, display_results


def tz_alternative(
    test_type: Literal['p', 't', 'z'] = 't', 
    alternative: Literal['less', 'greater', 'two-sided'] = 'two-sided', 
    dfn: int | float | None = None, 
    sig_level: float = 0.05,
    auto_display: bool = True,
    decimal_points: int = 4
) -> Result:
    """
    Determine the hypothesis symbol and critical value for hypothesis testing.

    Parameters
    ----------
    test_type : {'p', 't', 'z'}, optional (default='t')
        Type of test ('p' for proportions, 't' for t-test, 'z' for z-test).
    alternative : {'less', 'greater', 'two-sided'}
        Type of alternative hypothesis.
    dfn : {float, int}, optional (default=4)
        Degrees of freedom for the t-test.
    sig_level : float
        Significance level (alpha).
    decimal_points : int, optional (default=4)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    alternative_symbol : str
        A string with the Latex symbol for alternative hypothesis
        (<, >, \\ne for not equal to)
    crit_value: float
        The critical value of the t or z test 
        (this is the value from the distribution table).
    """
    crit_value = 0 # placeholder for invalid test_type
    if alternative == 'less':
        alternative_symbol = '<'
        if test_type == 't':
            crit_value = abs(t.ppf(q=sig_level / 2, df=dfn))
        else:
            crit_value = abs(norm.ppf(q=sig_level / 2))
    elif alternative == 'greater':
        alternative_symbol = '>'
        if test_type == 't':
            crit_value = abs(t.ppf(q=1 - sig_level / 2, df=dfn))
        else:
            crit_value = abs(norm.ppf(q=1 - sig_level / 2))
    else:
        alternative_symbol = '\\ne'
        if test_type == 't':
            crit_value = abs(t.ppf(q=1 - sig_level, df=dfn))
        else:
            crit_value = abs(norm.ppf(q=1 - sig_level))
            
    if isinstance(crit_value, (tuple, list)):
        crit_value = crit_value[0]
        
    crit_value = round(crit_value, decimal_points)
        
    if auto_display:
        result = alternative_symbol, crit_value
    else:
        result = Result(
            alternative_symbol=alternative_symbol, crit_value=crit_value
        )
    
    return result


def hyp_corr_tpvalue(
    corr: float, 
    n: int, 
    alternative: Literal['less', 'greater', 'two-sided'] = 'two-sided', 
    sig_level: float = 0.05,
    auto_display: bool = True,
    decimal_points: int = 4
) -> Result:
    """
    Calculate the t-statistic and p-value for correlation coefficient.

    Parameters
    ----------
    corr : float
        The correlation coefficient.
    n : int
        The sample size.
    alternative : {'less', 'greater', 'two-sided'}, optional (default='two-sided')
        Type of alternative hypothesis.
    sig_level : float, optional (default=0.05)
        Significance level (alpha).
    auto_display : bool, optional (default=True)
        If `True`, results will be displayed automatically.
    decimal_points : int, optional (default=4)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    tstat : float
        The t-statistic value.
    pvalue : float
        The p-value associated with the t-statistic.
    conclusion : str
        A conclusion based on the significance level.
        
    Examples
    --------
    >>> import stemlab as sta
    >>> sta.hyp_corr_tpvalue(corr=0.7, n=50, alternative='two-sided',
    ... sig_level=0.05)
    Result(
        tstat: 6.791
        pvalue: 0.0
        conclusion: Correlation is statistically significant
    )
    
    >>> sta.hyp_corr_tpvalue(corr=-0.5, n=30, alternative='less',
    ... sig_level=0.05)
    Result(
        tstat: -3.0551
        pvalue: 0.002
        conclusion: Correlation is statistically significant
    )
    
    >>> sta.hyp_corr_tpvalue(corr=0.3069, n=25, alternative='greater',
    ... sig_level=0.05)
    Result(
        tstat: 1.5465
        pvalue: 0.068
        conclusion: Correlation is NOT statistically significant
    )
    """
    dfn = n - 2
    tstat = round(corr * sqrt(n - 2) / sqrt(1 - corr ** 2), decimal_points)
    if alternative == 'less':
        pvalue = t.cdf(x = tstat, df=dfn)
    elif alternative == 'greater':
        pvalue = 1 - t.cdf(x = tstat, df=dfn)
    else:
        pvalue = 2 * t.cdf(x = -abs(tstat), df=dfn)
    pvalue = round(pvalue, 3)

    if pvalue <= sig_level:
        conclusion = "Correlation is statistically significant"
    else:
        conclusion = "Correlation is NOT statistically significant"
    
    if auto_display:
        display_results(result_dict={
            't': tstat,
            'p': pvalue,
            'conclusion': conclusion
        })

    result = Result(tstat=tstat, pvalue=pvalue, conclusion=conclusion)
    
    return result