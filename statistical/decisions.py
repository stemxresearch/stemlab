from typing import Literal

from stemlab.core.validators.validate import ValidateArgs


def _pvalue_alternative(
    alternative: Literal['less', 'greater', 'two-sided'], 
    test_stat: float, 
    dfn: int | float | None = None, 
    p_value: float | None = None
) -> tuple[str, str]:
    """
    Format the p-value and provide a phrase for the alternative 
    hypothesis.

    Parameters
    ----------
    alternative : {'less', 'greater', 'two-sided'}
        The alternative hypothesis.
    test_stat : float
        The calculated t-value.
    dfn : {int, float}, optional (default=None)
        The degrees of freedom.
    p_value : float, optional (default=None)
        The p-value.

    Returns
    -------
    pvalue_str : str
        Formatted string representing the p-value.
    alternative : str
        Phrase describing the alternative hypothesis.
    """

    if p_value >= 0.001:
        if dfn is None:
            pvalue_str = f"p = {round(p_value, 3)}"
        else:
            pvalue_str = (
                f"t({round(dfn, 2)}) = {round(test_stat, 2)}, p = {round(p_value, 3)}"
            )
    else:
        pvalue_str = f"p < .001"
    # add trailing zeros -> must be a loop because of the varying `n`
    for n in range(9):
        pvalue_str = pvalue_str.replace(f'.{n})', f'.{n}0)')
        pvalue_str = pvalue_str.replace(f'.{n},', f'.{n}0,')
        pvalue_str = pvalue_str.replace(f'.0)', f'.00)')
        pvalue_str = pvalue_str.replace(f'.0,', f'.00,')
    
    pvalue_str = pvalue_str.replace('p = 1', 'p = 1.00')

    # direction in the conclusion
    alternative_map = {
        'less': 'less than',
        'greater': 'greater than',
        'two-sided': 'different from'
    }
    alternative = alternative_map.get(alternative, 'different from')

    return pvalue_str, alternative


def _conclusion_str(conclusion_str: str) -> str:
    """
    Ensure that decimal numbers in a string have two decimal places by 
    adding trailing zeros.

    Parameters
    ----------
    conclusion_str : str
        The string containing numerical values.

    Returns
    -------
    str
        The string with decimal numbers formatted.
    """
    # remove the leading zero in float
    conclusion_str.replace('(0.', '(.').replace(' 0.', '.')
    # remove trailing zero in integers
    conclusion_str = conclusion_str.replace('.0\\', '\\')
    for n in range(9):
        conclusion_str = conclusion_str\
        .replace(f'.{n})', f'.{n}0)')\
        .replace(f'.{n},', f'.{n}0,')\
        .replace(f'.0)', f'.00)')\
        .replace(f'.0,', f'.00,')

    return conclusion_str


def test_decision(p_value: float, sig_level: float) -> str:
    """
    Make a decision about the null hypothesis based on the p-value and 
    significance level.

    Parameters
    ----------
    p_value : float
        The p-value calculated from the hypothesis test.
    sig_level : float
        The chosen level of significance.

    Returns
    -------
    decision_str : str
        A string indicating whether to reject or fail to reject the 
        null hypothesis.
    """
    # p_value
    p_value = ValidateArgs.check_numeric(
        par_name='p_value', 
        limits=[0, 1], 
        boundary='inclusive', 
        user_input=p_value
    )
    
    # sig_level
    sig_level = ValidateArgs.check_member(
        par_name='sig_level', 
        valid_items=[.1, .05, .01],
        is_string=False,
        user_input=sig_level
    )

    if p_value <= sig_level:
        decision_str = (
            f"Reject \(\\text{{H}}_{{0}}\) "
            f"since the p-value \(({round(p_value, 3)})\) is less "
            f"than the level of significance \(({round(sig_level, 3)})\)."
        )
    else:
        decision_str = (
            f"Fail to reject \(\\text{{H}}_{{0}}\) "
            f"since the p-value \(({round(p_value, 3)})\) is greater "
            f"than the level of significance \(({round(sig_level, 3)})\)."
        )
    decision_str = decision_str\
    .replace('(0.', '(.')\
    .replace(' 0.', '.')\
    .replace('1.0)', '1.000)')\
    .replace('.0)', '.000)')

    return decision_str


def test_conclusion(
    test_name,
    sample1_name,
    sample2_name,
    mean1, mean2,
    pop_mean,
    std1,
    std2,
    test_stat,
    dfn,
    alternative,
    p_value,
    sig_level
):
    """
    Make a decision about the null hypothesis based on the test 
    results.

    Parameters
    ----------
    test_name : str
        The name of the hypothesis test.
    sample1_name : str
        The name of the first sample.
    sample2_name : str
        The name of the second sample.
    mean1 : float
        The mean of the first sample.
    mean2 : float
        The mean of the second sample.
    pop_mean : float
        The population mean.
    std1 : float
        The standard deviation of the first sample.
    std2 : float
        The standard deviation of the second sample.
    test_stat : float
        The calculated t-value.
    dfn : int
        The degrees of freedom.
    alternative : str
        The alternative hypothesis ('less', 'two-sided', 'greater').
    p_value : float
        The p-value of the test.
    sig_level : float
        The significance level.

    Returns
    -------
    str
        A string with the conclusion statement.
    """
    # test_name
    test_names = [
        'one-sample-p', 'two-samples-p', 'paired-samples-p',
        'one-sample-z', 'two-samples-z', 'paired-samples-z',
        'one-sample-t', 'two-samples-t', 'paired-samples-t'
    ]
    test_name = ValidateArgs.check_string(
        par_name='test_name', user_input=test_name
    )
    
    test_name = ValidateArgs.check_member(
        par_name='test_name', valid_items=test_names, user_input=test_name
    )

    # alternative,
    alternatives = ['less', 'two-sided', 'greater']
    alternative = ValidateArgs.check_string(par_name='alternative', user_input=alternative)
    alternative = ValidateArgs.check_member(
        par_name='alternative', 
        valid_items=alternatives, 
        user_input=alternative
    )

    # p_value
    p_value = ValidateArgs.check_numeric(
        par_name='p_value', 
        limits=[0, 1], 
        boundary='inclusive', 
        user_input=p_value
    )

    # sig_level
    sig_level = ValidateArgs.check_member(
        par_name='sig_level', 
        valid_items=[.1, .05, .01],
        is_string=False,
        user_input=sig_level
    )

    zt_tests = [
        'two-samples-z', 'paired-samples-z', 'two-samples-t', 'paired-samples-t'
    ]
    
    if test_name == 'one-sample-p':
        conclusion_str = _p1_sample(
            alternative, mean1, pop_mean, test_stat, p_value, sig_level
        )
        
    elif test_name in ['two-samples-p', 'paired-samples-p']:
        conclusion_str = _p2_samples(alternative, p_value, sig_level)

    elif test_name in ['one-sample-z', 'one-sample-t']:
        conclusion_str = _test_one_sample(
            alternative=alternative,
            mean1=mean1,
            pop_mean=pop_mean,
            std1=std1,
            test_stat=test_stat,
            dfn=dfn,
            p_value=p_value,
            sig_level=sig_level
        )
    elif test_name in zt_tests:
        conclusion_str = _test_two_samples(
            alternative=alternative,
            sample1_name=sample1_name,
            sample2_name=sample2_name,
            mean1=mean1,
            mean2=mean2,
            std1=std1,
            std2=std2,
            test_stat=test_stat,
            dfn=dfn,
            p_value=p_value,
            sig_level=sig_level
        )

    return conclusion_str


def _p1_sample(alternative, p_value, sig_level):
    pass


def _p2_samples(alternative, p_value, sig_level):
     pass


def _z1_sample(alternative, p_value, sig_level):
     pass


def _z2_samples(alternative, p_value, sig_level):
     pass


def _test_one_sample(
    alternative,
    mean1,
    pop_mean,
    std1,
    test_stat,
    dfn,
    p_value,
    sig_level
):
    """
    Perform conclusion for one-sample hypothesis test.
    """
    pvalue_str, alternative = _pvalue_alternative(
        alternative=alternative,
        test_stat=test_stat,
        dfn=dfn,
        p_value=p_value
    )
    
    if p_value <= sig_level: # significant
        conclusion_str = (
            f"The sample mean "
            f"\((M = {round(mean1, 2)}, SD = {round(std1, 2)})\) is "
            f"{alternative} the population mean \((M = {round(pop_mean, 2)})\). "
            f"The mean difference of \({round(abs(mean1 - pop_mean), 2)}\) is "
            f"statistically significant at "
            f"\({round(sig_level, 3) * 100}\%, {pvalue_str}\)."
        )
    else: # not significant
        conclusion_str = (
            f"The sample mean "
            f"\((M = {round(mean1, 2)}, SD = {round(std1, 2)})\) "
            f"is not statistically significantly {alternative} the "
            f"population mean "
            f"\((M = {round(pop_mean, 2)}), \\,{pvalue_str}\)."
        )
    
    conclusion_str = _conclusion_str(conclusion_str)

    return conclusion_str


def _test_two_samples(
    alternative,
    sample1_name,
    sample2_name,
    mean1,
    mean2,
    std1,
    std2,
    test_stat, 
    dfn,
    p_value,
    sig_level
):
    """
    Perform conclusion for two-sample hypothesis test.
    """     
    pvalue_str, alternative = _pvalue_alternative(
        alternative=alternative,
        test_stat=test_stat,
        dfn=dfn,
        p_value=p_value
    )
    
    sample1_name = sample1_name.replace('_', ' ')
    sample2_name = sample2_name.replace('_', ' ')
    if p_value <= sig_level: # significant
        conclusion_str = (
            f"The sample mean of \(\\textbf{{{sample1_name}}}\) "
            f"\((M = {round(mean1, 2)}, SD = {round(std1, 2)})\) "
            f"is {alternative} that of \(\\textbf{{{sample2_name}}}\) "
            f"\((M = {round(mean2, 2)}, SD = {round(std2, 2)})\). "
            f"The mean difference of "
            f"\({round(abs(mean1 - mean2), 2)}\) is statistically "
            f"significant at \({round(sig_level, 3) * 100}\%, {pvalue_str}\)."
        )
    else: # not significant
        conclusion_str = (
            f"The sample mean of \(\\textbf{{{sample1_name}}}\) "
            f"\((M = {round(mean1, 2)}, SD = {round(std1, 2)})\) "
            f"is not statistically significantly {alternative} "
            f"that of \(\\textbf{{{sample2_name}}}\) "
            f"\((M = {round(mean2, 2)}, SD = {round(std2, 2)}), \\,{pvalue_str}\)."
        )
    
    conclusion_str = _conclusion_str(conclusion_str)

    return conclusion_str