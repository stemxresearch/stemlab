from IPython.display import display, HTML
from numpy import array
from pandas import DataFrame, Series, crosstab
from scipy.stats import chi2_contingency, fisher_exact, kendalltau
from scipy.stats.contingency import association # has cramer's v
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from statsmodels.stats import inter_rater as irr

from stemlab.core.display import Result
from stemlab.core.arraylike import conv_to_arraylike
from stemlab.core.validators.validate import ValidateArgs
from stemlab.core.datatypes import ListArrayLike
from stemlab.core.htmlatex import tex_table_to_latex


def add_totals(
    dframe: DataFrame,
    row_label: str = 'Total',
    col_label: str = 'Total',
    custom_row_totals: Series | None = None,
    custom_col_totals: Series | None = None
) -> DataFrame:
    """
    Adds a row total and a column total to the DataFrame.

    Parameters
    ----------
    dframe : pandas.DataFrame
        The input DataFrame to which totals will be added.
    row_label : str, optional (default='Total')
        The label for the row that contains the column totals.
    col_label : str, optional (default='Total')
        The label for the column that contains the row totals.

    Returns
    -------
    dframe : pandas.DataFrame
        A new DataFrame with the added totals.
    
    Examples
    --------
    >>> data = {'Math': [90, 80, 70, 85],
    ...         'English': [88, 79, 95, 85],
    ...         'Science': [95, 85, 89, 92]}
    >>> df = pd.DataFrame(data, index=['Alice', 'Bob', 'Charlie', 'David'])
    >>> df_with_totals = add_totals(df)
    >>> print(df_with_totals)
             Math  English  Science  Total
    Alice      90       88       95    273
    Bob        80       79       85    244
    Charlie    70       95       89    254
    David      85       85       92    262
    Total     325      347      361   1033
    """
    dframe[col_label] = (
        dframe.sum(axis=1) if custom_row_totals is None else custom_row_totals
    )
    dframe.loc[row_label] = (
        dframe.sum(axis=0) if custom_col_totals is None else custom_col_totals
    )

    return dframe

def ctab_chi2_independence(
    dframe: DataFrame, 
    columns: list, 
    correction: bool = True,
    variable_names: list =['variable1', 'variable2'],
    conf_level=0.95,
    decimal_points: int = 2
) -> Result:
    """
    Perform the Chi-square test for independence on a contingency table.

    Parameters
    ----------
    observed : array_like
        The contingency table. A 2D array where each element represents
        the observed frequency count of occurrences for the
        corresponding combination of categorical variables.
        
    correction : bool, optional (default=True)
        If `True`, apply Yates' correction for continuity when the
        contingency table is 2x2. This reduces the Chi-square value,
        making the test more conservative.

    Returns
    -------
    result : Result
        An object containing the attributes:
        statistic : float
            The computed Chi-square statistic.
            
        p_value : float
            The p-value of the test, which indicates the probability of
            observing the data if the null hypothesis is true.
            
        dof : int
            Degrees of freedom for the Chi-square distribution.
        
        expected : ndarray
            The expected frequencies, based on the marginal sums of
            the table, assuming the row and column variables are
            independent.

    Notes
    -----
    The Chi-square test for independence tests the null hypothesis
    that the row and column variables in the contingency table are
    independent. A low p-value suggests that the variables are
    dependent.

    Examples
    --------
    >>> import stemlab as stm
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='nhanes')
    >>> result = sta.ctab_chi2_independence(dframe=df,
    ... columns=['health', 'race'], decimal_points=4)
    """
    dframe = ValidateArgs.check_dframe(par_name='dframe', user_input=dframe)
    columns = conv_to_arraylike(
        array_values=columns, n=2, label='exactly', par_name='columns'
    )
    correction = ValidateArgs.check_boolean(user_input=correction, default=True)
    dframe = dframe[columns].dropna()
    tabulated = crosstab(dframe[columns[0]], dframe[columns[1]])
    row_names = list(tabulated.index)
    col_names = list(tabulated.columns)
    observed = tabulated.values
    chisquare = chi2_contingency(observed=observed, correction=correction)
    expected = chisquare.expected_freq
    zipped = zip(observed.flatten(), expected.flatten())
    combined_table = array(
        [f'{obs} ({round(expfreq, decimal_points)})' for obs, expfreq in zipped]
    )
    combined_table = DataFrame(
            data=combined_table.reshape(observed.shape),
            index=row_names,
            columns=col_names
        )
    observed = add_totals(
        dframe=DataFrame(data=observed, index=row_names, columns=col_names)
    )
    expected = add_totals(
        dframe=DataFrame(data=expected, index=row_names, columns=col_names)
    )
    combined_table['Total'] = expected.iloc[:, -1].astype(int)
    combined_table.loc['Total'] = expected.iloc[-1, :].astype(int)
    latex = tex_table_to_latex(
        data=combined_table.values,
        row_names=combined_table.index,
        col_names=combined_table.columns,
        row_title='Variable',
        caption='Chi-square test of independence, N (Expected)',
        decimal_points=decimal_points
    )
    latex_hypothesis = chisquare_hypothesis(
        chi2=chisquare.statistic,
        dof=chisquare.dof,
        pvalue=chisquare.pvalue,
        variables=variable_names,
        conf_level=conf_level,
        decimal_points=decimal_points
    )
    latex = latex + latex_hypothesis
    display(HTML(latex))
    result = Result(
        observed=observed,
        expected=expected,
        combined=combined_table,
        latex=latex,
        chi2=chisquare.statistic,
        dof=chisquare.dof,
        pvalue=chisquare.pvalue
    )
    
    return result


def chisquare_hypothesis(
    chi2: float,
    dof: int,
    pvalue: float,
    variables: list = ['variable1', 'variable2'],
    conf_level: float = 0.95,
    decimal_points: int = 2
):
    
    variable1, variable2 = variables
    description = [
        '\\mathrm{{H}}_{0}',
        '\\mathrm{{H}}_{1}',
        '\\chi^{2}',
        '\\mathrm{{df}}',
        '\\mathrm{{Decision}}',
        '\\mathrm{{Conclusion}}'
    ]
    sig_level = round(1 - conf_level, decimal_points)
    pvalue = round(pvalue, decimal_points)
    pvalue_str = f'p < .001' if pvalue == 0 else f'p = {pvalue}'
    if pvalue <= sig_level:
        decision = (
            f'\\text{{Reject $H_{{0}}$ since p-value = {pvalue} is less than '
            f'the level of significance $({sig_level})$.}}'
        )
        conclusion = (
            f'\\text{{There is a statistically significant association '
            f'between {variable1} and {variable2}, $\\chi^{{2}}({dof}) = '
            f'{round(chi2, 2)}, {pvalue_str}$.}}'
        )
    else:
        decision = (
            f'\\text{{Fail to reject $H_{{0}}$ since p-value = {pvalue} is '
            f'greater than the level of significance $({sig_level})$.}}'
        )
        conclusion = (
            f'\\text{{There is NO association between '
            f'{variable1} and {variable2}, $\\chi^{{2}}({dof}) = {round(chi2, 2)}, '
            f'{pvalue_str}$.}}'
        )
    chi2 = round(chi2, decimal_points)
    details = [
        f'\\text{{There is NO association between {variable1} and {variable2}}}.',
        f'\\text{{There is an association between {variable1} and {variable2}}}.\\hline',
        chi2,
        dof,
        decision,
        conclusion
    ]
    table_latex = tex_table_to_latex(
        data=array([description, details]).T,
        row_names=None,
        col_names=None,
        row_title='this',
        caption='',
        decimal_points=decimal_points
    ).replace('l|r', 'll', 1).replace('\\hline', '', 1)
    
    return table_latex


def ctab_chi2_independence_stats(
    observed: ListArrayLike, correction=True
) -> Result:
    dframe = ValidateArgs.check_dframe(par_name='dframe', user_input=dframe)
    columns = conv_to_arraylike(
        array_values=columns, n=2, label='exactly', par_name='columns'
    )
    correction = ValidateArgs.check_boolean(user_input=correction, default=True)
    dframe = dframe[columns].dropna()
    row_names = list(dframe.index)
    col_names = list(dframe.columns)
    observed = dframe.values
    chisquare = chi2_contingency(observed=observed, correction=correction)
    expected = chisquare.expected_freq
    zipped = zip(observed.flatten(), expected.flatten())
    combined_table = array([f'{obs} ({round(exp, 2)})' for obs, exp in zipped])
    combined_table = DataFrame(
        data=combined_table.reshape(observed.shape),
        index=row_names,
        columns=col_names
    )
    result = Result(
        observed=observed,
        expected=expected,
        observed_expected=combined_table,
        statistic=chisquare.statistic,
        dof=chisquare.dof,
        pvalue=chisquare.pvalue
    )
    
    return result


def ctab_chi2_goodness_of_fit(self, expected):
    chi2, p = chi2_contingency([self.data.flatten(), expected], correction=False)[:2]
    return chi2, p


def ctab_chi2_homogeneity(self):
    chi2, p, _, _ = chi2_contingency(self.data)
    return chi2, p


def ctab_yates_correction(self):
    chi2, p, _, _ = chi2_contingency(self.data, correction=True)
    return chi2, p


def ctab_mcnemar(self):
    result = mcnemar(self.data)
    return result.statistic, result.pvalue


def ctab_chi2_trend(self):
    chi2, p, _, _ = chi2_contingency(self.data)
    return chi2, p


def ctab_mantel_haenszel(self):
    result = irr.mantel_haenszel(self.data)
    return result[0], result[1]


def ctab_likelihood_ratio(self):
    chi2, p, _, _ = chi2_contingency(self.data, lambda_=False)
    return chi2, p


def ctab_chi2_variance(self):
    chi2, p = chi2_contingency(self.data)
    return chi2, p


def ctab_cochrans_q(self):
    result = cochrans_q(self.data)
    return result.statistic, result.pvalue


def ctab_fisher_exact(self):
    odds_ratio, p = fisher_exact(self.data)
    return odds_ratio, p


def ctab_g_test(self):
    chi2, p = None, None
    return chi2, p


def ctab_cochran_mantel_haenszel(self):
    result = irr.mantel_haenszel(self.data)
    return result[0], result[1]


def ctab_log_linear(self):
    # Log-linear analysis requires more advanced statistical methods
    raise NotImplementedError("Log-linear analysis is not implemented.")


def ctab_kruskall_tau(self):
    gamma = kendalltau(self.data[:, 0], self.data[:, 1])[0]
    return gamma


def ctab_kendall_tau(self):
    tau, _ = kendalltau(self.data[:, 0], self.data[:, 1])
    return tau


def ctab_cramers_v(self):
    return None
