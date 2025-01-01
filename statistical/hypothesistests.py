import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def test_significance(test_statistic, alpha, test_name, alternative, how_to_test):
    """
    Function to test the significance of a statistical test.

    Parameters
    ----------
    - test_statistic: float, the computed test statistic from the test
    - alpha: float, level of significance (e.g., 0.05)
    - test_name: str, name of the statistical test (e.g., 'chi-square', 'means_ttest')
    - alternative: str, specifies the alternative hypothesis ('lower', 'upper', 'two-sided')
    - how_to_test: str, specifies how to test ('test', 'statistic', 'p-value', 'graph')

    Returns
    -------
    - result: str or matplotlib figure, depending on 'how_to_test' parameter
    """

    # Define the critical value based on the level of significance (alpha)
    if alternative == 'two-sided':
        alpha = alpha / 2  # divide alpha by 2 for two-sided test

    if alternative in ['lower', 'two-sided']:
        critical_value = stats.distributions.norm.ppf(alpha)
    elif alternative == 'upper':
        critical_value = stats.distributions.norm.ppf(1 - alpha)

    # Determine the result based on how_to_test parameter
    if how_to_test == 'test':
        if alternative == 'lower':
            if test_statistic < critical_value:
                result = "Reject Null Hypothesis"
            else:
                result = "Fail to Reject Null Hypothesis"
        elif alternative == 'upper':
            if test_statistic > critical_value:
                result = "Reject Null Hypothesis"
            else:
                result = "Fail to Reject Null Hypothesis"
        elif alternative == 'two-sided':
            if np.abs(test_statistic) > critical_value:
                result = "Reject Null Hypothesis"
            else:
                result = "Fail to Reject Null Hypothesis"

    elif how_to_test == 'statistic':
        result = f"Computed test statistic: {test_statistic}"

    elif how_to_test == 'p-value':
        if alternative == 'lower':
            p_value = stats.norm.cdf(test_statistic)
        elif alternative == 'upper':
            p_value = 1 - stats.norm.cdf(test_statistic)
        elif alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(np.abs(test_statistic)))
        
        result = f"P-value: {p_value}"

    elif how_to_test == 'graph':
        # Generate a graph based on the test_name
        if test_name.lower() == 'chi-square':
            x = np.linspace(0, 30, 400)
            y = stats.chi2.pdf(x, df=1)  # Example: Chi-square distribution with df=1
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, 'b-', label=f'Chi-square Distribution (df=1)')
            plt.axvline(test_statistic, color='r', linestyle='--', label='Test Statistic')
            plt.fill_between(x[x >= test_statistic], 0, y[x >= test_statistic], color='r', alpha=0.3)
            plt.title(f'{test_name} Test - Critical Region (Alpha = {alpha})')
            plt.xlabel('X')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
        elif test_name.lower() == 'f-test':
            x = np.linspace(0, 5, 400)
            y = stats.f.pdf(x, dfn=10, dfd=5)  # Example: F-distribution with dfn=10, dfd=5
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, 'b-', label=f'F Distribution (dfn=10, dfd=5)')
            plt.axvline(test_statistic, color='r', linestyle='--', label='Test Statistic')
            plt.fill_between(x[x >= test_statistic], 0, y[x >= test_statistic], color='r', alpha=0.3)
            plt.title(f'{test_name} Test - Critical Region (Alpha = {alpha})')
            plt.xlabel('X')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
        else:
            # Default to normal distribution plot
            x = np.linspace(-4, 4, 1000)
            y = stats.norm.pdf(x)
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, 'b-', label='Standard Normal Distribution')
            plt.axvline(test_statistic, color='r', linestyle='--', label='Test Statistic')
            plt.fill_between(x[x <= test_statistic], 0, y[x <= test_statistic], color='r', alpha=0.3)
            plt.title(f'{test_name} Test - Critical Region (Alpha = {alpha})')
            plt.xlabel('X')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

        result = plt

    return result

# Example usage:
test_stat = 4.5  # Example test statistic
alpha = 0.05
test = 'chi-square'  # Example test name
alternative = 'upper'
how = 'graph'

result = test_significance(test_stat, alpha, test, alternative, how)
if isinstance(result, str):
    print(result)
elif isinstance(result, plt.Figure):
    plt.show()
