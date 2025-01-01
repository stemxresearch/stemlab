import numpy as np
from scipy.stats import t, norm, binom_test, binomtest, rankdata, kruskal
import pandas as pd
import matplotlib.pyplot as plt

from stemlab.core.htmlatex import sta_dframe_to_html


def draw_z_score(xx, cond, mu, sigma, title):
    """
    Draw the z-score.
    """

    yy = norm.pdf(xx, mu, sigma)
    zz = xx[cond]
    plt.plot(xx, yy)
    plt.fill_between(zz, 0, norm.pdf(zz, mu, sigma))
    plt.title(title)
    plt.tight_layout()


def graph_t(xx, mu, sigma, n, alternative, alpha):
    """
    Graph t-values.
    """
    t_lower = t.ppf(alpha / 2, n - 2)
    t_upper = t.ppf(1 - alpha / 2, n - 2)

    if alternative == "two-sided":
        xx = np.arange(-4, 4, 0.001)
        zz0 = t_upper
        draw_z_score(
            xx=xx,
            cond=(-zz0 < xx) & (xx > zz0),
            mu=mu,
            sigma=sigma,
            title=f"${np.round(t_lower, decimal_points = 4)} < t < {np.round(t_upper, decimal_points = 4)}$",
        )

    elif alternative == "less":
        xx = np.arange(-4, 4, 0.001)
        zz0 = t_lower
        draw_z_score(
            xx=xx,
            cond=xx < zz0,
            mu=mu,
            sigma=sigma,
            title=f"$ t < {np.round(t_lower, decimal_points = 4)} $",
        )

    elif alternative == "greater":
        xx = np.arange(-4, 4, 0.001)
        zz0 = t_upper
        draw_z_score(
            xx=xx,
            cond=xx > zz0,
            mu=mu,
            sigma=sigma,
            title=f"$ t > {np.round(t_upper, decimal_points = 4)} $",
        )


def decision_kruskal(n, p_value, alpha, decimal_points=4):
    """
    Kruskal Wallis test
    """
    decision = '<p style="font-weight:bold;margin-top:15px;"><span class="border_bottom_span">Decision</span></p>'
    conclusion = '<p style="font-weight:bold;"><span class="border_bottom_span">Conclusion</span></p>'

    if p_value <= alpha:
        decision += f"<p>Reject the null hypothesis since the p-value <strong>({round(p_value, 3)})</strong> is less than the level of significance <strong>({round(alpha, 2)})</strong>.</p>"
        conclusion += (
            f"<p>There exists one group whose median is different from the others.</p>"
        )
    else:
        decision += f"<p>Fail to reject the null hypothesis since the p-value <strong>({round(p_value, 3)})</strong> is greater than the level of significance <strong>({round(alpha, 2)})</strong>.</p>"
        conclusion += f"<p>All the {n} groups have the same median.</p>"

    result = f"{decision} {conclusion}"

    return result


def kruskal_inference(
    samples, alternative, decision_method, alpha=0.05, decimal_points=4
):
    """
    Kruskal Wallis inference.
    """

    chi_value, p_value = kruskal(*samples)

    results_title = '<p style="font-weight:bold;margin-top:15px;margin-bottom:20px;"><span class="border_bottom_span">Results table</span></p>'
    n = len(samples)
    N, Median, Sum_Rank, Mean_Rank = [], [], [], []
    for i in range(len(samples)):
        ranked = rankdata(samples[i])
        N.append(len(samples[i]))
        Median.append(np.median(samples[i]))
        Sum_Rank.append(np.sum(ranked))
        Mean_Rank.append(np.mean(ranked))

    chi_value_col = [chi_value] + [np.nan] * (n - 1)
    p_value_col = [p_value] + [np.nan] * (n - 1)
    np_table = np.round(
        np.array(
            [N, Median, Sum_Rank, Mean_Rank, chi_value_col, p_value_col],
            dtype=np.float64,
        ),
        decimal_points,
    )
    table_results = pd.DataFrame(np_table).T
    table_results = table_results.fillna("")
    table_results.index = np.arange(1, table_results.shape[0] + 1)
    table_results.columns = [
        "N",
        "Median",
        "Sum Rank",
        "Mean Rank",
        "H-value",
        "P-value",
    ]
    table_results = sta_dframe_to_html(table_results, row_title="Group")

    hypothesis = '<p style="font-weight:bold;margin-top:15px;"><span class="border_bottom_span">Hypothesis<span></p>'
    hypothesis += f"<p>$H_{0}:$ All the {n} groups have the same median.<br />"
    hypothesis += (
        "$H_{1}:$ There exists one group whose median is different from the others.</p>"
    )

    presentation = decision_kruskal(
        n=n, p_value=p_value, alpha=alpha, decimal_points=decimal_points
    )

    result = " ".join([hypothesis, results_title, table_results, presentation])

    return result


def correlation_strength(rho):
    """
    Strength of correlation coefficient.
    """
    if 0.01 <= abs(rho) < 0.30:
        rho_strength = "very weak"
    elif 0.30 <= abs(rho) < 0.50:
        rho_strength = "weak"
    elif 0.50 <= abs(rho) < 0.70:
        rho_strength = "moderate"
    elif 0.70 <= abs(rho) < 0.90:
        rho_strength = "strong"
    elif 0.90 <= abs(rho) < 1:
        rho_strength = "very strong"
    elif abs(rho) == 1:
        rho_strength = "perfect"
    else:
        rho_strength = "no"

    return rho_strength


def correlation_inference(
    method, 
    decision_method, 
    rho, 
    n, 
    test_value, 
    p_value, 
    alpha, 
    decimal_points=4
):
    """
    Correlation inference
    """
    table_value = t.ppf(q=1 - alpha, df=n - 1)
    decision = '<p style="font-weight:bold;margin-top:15px;"><span class="border_bottom_span">Decision</span></p>'
    conclusion = '<p style="font-weight:bold;"><span class="border_bottom_span">Conclusion</span></p>'

    rho_strength = correlation_strength(rho)

    if rho == 0:
        rho_direction = "neutral"
    elif rho < 0:
        rho_direction = "negative"
    else:
        rho_direction = "positive"

    alpha = round(alpha, 3)
    p_value = round(p_value, 3)
    test_value = round(test_value, decimal_points)
    table_value = round(table_value, decimal_points)

    # decision
    # --------
    if decision_method == "t-statistic":
        if test_value >= table_value:
            decision += f"<p>Reject the null hypothesis since the calculated value of $t({n - 1}) = {test_value}$ is greater than the critical value of $t_{ {alpha} }({n - 1}) = {table_value}$.</p>"
        else:
            decision += f"<p>Fail to reject the null hypothesis since the calculated value of $t({n - 1}) = {test_value}$ is less than the critical value of $t_{ {alpha} }({n - 1}) = {table_value}$.</p>"

    if decision_method == "p-value":
        if p_value <= alpha:
            decision += f"<p>Reject the null hypothesis since the p-value $({p_value})$ is less than the level of significance $({alpha})$.</p>"
        else:
            decision += f"<p>Fail to reject the null hypothesis since the p-value $({p_value})$ is greater than the level of significance $({alpha})$.</p>"

    if decision_method == "graph":
        if test_value >= table_value:
            decision += f"<p>Reject the null hypothesis since the calculated value of $t$ ({test_value}) lies in the rejection region as shown in the diagram below.</p>"
        else:
            decision += f"<p>Fail to reject the null hypothesis since the calculated value of $t$ ({test_value}) lies in the acceptance region as shown in the diagram below.</p>"

    # conclusion
    # ----------
    if round(p_value, 3) == 0:
        pvalue_present = "p < .001"
    else:
        pvalue_present = f"p = {p_value}"

    method = method.capitalize().replace("-", " ")
    if p_value <= alpha:
        conclusion += f"<p>The {method} procedure was performed to assess the relationship between the variables $x$ and $y$. The results found a <strong>{rho_direction}</strong> and <strong>{rho_strength}</strong> relationship that was statistically significant at the <strong>{int(alpha * 100)}%</strong> level of significance, $t({n - 1}) = {round(test_value, 2)}, \\, {pvalue_present}$.</p>"
    else:
        conclusion += f"<p>The {method} procedure was performed out to assess the relationship between the variables $x$ and $y$. The results found a <strong>{rho_direction}</strong> and <strong>{rho_strength}</strong> relationship. This was relationship was however not statistically significant at the <strong>{int(alpha * 100)}%</strong> level of significance, $t({n - 1}) = {round(test_value, 2)}, \\, {pvalue_present}$.</p>"

    result = f"<p>{decision}</p><p>{conclusion}</p>"

    return result
