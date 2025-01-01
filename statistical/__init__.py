from .dataframes import dm_dframe_lower
from .contingencytables import (
    ctab_fisher_exact, ctab_chi2_goodness_of_fit,
    ctab_chi2_homogeneity, ctab_chi2_independence, ctab_chi2_trend,
    ctab_chi2_variance, ctab_cochran_mantel_haenszel, ctab_cochrans_q,
    ctab_cramers_v, ctab_g_test, ctab_kendall_tau, ctab_kruskall_tau,
    ctab_likelihood_ratio, ctab_log_linear, ctab_mantel_haenszel, ctab_mcnemar,
    ctab_yates_correction
)
from .core import hyp_corr_tpvalue
from .descriptive import (
    eda_pooled_variance, eda_standard_error, df_satterthwaite, df_welch,
    df_satterthwaite_stats, df_welch_stats, eda_pooled_variance_stats,
    eda_standard_error_stats, eda_mode_series, eda_mode_freq, dm_unique_cat, eda_freq_tables, 
    eda_tabstat, eda_tabstat_series, eda_descriptive
)

from .sampledatasets import dm_data_random, dm_dataset_random
from .nonparametric import (
    np_chisquare_goodness_of_fit, np_friedman_nindependent,
    np_kendall_correlation, np_kolmogorov_smirnov_goodness_of_fit,
    np_kruskal_wallis_nindependent, np_mann_whitney_u_2independent,
    np_sign_1sample, np_sign_paired, np_spearman_correlation,
    np_wilcoxon_ranksum_2independent, np_wilcoxon_signed_rank_1sample,
    np_wilcoxon_signed_rank_paired
)
from .probability import (
    prob_normal_gt_x, prob_normal_lt_x, prob_normal_between_x1_x2
)
from .ptztests import (
    means_1sample_t, means_1sample_t_stats, 
    means_1sample_z, means_1sample_z_stats,
    means_2independent_t, means_2independent_t_stats,
    means_paired_t,
    means_2independent_z, means_2independent_z_stats,
    means_paired_z,
    prop_1sample, prop_1sample_stats, 
    prop_2independent, prop_2independent_stats,
    var_ratio_test_1sample, var_ratio_test_1sample_stats,
    var_ratio_test_2samples, var_ratio_test_2samples_stats
)
from .wrangle import (
    dm_relocate, dm_insert, dm_dframe_split, dm_outliers,
    dm_outliers_replace, dm_scale, dm_stack_cols, dm_unstack_cols,
    dm_dframe_order_by_list, dm_drop_contains, dm_na_replace
)


def __dir__():
    
    all_attrs = globals().keys()
    dirs = ['contingencytables', 'core', 'dataframes', 'decisions',
            'descriptive', 'hypothesistests', 'inferences', 'nonparametric',
            'ptztests', 'sampledatasets', 'wrangle']
    visible_attrs = [attr for attr in all_attrs if attr not in dirs]

    return visible_attrs

