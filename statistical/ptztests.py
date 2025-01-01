from typing import Literal

from pandas import DataFrame, Series, concat
from numpy import nan, unique, round, asfarray, sqrt
from scipy.stats import (
    t, ttest_1samp, ttest_ind, ttest_rel, ttest_ind_from_stats, norm
)

from stemlab.core.arraylike import conv_to_arraylike
from stemlab.core.validators.errors import RequiredError
from stemlab.core.htmlatex import ptz_table
from stemlab.statistical.dataframes import series_name
from stemlab.statistical.decisions import test_decision, test_conclusion
from stemlab.statistical.core import tz_alternative
from stemlab.core.display import Result, display_latex
from stemlab.statistical.descriptive import (
    df_satterthwaite_stats, df_welch_stats, eda_standard_error_stats
)
from stemlab.core.validators.validate import ValidateArgs
from stemlab.core.datatypes import ListArrayLike


def _sample_names(sample_names, is_one_sample: bool):
    """
    Prepare sample names
    """
    if sample_names:
        sample_names = conv_to_arraylike(
            array_values=sample_names,
            n=2,
            label='at most',
            includes_str=True,
            par_name='sample_name' if is_one_sample else 'sample_names'
        )
        sample1_name = sample_names[0]
        sample2_name = sample_names[1] if len(sample_names) == 2 else 'Sample 2'
        
        # if the names are the same, then append codes 1 and 2
        if sample1_name == sample2_name:
            sample1_name, sample2_name = f'{sample1_name}1', f'{sample2_name}2'
    else:
        sample1_name, sample2_name = '', ''
    
    sample1_name = 'Sample 1' if not sample1_name else sample1_name
    sample2_name = 'Sample 2' if not sample2_name else sample2_name
            
    return sample1_name, sample2_name


VALID_METHODS = ['one-sample', 'two-samples', 'paired-samples']
LiteralMethods = Literal['one-sample', 'two-samples', 'paired-samples']
LiteralAlternative = Literal['less', 'two-sided', 'greater']


class MeansTZTest:
    """Perform t or z test"""
    def __init__(
            self,
            method: LiteralMethods,
            is_ztest: bool,
            sample1: ListArrayLike,
            sample2_or_group: ListArrayLike | None = None,
            pop_mean: int | float | None = None,
            pop_std_diff: int | float | None = None,
            pop_std_within: int | float | None = None,
            rho: float | None = None,
            pop1_std: int | float = 1,
            pop2_std: int | float | None = None,
            alternative: LiteralAlternative = 'two-sided',
            equal_var: bool | None = True,
            welch: bool | None = False,
            conf_level: float = 0.95,
            sample_names: list[str] | None = None,
            decimal_points: int = 4
        ):
        
        self.method = method.lower()
        self.is_ztest = is_ztest
        self.sample1 = sample1
        self.sample2_or_group = sample2_or_group
        self.pop_mean = pop_mean
        self.std_diff = pop_std_diff
        self.std_within = pop_std_within
        self.rho = rho
        self.std1 = pop1_std
        self.std2 = pop2_std
        self.alternative = alternative.lower()
        self.equal_var = equal_var
        self.welch = welch
        self.conf_level = conf_level
        self.sample_names = sample_names
        self.decimal_points = decimal_points

        # method
        if self.method in ['two-sample', 'paired-sample']:
            self.method = f'{self.method}s'
        
        self.method = ValidateArgs.check_member(
            par_name='method', 
            valid_items=VALID_METHODS, 
            user_input=self.method
        )
        
        self.is_ztest = ValidateArgs.check_boolean(user_input=self.is_ztest, default=False)
        
        # sample1
        self.sample1_name = series_name(data=self.sample1, n=1)
        self.sample1 = conv_to_arraylike(
            array_values=self.sample1, to_ndarray=True, par_name='sample1'
        )

        # sample2_or_group
        if self.sample2_or_group is None: # one-sample t/z test
            self.method = 'one-sample'
            self.sample2, self.sample2_name = None, None
            self.by_group = False
        else:
            if 'one' not in self.method: # two and paired samples t/z test
                self.sample2_name = series_name(data=self.sample2_or_group, n=2)
                self.sample2 = conv_to_arraylike(
                    array_values=self.sample2_or_group, 
                    to_ndarray=False, # do not use True to avoid crush if by_group 
                    par_name='sample2_or_group'
                )
                # if there are only two categories, then it must be 
                # `grouped by` so force by_group=True 
                # (this avoids crushing, and going back to user)
                sample1_count = len(unique(self.sample1))
                sample2_count = len(unique(self.sample2))
                self.by_group = (
                    True if sample2_count == 2 and sample1_count != 2 else False
                )
                
                # paired
                if self.method == 'paired-samples':
                    if sample2_count == 2 and sample1_count != 2:
                        raise ValueError(
                            'Paired samples test does not allow for '
                            'a grouping variable, did you intend to '
                            'perform a two-samples independent test?'
                        )
                    self.sample1 = asfarray(self.sample1).flatten()
                    self.sample2 = asfarray(self.sample2).flatten()
                    ValidateArgs.check_len_equal(
                        x=self.sample1, 
                        y=self.sample2,
                        par_name=['sample1', 'sample2_or_group']
                    )
                    # paired does not allow for grouping variable so 
                    # force `self.by_group` to be False so that the 
                    # branch below is not evaluated
                    self.by_group = False

                if self.by_group: # this is two-independent t/z test
                    ValidateArgs.check_len_equal(
                        x=self.sample1, 
                        y=self.sample2, 
                        par_name=['sample1', 'sample2_or_group']
                    )
                    # check that there are exactly 2 categories in sample2
                    self.unique_groups = conv_to_arraylike(
                        array_values=unique(self.sample2), 
                        n=2, 
                        label='exactly', 
                        par_name='sample2_or_group'
                    )
                    # get the sample names from the categories
                    self.sample1_name, self.sample2_name = self.unique_groups
                    # convert to DataFrame then extract the samples
                    dframe = DataFrame(data=[self.sample1, self.sample2]).T
                    dframe.columns = ['values', 'group']
                    # filter sample1 and sample2 from the DataFrame
                    sample1 = dframe[dframe['group'] == self.unique_groups[0]]
                    sample2 = dframe[dframe['group'] == self.unique_groups[1]]
                    self.sample1 = asfarray(sample1.iloc[:, 0].values).flatten()
                    self.sample2 = asfarray(sample2.iloc[:, 0].values).flatten()
                # check `identical` only if `sample1` and `sample2` have the 
                # same length
                if len(self.sample1) == len(self.sample2):
                    ValidateArgs.check_identical(
                        x=self.sample1, 
                        y=self.sample2, 
                        x_par_name='Sample 1', 
                        y_par_name='sample 2'
                    )
                self.sample2 = ValidateArgs.check_constant(
                    user_input=self.sample2, par_name='Sample 2'
                )
                
        self.is_one_sample = 'one' in self.method
        
        if self.is_one_sample: # one-sample t/z test
            self.by_group = False
            self.sample2, self.sample2_name = (None, None)
            if self.pop_mean is None:
                raise RequiredError(
                    par_name='pop_mean', required_when="method='one-sample'"
                )
            else:
                self.pop_mean = ValidateArgs.check_numeric(
                    user_input=self.pop_mean,
                    to_float=False,
                    par_name='pop_mean'
                )
        
        if self.is_ztest:
            if self.method == 'two-samples':
                # both `std1` and `std2` cannot be `None` 
                # if `method='two-samples'`
                if self.std1 is None and self.std2 is None:
                    raise ValueError(
                        f"Expected a valid numeric value for 'pop1_std' or "
                        f"'pop2_std' to be provided. Required when "
                        f"'method=two-samples'"
                    )
                
                # either of `std1` and `std2` can be given
                if self.std1 is None and self.std2 is not None:
                    self.std1 = self.std2
                
                if self.std1 is not None and self.std2 is None:
                    self.std2 = self.std1
                
                # now validate - should be here, not in the above 
                # `if` statements
                self.std1 = ValidateArgs.check_numeric(
                    user_input=self.std1, to_float=False, par_name='pop1_std'
                )
                self.std2 = ValidateArgs.check_numeric(
                    user_input=self.std2, to_float=False, par_name='pop2_std'
                )
            elif self.method == 'paired-samples':
                if self.rho is None:
                    if self.std_diff is None and self.std_within is None:
                        raise ValueError(
                            f"Expected a valid numeric value for "
                            f"'pop_std_diff' or 'pop_std_within' to be "
                            f"provided. Required when 'rho=None'"
                        )
                    if self.std_diff is not None:
                        self.std_diff = ValidateArgs.check_numeric(
                            user_input=self.std_diff, to_float=False, par_name='pop_std_diff'
                        )
                        
                    if self.std_within is not None:
                        self.std_within = ValidateArgs.check_numeric(
                            user_input=self.std_within, to_float=False, par_name='pop_std_within'
                        )
                else:
                    self.rho = ValidateArgs.check_numeric(
                        user_input=self.rho,
                        limits=[0, 1],
                        to_float=False,
                        par_name='rho'
                    )
                    
                    # both `std1` and `std2` cannot be `None` if 
                    # `method='two-samples'` and `rho` is not `None`
                    if self.std1 is None and self.std2 is None:
                        raise ValueError(
                            f"Expected a valid numeric value for 'pop1_std' or "
                            f"'pop2_std' to be provided. Required when "
                            f"'method=two-samples' and `rho` is not `None`"
                        )
                        
                    # either of `std1` and `std2` can be given
                    if self.std1 is None and self.std2 is not None:
                        self.std1 = self.std2
                    
                    if self.std1 is not None and self.std2 is None:
                        self.std2 = self.std1
                    
                    # now validate - should be here not in the above 
                    # `if` statements
                    self.std1 = ValidateArgs.check_numeric(
                        user_input=self.std1, to_float=False, par_name='pop1_std'
                    )
                    self.std2 = ValidateArgs.check_numeric(
                        user_input=self.std2, to_float=False, par_name='pop2_std'
                    )
                
        self.sample1 = ValidateArgs.check_constant(
            user_input=self.sample1, par_name='Sample 1'
        ) # must be here
        
        self.equal_var = False if self.is_one_sample else equal_var
        
        self.tz = 'z' if self.is_ztest else 't'
        self.alternative = ValidateArgs.check_alternative(user_input=self.alternative)
        
        if self.method == 'two-samples':
            self.equal_var = ValidateArgs.check_boolean(
                user_input=self.equal_var, default=True
            )
            self.welch = ValidateArgs.check_boolean(user_input=self.welch, default=False)
        else:
            self.equal_var, self.welch = (False, False)
            
        self.sample1_name, self.sample2_name = _sample_names(
            sample_names=self.sample_names, is_one_sample=self.is_one_sample
        )
            
        # conf_level
        self.conf_level = ValidateArgs.check_conf_level(user_input = self.conf_level)
        
        # decimal_points
        decimal_points = ValidateArgs.check_decimals(x=decimal_points)

        self.test_name = f'{self.method}-{self.tz}'
        self.table_title = f'{self.method.capitalize()} {self.tz} test'
                
        
        # end of validation of inputs

        self._statistics() # must be here, to ensure updated attributes

        self.sig_level = 1 - self.conf_level
        # hypothesis and t/z critical
        self.test_type = f'{self.tz}'
        self.hyp_sign, self.crit_value = tz_alternative(
            test_type=self.test_type,
            alternative=self.alternative,
            dfn=self.dfn,
            sig_level=self.sig_level, 
            decimal_points=self.decimal_points
        )

        self.decision = test_decision(
            p_value=self.p_value,
            sig_level=self.sig_level
        )
        
        self.conclusion = test_conclusion(
            test_name=self.test_name,
            sample1_name=self.sample1_name,
            sample2_name=self.sample2_name,
            mean1=self.mean1,
            mean2=self.mean2,
            pop_mean=self.pop_mean,
            std1=Series(self.sample1).std(), # do not use std1 because of z-test
            std2=Series(self.sample2).std(), # same as above
            test_stat=self.test_stat,
            dfn=self.dfn,
            alternative=self.alternative,
            p_value=self.p_value,
            sig_level=self.sig_level
        )
        
        self.table_title = self.table_title.capitalize()\
        .replace('-', ' ')\
        .replace(' with', ' t test with')
        pop_mean = 0 if pop_mean is None else pop_mean
        self.result_table = ptz_table(
            dct={
                'table_title': self.table_title,
                'rho': self.rho,
                'sample1_name': self.sample1_name,
                'sample2_name': self.sample2_name,
                'n1': self.n1,
                'n2': self.n2,
                'n12': self.n12,
                'n12_diff': self.n12_diff,
                'mean1': self.mean1,
                'mean2': self.mean2,
                'mean12': self.mean12,
                'mean12_diff': self.mean12_diff,
                'std1': self.std1,
                'std2': self.std2,
                'std12': self.std12,
                'std12_diff': self.std12_diff,
                'sem1': self.sem1,
                'sem2': self.sem2,
                'sem12': self.sem12,
                'sem12_diff': self.sem12_diff,
                'LCI1': self.LCI1,
                'UCI1': self.UCI1,
                'LCI2': self.LCI2,
                'UCI2': self.UCI2,
                'LCI12': self.LCI12,
                'UCI12': self.UCI12,
                'LCI12_diff': self.LCI12_diff,
                'UCI12_diff': self.UCI12_diff,
                'test_stat': self.test_stat,
                'pop_mean': pop_mean,
                'mean_diff': self.mean1 - pop_mean,
                'dfn': self.dfn,
                'dfn_name': self.dfn_name,
                'hyp_sign': self.hyp_sign,
                'crit_value': self.crit_value,
                'p_value': self.p_value,
                'conf_level': self.conf_level,
                'sig_level': self.sig_level,
                'is_one_sample': self.is_one_sample,
                'decision': self.decision,
                'conclusion': self.conclusion
            },
            decimal_points=self.decimal_points
        )
    

    def _statistics(self):
        """
        Upates the required statistics
        """
        if self.is_one_sample:
            # placeholders (for one sample t/z test)
            self.dfn_name = 'Degrees of freedom'
            (self.n2, self.n12, self.n12_diff, self.mean2, self.mean12,
                self.std2, self.std12, self.sem2, self.sem12) = [0] * 9
            
            self.mean12_diff, self.std12_diff, self.sem12_diff = [0] * 3
            
            (self.LCI2, self.UCI2, self.LCI12, self.UCI12, self.LCI12_diff,
                self.UCI12_diff ) = [0] * 6
        
        # the order of the following functions is important
        self._samples()
        self._counts()
        self._means()
        self._std_deviations()
        self._std_errors()
        self._degrees_of_freedom()
        self._confidence_intervals()
        self._hypothesis()
        self._tzpvalue()
        
        
    def _samples(self):
        """
        Create the samples.
        """
        self.sample1 = Series(self.sample1)
        if not self.is_one_sample:
            self.sample2 = Series(self.sample2)
            self.sample12 = concat([self.sample1, self.sample2])
            self.sample_diff = self.sample1 - self.sample2
        
    
    def _counts(self):
        """
        Get the lengths or arrays.
        """
        self.n1 = len(self.sample1)
        if not self.is_one_sample:
            self.n2 = len(self.sample2)
            self.n12 = self.n1 + self.n2
            self.n12_diff = self.n1
            
    
    def _means(self):
        """
        Calculate sample means
        """
        self.mean1 = self.sample1.mean()
        if self.is_one_sample:
            self.mean12_diff = self.mean1 - self.pop_mean
        else:
            self.mean2 = self.sample2.mean()
            self.mean12 = self.sample12.mean()
            if self.method == 'paired-samples':
                self.mean12_diff = self.sample_diff.mean()
            else:
                self.mean12_diff = self.sample1.mean() - self.sample2.mean()
    
                    
    def _std_deviations(self):
        """
        Calculate mean differences
        """
        if self.is_ztest:
            if self.is_one_sample:
                self.std12_diff = ''
            else:
                self.std12 = ''
                if self.method == 'paired-samples':
                    self.std12_diff = self._std_pop_z()
                else:
                    self.std12_diff = nan
        else:
            self.std1 = self.sample1.std()
            if self.is_one_sample:
                self.std12_diff = ''
            else:
                self.std2 = self.sample2.std()
                self.std12 = self.sample12.std()
                if self.method == 'paired-samples':
                    self.std12_diff = self.sample_diff.std()
                else:
                    self.std12_diff = nan


    def _std_errors(self):
        """
        Calculate mean differences
        """
        if self.is_ztest:
            if self.method == 'paired-samples':
                if self.rho is None:
                    self.sem1 = self._std_pop_z() / sqrt(self.n1)
                else:
                    self.sem1 = self.std1 / sqrt(self.n1)
            else:
                self.sem1 = self.std1 / sqrt(self.n1)
                
            if self.is_one_sample:
                self.sem12_diff = ''
            else:
                if self.method == 'paired-samples':
                    if self.rho is not None:
                        self.sem2 = self.std2 / sqrt(self.n2)
                    else:
                        self.sem2 = ''
                    self.sem12 = nan
                    self.sem12_diff = self._std_pop_z() / sqrt(self.n1)
                else: # two-samples
                    self.sem2 = self.std2 / sqrt(self.n2)
                    self.sem12 = self.sample12.sem()
                    self.sem12_diff = eda_standard_error_stats(
                        n1=self.n1,
                        n2=self.n2,
                        std1=self.std1,
                        std2=self.std2,
                        is_pooled=False
                    )
        else:
            self.sem1 = self.sample1.sem()
            if self.is_one_sample:
                self.sem12_diff = ''
            else:
                self.sem2 = self.sample2.sem()
                if self.method == 'paired-samples':
                    self.sem12 = self.sample12.sem() # just a placeholder
                    self.sem12_diff = self.sample_diff.sem()
                else: # two-samples
                    self.sem12 = self.sample12.sem()
                    self.sem12_diff = eda_standard_error_stats(
                        n1=self.n1,
                        n2=self.n2,
                        std1=self.std1,
                        std2=self.std2,
                        is_pooled=self.equal_var
                    )


    def _degrees_of_freedom(self):
        """
        This function updates the following:
            - dfn_name : Degrees of freedom `label`.
            - dfn : Calculated degrees of freedom.
        """
        if self.is_ztest:
            self.dfn_name, self.dfn = ('', None)
        else: # t test
            if self.method == 'one-sample' or self.method == 'paired-samples':
                self.dfn_name = 'Degrees of freedom'
                self.dfn = self.n1 - 1
            else: # two-samples
                if self.equal_var:
                    self.dfn_name = 'Degrees of freedom'
                    self.dfn = self.n12 - 2
                else:
                    if self.welch: # Welch's degrees of freedom
                        self.dfn_name = "Welch's DF"
                        self.dfn = df_welch_stats(
                            n1=self.n1,
                            std1=self.std1,
                            n2=self.n2,
                            std2=self.std2
                        )
                    else: # Satterthwaite's degrees of freedom
                        self.dfn_name = "Satterthwaite's DF"
                        self.dfn = df_satterthwaite_stats(
                            n1=self.n1,
                            std1=self.std1,
                            n2=self.n2,
                            std2=self.std2
                        )
    
    
    def _confidence_intervals(self):
        """
        Calculate confidence intervals
        """
        if self.is_ztest:
            self.LCI1, self. UCI1 = norm.interval(
                confidence=self.conf_level,
                loc=self.mean1,
                scale=self.sem1
            )
            
            if not self.is_one_sample:
                if self.method == 'two-samples':
                    self.LCI2, self.UCI2 = norm.interval(
                        confidence=self.conf_level,
                        loc=self.mean2,
                        scale=self.sem2
                    )
                    
                    self.LCI12, self.UCI12 = norm.interval(
                        confidence=self.conf_level,
                        loc=self.mean12,
                        scale=self.sem12
                    )
                    
                    self.LCI12_diff, self.UCI12_diff = norm.interval(
                        confidence=self.conf_level,
                        loc=self.mean12_diff,
                        scale=self.sem12_diff
                    )
                else: # paired samples
                    if self.rho is None:
                        self.LCI2, self.UCI2, self.LCI12, self.UCI12 = [None] * 4
                    else:
                        self.LCI2, self.UCI2 = norm.interval(
                            confidence=self.conf_level,
                            loc=self.mean2,
                            scale=self.sem2
                        )
                        
                        self.LCI12, self.UCI12 = norm.interval(
                            confidence=self.conf_level,
                            loc=self.mean12,
                            scale=self.sem12
                        )
                    self.LCI12_diff, self.UCI12_diff = norm.interval(
                        confidence=self.conf_level,
                        loc=self.mean12_diff,
                        scale=self.sem12_diff
                    )
        else:
            self.LCI1, self. UCI1 = t.interval(
                confidence=self.conf_level,
                loc=self.mean1,
                df=self.n1 - 1,
                scale=self.sem1
            )
            
            if self.method in ['two-samples', 'paired-samples']:
                self.LCI2, self.UCI2 = t.interval(
                    confidence=self.conf_level,
                    loc=self.mean2,
                    df=self.n2 - 1,
                    scale=self.sem2
                )
                
                self.LCI12, self.UCI12 = t.interval(
                    confidence=self.conf_level,
                    loc=self.mean12,
                    df=self.n1 + self.n2 - 1,
                    scale=self.sem12
                )
                
                self.LCI12_diff,self.UCI12_diff = t.interval(
                    confidence=self.conf_level,
                    loc=self.mean12_diff,
                    df=self.dfn,
                    scale=self.sem12_diff
                )
                
    
    def _std_pop_z(self):
        """
        Calculate degrees of freedom for paired-z test.
        
        References
        ----------
        https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/PASS/Paired_Z-Tests.pdf
        """
        if self.std_diff:
            self.std_pop = self.std_diff
        elif self.std_within:
            self.std_pop = sqrt(2 * self.std_within ** 2)
        else: # self.rho:
            if self.std1 == self.std2:
                self.std_pop = sqrt(2 * self.std1 ** 2 * (1 - self.rho))
            else:
                self.std_pop = sqrt(
                    self.std1 ** 2 + self.std2 ** 2 - 2 * self.rho * self.std1 * self.std2
                )
        return self.std_pop
                
                
    def _hypothesis(self):
        """
        Perform a hypothesis test based on the specified method and 
        samples.
        
        Parameters
        ----------
        None
        
        Attributes
        ----------
        self.test_name : str
            The name of the test performed.
        self.table_title : str
            The title describing the test.
        self.test_stat : float
            The calculated test statistic.
        self.p_value : float
            The p-value of the test.
        
        Notes
        -----
        This method performs different types of hypothesis tests including 
        one-sample t/z-test, two-sample t/z-test, and paired t/z-test. 
        The type of test is determined by the `method`attribute of the 
        class instance.
        """
        if self.is_ztest:
            # do manual calculations, norm.ztest is giving results that are 
            # different from Stata
            if self.method == 'one-sample':
                self.test_stat = (self.mean1 - self.pop_mean) / self.sem1
            elif self.method == 'two-samples':
                self.test_stat = (self.mean1 - self.mean2) / self.sem12_diff
            elif self.method == 'paired-samples':
                self.std_pop = self._std_pop_z()
                self.test_stat = self.mean12_diff / (self.std_pop / sqrt(self.n1))
        else:
            if self.is_one_sample:
                self.test_stat, self.p_value = ttest_1samp(
                    a=self.sample1,
                    popmean=self.pop_mean,
                    alternative=self.alternative
                )
            elif self.method == 'two-samples':
                self.test_stat, self.p_value = ttest_ind(
                    a=self.sample1, 
                    b=self.sample2, 
                    equal_var=self.equal_var, 
                    alternative=self.alternative
                )
                if self.equal_var:
                    self.table_title = f'{self.method} with equal variances'
                else:
                    # calculate p-value for Welch, Scipy doesn't give results
                    # similar to Stata (what it gives is same as Satterthwaite's DF)
                    self.p_value = self._tzpvalue()
                    self.table_title = f'{self.method} with unequal variances'
            elif self.method == 'paired-samples':
                self.test_stat, self.p_value = ttest_rel(
                    a=self.sample1, 
                    b=self.sample2, 
                    alternative=self.alternative
                )    
            

    def _tzpvalue(self):
        """
        Calculate the p-value for t or z test.

        Parameters
        ----------
        None

        Returns
        -------
        self.p_value : float
            The calculated p-value.
        """
        if self.is_ztest:
            if self.alternative == 'less':
                self.p_value = norm.cdf(x=self.test_stat)
            elif self.alternative == 'greater':
                self.p_value = 1 - norm.cdf(x=self.test_stat)
            else: # alternative == 'two sided':
                self.p_value = 2 * norm.cdf(x=-abs(self.test_stat))
        else:
            if self.alternative == 'less':
                self.p_value = t.cdf(x=self.test_stat, df=self.dfn)
            elif self.alternative == 'greater':
                self.p_value = 1 - t.cdf(x=self.test_stat, df=self.dfn)
            else: # alternative == 'two sided':
                self.p_value = 2 * t.cdf(x=-abs(self.test_stat), df=self.dfn)
            
            
    def compute(self):
        """
        Perform t/z test.
        """
        display_latex(html_list=self.result_table)
        if self.is_one_sample:
            result = Result(
                test_name=f'{self.test_name} test',
                alternative=self.alternative,
                mean=self.mean1,
                pop_mean=self.pop_mean,
                mean_diff=self.mean1 - self.pop_mean,
                std=self.std1,
                sem=self.sem1,
                LCI=self.LCI1,
                UCI=self.UCI1,
                test_stat=self.test_stat,
                df=self.dfn,
                crit_value=self.crit_value,
                sig_level=round(self.sig_level, 4),
                conf_level=round(self.conf_level, 4),
                p_value=self.p_value
            )
        else:
            result = Result(
                test_name=f'{self.test_name} test',
                alternative=self.alternative,
                mean1=self.mean1,
                mean2=self.mean2,
                pop_mean=self.pop_mean,
                mean_comb=self.mean12,
                mean_diff=self.mean12_diff,
                std1=self.std1,
                std2=self.std2,
                sem1=self.sem1,
                sem2=self.sem2,
                LCI={'LCI1': self.LCI1, 'LCI2': self.LCI2, 'LCI12': self.LCI12},
                UCI={'UCI1': self.UCI1, 'UCI2': self.UCI2, 'UCI12': self.UCI12},
                test_stat=self.test_stat,
                df=self.dfn,
                crit_value=self.crit_value,
                sig_level=round(self.sig_level, 4),
                conf_level=round(self.conf_level, 4),
                p_value=self.p_value
            )
            
        return result


def means_1sample_t(
    sample: ListArrayLike,
    pop_mean: int | float,
    alternative: LiteralAlternative = 'two-sided',
    conf_level: float = 0.95,
    sample_name: str | None = None,
    decimal_points: int = 4
) -> Result:
    """
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')
    >>> result = sta.means_1sample_t(sample=df['score_before'],
    ... pop_mean=30.47, alternative='less', decimal_points=4)
    
    Same as above but using `alternative='greater'`.
    
    >>> df = stm.dataset_read(name='scores')
    >>> result = sta.means_1sample_t(sample=df['score_before'],
    ... pop_mean=30.47, alternative='greater', decimal_points=4)
    
    Another example using `potato_yield` data.
    
    >>> potato_yield = [21.5, 24.5, 18.5, 17.2, 14.5, 23.2, 22.1, 20.5,
    ... 19.4, 18.1, 24.1, 18.5]
    >>> potato_yield = pd.Series(potato_yield, name="potato_yield")
    >>> result = sta.means_1sample_t(sample=potato_yield,
    ... pop_mean=17.38, alternative='two-sided', decimal_points=4)
    """
    solver = MeansTZTest(
        method='one-sample',
        is_ztest=False,
        sample1=sample,
        sample2_or_group=None,
        pop_mean=pop_mean,
        pop_std_diff=None,
        pop_std_within=None,
        rho=None,
        pop1_std=None,
        pop2_std=None,
        alternative=alternative,
        equal_var=None,
        welch=None,
        conf_level=conf_level,
        sample_names=sample_name,
        decimal_points=decimal_points
    )
    
    return solver.compute()


def means_2independent_t(
    sample1: ListArrayLike,
    sample2_or_group: ListArrayLike,
    alternative: LiteralAlternative = 'two-sided',
    equal_var=True,
    welch=False,
    conf_level: float = 0.95,
    sample_names: list[str] | None = None,
    decimal_points: int = 4
) -> Result:
    """
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')
    
    Two samples using groups

    >>> result = sta.means_2independent_t(sample1=df['score_after'],
    ... sample2_or_group=df['gender'], alternative='less',
    ... decimal_points=4)
    
    Prepare data for two samples using variables
    
    >>> female = df.loc[df['gender'] == 'Female', 'score_after']
    >>> male = df.loc[df['gender'] == 'Male', 'score_after']
    
    Two independent samples t test using variables: Equal variances
    
    >>> result = sta.means_2independent_t(sample1=female,
    ... sample2_or_group=male, alternative='less', equal_var=True,
    ... decimal_points=4)
    
    Two independent samples t test using variables: Unequal variances, Satterthwaite's degrees of freedom
    
    >>> result = sta.means_2independent_t(sample1=female,
    ... sample2_or_group=male, alternative='less', equal_var=False,
    ... welch=False, decimal_points=4)
    
    Two independent samples t test using variables: Unequal variances, Welch's degrees of freedom
    
    >>> result = sta.means_2independent_t(sample1=female,
    ... sample2_or_group=male, alternative='less', equal_var=False,
    ... welch=True, decimal_points=4)
    """
    solver = MeansTZTest(
        method='two-samples',
        is_ztest=False,
        sample1=sample1,
        sample2_or_group=sample2_or_group,
        pop_mean=None,
        pop_std_diff=None,
        pop_std_within=None,
        rho=None,
        pop1_std=None,
        pop2_std=None,
        alternative=alternative,
        equal_var=equal_var,
        welch=welch,
        conf_level=conf_level,
        sample_names=sample_names,
        decimal_points=decimal_points
    )
    
    return solver.compute()


def means_paired_t(
    sample1: ListArrayLike,
    sample2: ListArrayLike,
    alternative: LiteralAlternative = 'two-sided',
    conf_level: float = 0.95,
    sample_names: list[str] | None = None,
    decimal_points: int = 4
) -> Result:
    """
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')
    >>> result = sta.means_paired_t(sample1=df['score_after'],
    ... sample2=df['score_before'], alternative='less',
    ... decimal_points=4)
    
    Another example
    
    >>> x = [18, 21, 16, 22, 19, 24, 17, 21, 23, 18,
         14, 16, 16, 19, 18, 20, 12, 22, 15, 17]
         
    >>> y = [22, 25, 17, 24, 16, 29, 20, 23, 19, 20,
         15, 15, 18, 26, 18, 24, 18, 25, 19, 16]

    >>> result = sta.means_paired_t(sample1=x, sample2=y,
    ... alternative='less', sample_names=['x', 'y'], decimal_points=4)

    """
    solver = MeansTZTest(
        method='paired-samples',
        is_ztest=False,
        sample1=sample1,
        sample2_or_group=sample2,
        pop_mean=None,
        pop_std_diff=None,
        pop_std_within=None,
        rho=None,
        pop1_std=None,
        pop2_std=None,
        alternative=alternative,
        equal_var=None,
        welch=None,
        conf_level=conf_level,
        sample_names=sample_names,
        decimal_points=decimal_points
    )
    
    return solver.compute()


class MeansTZTestStats:
    """Perform t or z test given statistics"""
    def __init__(
            self,
            method: Literal['one-sample', 'two-samples'],
            is_ztest: bool,
            sample1_size: int,
            sample2_size: int | None = None,
            pop1_mean: int | float | None = None,
            pop2_mean: int | float | None = None,
            pop1_std: int | float | None = None,
            pop2_std: int | float | None = None,
            alternative: LiteralAlternative = 'two-sided',
            equal_var: bool | None = True,
            welch: bool | None = False,
            conf_level: float = 0.95,
            sample_names: list[str] | None = None,
            decimal_points: int = 4
        ):
        
        self.method = method.lower()
        self.is_ztest = is_ztest
        self.sample1_size = sample1_size
        self.sample2_size = sample2_size
        self.pop1_mean = pop1_mean
        self.pop2_mean = pop2_mean
        self.std1 = pop1_std
        self.std2 = pop2_std
        self.alternative = alternative.lower()
        self.equal_var = equal_var
        self.welch = welch
        self.conf_level = conf_level
        self.sample_names = sample_names
        self.decimal_points = decimal_points

        # method
        if self.method == 'two-sample':
            self.method = f'{self.method}s'
        
        self.method = ValidateArgs.check_member(
            par_name='method', 
            valid_items=['one-sample', 'two-samples'], 
            user_input=self.method
        )
        
        self.is_ztest = ValidateArgs.check_boolean(user_input=self.is_ztest, default=False)
        
        self.sample1_size = ValidateArgs.check_numeric(
            par_name='sample1_size',
            is_positive=True,
            is_integer=True,
            user_input=sample1_size 
        )

        # sample2_size
        if self.sample2_size is None: # one-sample t/z test
            self.method = 'one-sample'
            self.sample2, self.sample2_name = None, None
        else:
            if 'one' not in self.method: # two and paired samples t/z test
                self.sample2_size = ValidateArgs.check_numeric(
                    par_name='sample2_size',
                    is_positive=True,
                    is_integer=True,
                    user_input=sample2_size 
                )
                
        self.is_one_sample = 'one' in self.method
        
        if self.is_one_sample: # one-sample t/z test
            self.sample2_size, self.sample2_name = (None, None)
            if self.pop1_mean is None:
                raise RequiredError(
                    par_name='pop1_mean', required_when="method='one-sample'"
                )
            else:
                self.pop_mean = ValidateArgs.check_numeric(
                    user_input=self.pop1_mean,
                    to_float=False,
                    par_name='pop1_mean'
                )
            
        if self.is_ztest:
            # both `std1` and `std2` cannot be `None` 
            # if `method='two-samples'`
            if self.std1 is None and self.std2 is None:
                raise ValueError(
                    f"Expected a valid numeric value for 'pop1_std' or "
                    f"'pop2_std' to be provided. Required when "
                    f"'method=two-samples'"
                )
            
            # either of `std1` and `std2` can be given
            if self.std1 is None and self.std2 is not None:
                self.std1 = self.std2
            
            if self.std1 is not None and self.std2 is None:
                self.std2 = self.std1
            
            # now validate - should be here, not in the above 
            # `if` statements
            self.std1 = ValidateArgs.check_numeric(
                user_input=self.std1, to_float=False, par_name='pop1_std'
            )
            self.std2 = ValidateArgs.check_numeric(
                user_input=self.std2, to_float=False, par_name='pop2_std'
            )
        
        self.equal_var = False if self.is_one_sample else equal_var
        self.tz = 'z' if self.is_ztest else 't'
        self.alternative = ValidateArgs.check_alternative(user_input=self.alternative)
        
        if self.method == 'two-samples':
            self.equal_var = ValidateArgs.check_boolean(
                user_input=self.equal_var, default=True
            )
            self.welch = ValidateArgs.check_boolean(user_input=self.welch, default=False)
        else:
            self.equal_var, self.welch = (False, False)
            
        self.sample1_name, self.sample2_name = _sample_names(
            sample_names=self.sample_names, is_one_sample=self.is_one_sample
        )
            
        # conf_level
        self.conf_level = ValidateArgs.check_conf_level(user_input = self.conf_level)
        
        # decimal_points
        decimal_points = ValidateArgs.check_decimals(x=decimal_points)

        self.test_name = f'{self.method}-{self.tz}'
        self.table_title = f'{self.method.capitalize()} {self.tz} test'
                
        
        # end of validation of inputs

        self._statistics() # must be here, to ensure updated attributes

        self.sig_level = 1 - self.conf_level
        # hypothesis and t/z critical
        self.test_type = f'{self.tz}'
        self.hyp_sign, self.crit_value = tz_alternative(
            test_type=self.test_type,
            alternative=self.alternative,
            dfn=self.dfn,
            sig_level=self.sig_level, 
            decimal_points=self.decimal_points
        )

        self.decision = test_decision(
            p_value=self.p_value,
            sig_level=self.sig_level
        )
        
        self.conclusion = test_conclusion(
            test_name=self.test_name,
            sample1_name=self.sample1_name,
            sample2_name=self.sample2_name,
            mean1=self.pop1_mean,
            mean2=self.pop2_mean,
            std1=pop1_std,
            std2=pop2_std,
            test_stat=self.test_stat,
            dfn=self.dfn,
            alternative=self.alternative,
            p_value=self.p_value,
            sig_level=self.sig_level
        )
        
        self.table_title = self.table_title.capitalize()\
        .replace('-', ' ')\
        .replace(' with', ' t test with')
        self.result_table = ptz_table(
            dct={
                'table_title': self.table_title,
                'rho': '0',
                'sample1_name': self.sample1_name,
                'sample2_name': self.sample2_name,
                'n1': self.n1,
                'n2': self.n2,
                'n12': self.n12,
                'n12_diff': self.n12_diff,
                'mean1': self.mean1,
                'mean2': self.mean2,
                'mean12': self.mean12,
                'mean12_diff': self.mean12_diff,
                'std1': self.std1,
                'std2': self.std2,
                'std12': self.std12,
                'std12_diff': self.std12_diff,
                'sem1': self.sem1,
                'sem2': self.sem2,
                'sem12': self.sem12,
                'sem12_diff': self.sem12_diff,
                'LCI1': self.LCI1,
                'UCI1': self.UCI1,
                'LCI2': self.LCI2,
                'UCI2': self.UCI2,
                'LCI12': self.LCI12,
                'UCI12': self.UCI12,
                'LCI12_diff': self.LCI12_diff,
                'UCI12_diff': self.UCI12_diff,
                'test_stat': self.test_stat,
                'pop_mean': 0,
                'mean_diff': self.mean1 - 0,
                'dfn': self.dfn,
                'dfn_name': self.dfn_name,
                'hyp_sign': self.hyp_sign,
                'crit_value': self.crit_value,
                'p_value': self.p_value,
                'conf_level': self.conf_level,
                'sig_level': self.sig_level,
                'is_one_sample': self.is_one_sample,
                'decision': self.decision,
                'conclusion': self.conclusion
            },
            decimal_points=self.decimal_points
        )
    

    def _statistics(self):
        """
        Updates the required statistics
        """
        if self.is_one_sample:
            # placeholders (for one sample t/z test)
            self.dfn_name = 'Degrees of freedom'
            (self.n2, self.n12, self.n12_diff, self.mean2, self.mean12,
                self.std2, self.std12, self.sem2, self.sem12) = [0] * 9
            
            self.mean12_diff, self.std12_diff, self.sem12_diff = [0] * 3
            
            (self.LCI2, self.UCI2, self.LCI12, self.UCI12, self.LCI12_diff,
                self.UCI12_diff ) = [0] * 6
        
        # the order of the following functions is important
        self._counts()
        self._means()
        self._std_deviations()
        self._std_errors()
        self._degrees_of_freedom()
        self._confidence_intervals()
        self._hypothesis()
        self._tzpvalue()
        
    
    def _counts(self):
        """
        Get the lengths or arrays.
        """
        self.n1 = self.sample1_name
        if not self.is_one_sample:
            self.n2 = len(self.sample2_name)
            self.n12 = self.n1 + self.n2
            self.n12_diff = self.n1
            
    
    def _means(self):
        """
        Calculate sample means
        """
        self.mean1 = self.pop1_mean
        if self.is_one_sample:
            self.mean12_diff = self.mean1 - self.pop_mean
        else:
            self.mean2 = self.pop2_mean
            self.mean12 = nan
            self.mean12_diff = nan
    
                    
    def _std_deviations(self):
        """
        Calculate mean differences
        """
        if self.is_ztest:
            if self.is_one_sample:
                self.std12_diff = ''
            else:
                self.std12 = ''
                self.std12_diff = nan
        else:
            if self.is_one_sample:
                self.std12_diff = ''
            else:
                self.std12 = self.sample12.std()
                self.std12_diff = nan


    def _std_errors(self):
        """
        Calculate mean differences
        """
        if self.is_ztest:
            self.sem1 = self.std1 / sqrt(self.n1)
            if self.is_one_sample:
                self.sem12_diff = ''
            else:
                self.sem2 = self.std2 / sqrt(self.n2)
                self.sem12 = nan
                self.sem12_diff = eda_standard_error_stats(
                    n1=self.n1,
                    n2=self.n2,
                    std1=self.std1,
                    std2=self.std2,
                    is_pooled=False
                )
        else:
            self.sem1 = self.std1 / sqrt(self.n1)
            if self.is_one_sample:
                self.sem12_diff = ''
            else:
                self.sem2 = self.std2 / sqrt(self.n2)
                self.sem12 = nan
                self.sem12_diff = eda_standard_error_stats(
                    n1=self.n1,
                    n2=self.n2,
                    std1=self.std1,
                    std2=self.std2,
                    is_pooled=self.equal_var
                )
                

    def _degrees_of_freedom(self):
        """
        This function updates the following:
            - dfn_name : Degrees of freedom `label`.
            - dfn : Calculated degrees of freedom.
        """
        if self.is_ztest:
            self.dfn_name, self.dfn = ('', None)
        else: # t test
            if self.method == 'one-sample' or self.method == 'paired-samples':
                self.dfn_name = 'Degrees of freedom'
                self.dfn = self.n1 - 1
            else: # two-samples
                if self.equal_var:
                    self.dfn_name = 'Degrees of freedom'
                    self.dfn = self.n12 - 2
                else:
                    if self.welch: # Welch's degrees of freedom
                        self.dfn_name = "Welch's DF"
                        self.dfn = df_welch_stats(
                            n1=self.n1,
                            std1=self.std1,
                            n2=self.n2,
                            std2=self.std2
                        )
                    else: # Satterthwaite's degrees of freedom
                        self.dfn_name = "Satterthwaite's DF"
                        self.dfn = df_satterthwaite_stats(
                            n1=self.n1,
                            std1=self.std1,
                            n2=self.n2,
                            std2=self.std2
                        )
    
    
    def _confidence_intervals(self):
        """
        Calculate confidence intervals
        """
        if self.is_ztest:
            self.LCI1, self. UCI1 = norm.interval(
                confidence=self.conf_level,
                loc=self.mean1,
                scale=self.sem1
            )
            
            if not self.is_one_sample:
                if self.method == 'two-samples':
                    self.LCI2, self.UCI2 = norm.interval(
                        confidence=self.conf_level,
                        loc=self.mean2,
                        scale=self.sem2
                    )
                    
                    self.LCI12, self.UCI12 = norm.interval(
                        confidence=self.conf_level,
                        loc=self.mean12,
                        scale=self.sem12
                    )
                    
                    self.LCI12_diff, self.UCI12_diff = norm.interval(
                        confidence=self.conf_level,
                        loc=self.mean12_diff,
                        scale=self.sem12_diff
                    )
                else: # paired samples
                    if self.rho is None:
                        self.LCI2, self.UCI2, self.LCI12, self.UCI12 = [None] * 4
                    else:
                        self.LCI2, self.UCI2 = norm.interval(
                            confidence=self.conf_level,
                            loc=self.mean2,
                            scale=self.sem2
                        )
                        
                        self.LCI12, self.UCI12 = norm.interval(
                            confidence=self.conf_level,
                            loc=self.mean12,
                            scale=self.sem12
                        )
                    self.LCI12_diff, self.UCI12_diff = norm.interval(
                        confidence=self.conf_level,
                        loc=self.mean12_diff,
                        scale=self.sem12_diff
                    )
        else:
            self.LCI1, self. UCI1 = t.interval(
                confidence=self.conf_level,
                loc=self.mean1,
                df=self.n1 - 1,
                scale=self.sem1
            )
            
            if self.method in ['two-samples', 'paired-samples']:
                self.LCI2, self.UCI2 = t.interval(
                    confidence=self.conf_level,
                    loc=self.mean2,
                    df=self.n2 - 1,
                    scale=self.sem2
                )
                
                self.LCI12, self.UCI12 = t.interval(
                    confidence=self.conf_level,
                    loc=self.mean12,
                    df=self.n1 + self.n2 - 1,
                    scale=self.sem12
                )
                
                self.LCI12_diff,self.UCI12_diff = t.interval(
                    confidence=self.conf_level,
                    loc=self.mean12_diff,
                    df=self.dfn,
                    scale=self.sem12_diff
                )
               
                
    def _hypothesis(self):
        """
        Perform a hypothesis test based on the specified method and 
        samples.
        
        Parameters
        ----------
        None
        
        Attributes
        ----------
        self.test_name : str
            The name of the test performed.
        self.table_title : str
            The title describing the test.
        self.test_stat : float
            The calculated test statistic.
        self.p_value : float
            The p-value of the test.
        
        Notes
        -----
        This method performs different types of hypothesis tests including 
        one-sample t/z-test, two-sample t/z-test, and paired t/z-test. 
        The type of test is determined by the `method`attribute of the 
        class instance.
        """
        if self.is_ztest:
            # do manual calculations, norm.ztest is giving results that are 
            # different from Stata
            if self.method == 'one-sample':
                self.test_stat = (self.mean1 - self.pop_mean) / self.sem1
            else: # self.method == 'two-samples':
                self.test_stat = (self.mean1 - self.mean2) / self.sem12_diff
        else:
            if self.is_one_sample:
                self.test_stat, self.p_value = ttest_1samp(
                    a=self.sample1,
                    popmean=self.pop_mean,
                    alternative=self.alternative
                )
            else:# self.method == 'two-samples':
                self.test_stat, self.p_value = ttest_ind(
                    a=self.sample1, 
                    b=self.sample2, 
                    equal_var=self.equal_var, 
                    alternative=self.alternative
                )
                if self.equal_var:
                    self.table_title = f'{self.method} with equal variances'
                else:
                    # calculate p-value for Welch, Scipy doesn't give results
                    # similar to Stata (what it gives is same as Satterthwaite's DF)
                    self.p_value = self._tzpvalue()
                    self.table_title = f'{self.method} with unequal variances'
            

    def _tzpvalue(self):
        """
        Calculate the p-value for t or z test.

        Parameters
        ----------
        None

        Returns
        -------
        self.p_value : float
            The calculated p-value.
        """
        if self.is_ztest:
            if self.alternative == 'less':
                self.p_value = norm.cdf(x=self.test_stat)
            elif self.alternative == 'greater':
                self.p_value = 1 - norm.cdf(x=self.test_stat)
            else: # alternative == 'two sided':
                self.p_value = 2 * norm.cdf(x=-abs(self.test_stat))
        else:
            if self.alternative == 'less':
                self.p_value = t.cdf(x=self.test_stat, df=self.dfn)
            elif self.alternative == 'greater':
                self.p_value = 1 - t.cdf(x=self.test_stat, df=self.dfn)
            else: # alternative == 'two sided':
                self.p_value = 2 * t.cdf(x=-abs(self.test_stat), df=self.dfn)
            
            
    def compute(self):
        """
        Perform t/z test.
        """
        display_latex(html_list=self.result_table)
        if self.is_one_sample:
            result = Result(
                test_name=f'{self.test_name} test',
                alternative=self.alternative,
                mean=self.mean1,
                pop_mean=self.pop_mean,
                mean_diff=self.mean1 - self.pop_mean,
                std=self.std1,
                sem=self.sem1,
                LCI=self.LCI1,
                UCI=self.UCI1,
                test_stat=self.test_stat,
                df=self.dfn,
                crit_value=self.crit_value,
                sig_level=round(self.sig_level, 4),
                conf_level=round(self.conf_level, 4),
                p_value=self.p_value
            )
        else:
            result = Result(
                test_name=f'{self.test_name} test',
                alternative=self.alternative,
                mean1=self.mean1,
                mean2=self.mean2,
                pop_mean=self.pop_mean,
                mean_comb=self.mean12,
                mean_diff=self.mean12_diff,
                std1=self.std1,
                std2=self.std2,
                sem1=self.sem1,
                sem2=self.sem2,
                LCI={'LCI1': self.LCI1, 'LCI2': self.LCI2, 'LCI12': self.LCI12},
                UCI={'UCI1': self.UCI1, 'UCI2': self.UCI2, 'UCI12': self.UCI12},
                test_stat=self.test_stat,
                df=self.dfn,
                crit_value=self.crit_value,
                sig_level=round(self.sig_level, 4),
                conf_level=round(self.conf_level, 4),
                p_value=self.p_value
            )
            
        return result


def means_1sample_t_stats():
    pass


def means_2independent_t_stats():
    pass


def means_1sample_z(
    sample: ListArrayLike,
    pop_mean: int | float,
    pop_std: int | float,
    alternative: LiteralAlternative = 'two-sided',
    conf_level: float = 0.95,
    sample_name: str | None = None,
    decimal_points: int = 4
) -> Result:
    """
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')
    >>> result = sta.means_1sample_z(sample=df['score_before'],
    ... pop_mean=30.47, pop_std=8.53, alternative='less',
    ... decimal_points=4)
    
    Same as above but using `alternative='greater'`.
    
    >>> result = sta.means_1sample_z(sample=df['score_before'],
    ... pop_mean=30.47, pop_std=8.53, alternative='greater',
    ... decimal_points=4)
    
    Another example using `potato_yield` data.
    
    >>> potato_yield = [21.5, 24.5, 18.5, 17.2, 14.5, 23.2, 22.1, 20.5,
    ... 19.4, 18.1, 24.1, 18.5]
    >>> potato_yield = pd.Series(potato_yield, name="potato_yield")
    >>> result = sta.means_1sample_z(sample=potato_yield,
    ... pop_mean=17.38, pop_std=3.74, alternative='two-sided',
    ... decimal_points=4)
    """
    solver = MeansTZTest(
        method='one-sample',
        is_ztest=True,
        sample1=sample,
        sample2_or_group=None,
        pop_mean=pop_mean,
        pop_std_diff=None,
        pop_std_within=None,
        rho=None,
        pop1_std=pop_std,
        pop2_std=None,
        alternative=alternative,
        equal_var=None,
        welch=None,
        conf_level=conf_level,
        sample_names=sample_name,
        decimal_points=decimal_points
    )
    
    return solver.compute()
    

def means_2independent_z(
    sample1: ListArrayLike,
    sample2_or_group: ListArrayLike,
    pop1_std: int | float | None = None,
    pop2_std: int | float | None = None,
    alternative: LiteralAlternative = 'two-sided',
    conf_level: float = 0.95,
    sample_names: list | str | None = None,
    decimal_points: int = 4
) -> Result:
    """
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')

    Two samples using groups: Standard deviations of individual populations

    >>> result = sta.means_2independent_z(sample1=df['score_after'],
    ... sample2_or_group=df['gender'], pop1_std=15, pop2_std=12,
    ... alternative='less', decimal_points=4)

    Prepare data for two samples using variables

    >>> female = df.loc[df['gender'] == 'Female', 'score_after']
    >>> male = df.loc[df['gender'] == 'Male', 'score_after']

    Two independent samples z test using variables: Common standard deviation

    >>> result = sta.means_2independent_z(sample1=female,
    ... sample2_or_group=male, pop1_std=14, alternative='less',
    ... decimal_points=4)

    Two independent samples z test using variables: Standard deviations of individual populations

    >>> result = sta.means_2independent_z(sample1=female,
    ... sample2_or_group=male, pop1_std=15, pop2_std=12,
    ... alternative='less', decimal_points=4)
    """
    solver = MeansTZTest(
        method='two-sample',
        is_ztest=True,
        sample1=sample1,
        sample2_or_group=sample2_or_group,
        pop_mean=None,
        pop_std_diff=None,
        pop_std_within=None,
        rho=None,
        pop1_std=pop1_std,
        pop2_std=pop2_std,
        alternative=alternative,
        equal_var=None,
        welch=None,
        conf_level=conf_level,
        sample_names=sample_names,
        decimal_points=decimal_points
    )
    
    return solver.compute()


def means_paired_z(
    sample1: ListArrayLike,
    sample2: ListArrayLike,
    pop_std_diff: int | float | None = None,
    pop_std_within: int | float | None = None,
    rho: float | None = None,
    pop1_std: int | float | None = None,
    pop2_std: int | float | None = None,
    alternative: LiteralAlternative = 'two-sided',
    conf_level: float = 0.95,
    sample_names: list | str | None = None,
    decimal_points: int = 4
) -> Result:
    """
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')
    >>> result = sta.means_paired_z(sample1=df['score_after'],
    ... sample2=df['score_before'], pop_std_diff=14, alternative='less',
    ... decimal_points=4)
    
    Same as above but using `pop_std_within` (same results as above).
    
    >>> df = stm.dataset_read(name='scores')
    >>> result = sta.means_paired_z(sample1=df['score_after'],
    ... sample2=df['score_before'], pop_std_within=14,
    ... alternative='greater', decimal_points=4)
    
    Given correlation (rho), common population standard deviation
    
    >>> df = stm.dataset_read(name='scores')
    >>> result = sta.means_paired_z(sample1=df['score_after'],
    ... sample2=df['score_before'], rho=0.432, pop1_std=12,
    ... alternative='greater', decimal_points=4)
    
    Given correlation (rho), and individual population standard deviations
    
    >>> df = stm.dataset_read(name='scores')
    >>> result = sta.means_paired_z(sample1=df['score_after'],
    ... sample2=df['score_before'], rho=0.432, pop1_std=15, pop_std=12, 
    ... alternative='greater', decimal_points=4)
    """
    solver = MeansTZTest(
        method='paired-sample',
        is_ztest=True,
        sample1=sample1,
        sample2_or_group=sample2,
        pop_mean=None,
        pop_std_diff=pop_std_diff,
        pop_std_within=pop_std_within,
        rho=rho,
        pop1_std=pop1_std,
        pop2_std=pop2_std,
        alternative=alternative,
        equal_var=None,
        welch=None,
        conf_level=conf_level,
        sample_names=sample_names,
        decimal_points=decimal_points
    )
    
    return solver.compute()


def means_1sample_z_stats():
    pass


def means_2independent_z_stats():
    pass


def prop_1sample():
    pass


def prop_2independent():
    pass


def prop_1sample_stats():
    pass


def prop_2independent_stats():
    pass


def var_ratio_test_1sample():
    pass


def var_ratio_test_2samples():
    pass


def var_ratio_test_1sample_stats():
    pass


def var_ratio_test_2samples_stats():
    pass