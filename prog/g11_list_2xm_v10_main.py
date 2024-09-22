
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import rankdata
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import chi2_contingency, chi2, kendalltau
from util_jrnl_trac import pgrm_fini, pgrm_init, plug_inpu, plug_oupu
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from scipy.stats import linregress, norm
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

from g11_list_xxx_v10_inpu import inpu_met2, main_21_agbi_sexe, main_22_agbi_mbre, main_23_ceap_sexe, main_24_ceap_mbre
from g11_list_xxx_v10_plot import dist_2xm_plot_agbi, dist_2xm_plot_ceap

# Test : ttest_ind
# Test : ttest_rel
# Test : Wilcoxon
# Test : Mann-Whitney U
# Test : Kolmogorov-Smirnov
# Test : Permutation
# Test : Kruskal-Wallis H
# Test : Bland-Altmann
# Test : Spearman
# Test : Pearson
# Test : Chi2
# Test : Cramer V
# Test : Mutual Info
# Test : Ztest
# Test : Zconf

# ----
# Util
# ----
# ----
# intp_pval
# ----
def intp_pval(pval):
    # pval
    if pval < 0.001:
        sign_pval = "3/3" # "Extremely significant"
    elif pval < 0.01:
        sign_pval = "2/3" # "Highly significant"
    elif pval < 0.05:
        sign_pval = "1/3" #c"Significant"
    else:
        sign_pval = "0/3" # "Not significant"
    return sign_pval
# ----
# sign_cohe : Cohen d (confirms 'amplitude' of pval)
# ----
def sign_cohe(datM, datF):

    if False:
        datM = [11, 0, 14, 19, 4, 5, 33]  # Group M
        datF = [6, 3, 8, 47, 8, 7, 17]    # Group F

    is_parametric = False  # Set to True for parametric tests
    if is_parametric:
        listM = datM
        listF = datF
    else:
        listM = rankdata(datM)
        listF = rankdata(datF)

    mean_datM = np.mean(listM)
    mean_datF = np.mean(listF)
    sd_datM = np.std(listM, ddof=1)  # Sample standard deviation
    sd_datF = np.std(listF, ddof=1)  # Sample standard deviation
    # Calculate sample sizes
    n1 = len(datM)
    n2 = len(datF)
    # Calculate pooled standard deviation
    sd_pooled = np.sqrt(((n1 - 1) * sd_datM**2 + (n2 - 1) * sd_datF**2) / (n1 + n2 - 2))
    # Calculate Cohen's d
    cohen_d = (mean_datM - mean_datF) / sd_pooled

    # Display the results
    # print(f"Mean of Group M: {mean_datM:.3f}")
    # print(f"Mean of Group F: {mean_datF:.3f}")
    # print(f"Cohen's d: {cohen_d:.3f}")
    # print(f"Cohen's d effect size: {cohen_d:.2f}")
    #
    if abs(cohen_d) < 0.2:
        sign_cohe = '0/3' # print("The effect size is negligible.")
    elif abs(cohen_d) < 0.5:
        sign_cohe = '1/3' # print("The effect size is small.")
    elif abs(cohen_d) < 0.8:
        sign_cohe = '2/3' # print("The effect size is medium.")
    else:
        sign_cohe = '3/3' # print("The effect size is large.")
    return sign_cohe
# ----
# Chck
# ---- 
def chck_same_leng(gro1, gro2):
    # Check if the lengths of the groups are the same
    return len(gro1) == len(gro2) # "The lengths of the two groups must be the same.")
# Function to check normality
def chck_norm(data, grop_name):
    _, p_value = stats.shapiro(data)
    print(f"Shapiro-Wilk test for {grop_name}: p-value = {p_value:.4f}")
    is_normal = p_value > 0.05
    if is_normal:
        info = '✓' # print(f"The {grop_name} data appears to be normally distributed (p > 0.05).")
    else:
        info = '¬' # print(f"The {grop_name} data does not appear to be normally distributed (p <= 0.05).")
    return is_normal, info
# Function to check homogeneity of variances
def chck_vari_homo(gro1, gro2):
    _, p_value = stats.levene(gro1, gro2)
    print(f"Levene's test for homogeneity of variances: p-value = {p_value:.4f}")
    equal_var = p_value > 0.05
    if equal_var:
        info = '✓' # print("The variances appear to be homogeneous (p > 0.05).")
    else:
        info = '¬' # print("The variances do not appear to be homogeneous (p <= 0.05).")
    return equal_var, info
def norm_chck_yes(obsM, obsF): # print("Checking preconditions for t-test test:")
    norM, norM_info = chck_norm(obsM, "obsM")
    norF, norF_info = chck_norm(obsF, "obsF")
    vari, vari_info = chck_vari_homo(obsM, obsF)
    isOK = norM and norF and vari

    chck_list = [i+1 for i, value in enumerate([norM, norF, vari]) if not value]
    if not chck_list:
        chck_info = '✓' 
    else:
        chck_info = f'¬({",".join(map(str, chck_list))})'
    return isOK, chck_info

def norm_chck_not(obsM, obsF): # print("Checking preconditions for Mann-Whitney U test:")
    norM, norM_info = chck_norm(obsM, "obsM")
    norF, norF_info = chck_norm(obsF, "obsF")
    vari, vari_info = chck_vari_homo(obsM, obsF)
    isOK = not(norM and norF and vari)
    
    chck_list = [i for i, value in enumerate([norM, norF, vari]) if not value]
    if not chck_list:
        chck_info = '✓'   
    else:
        chck_info = f'¬({",".join(map(str, chck_list))})'
    # Force
    isOK = True
    return isOK, chck_info
# ----
# ttin
# ---- 
def ttin_chck(obsM, obsF):
    return norm_chck_yes(obsM, obsF)
def ttin_intp(pval):
    ttin_sign = '?'
    # intp
    if pval < 0.05:
        print("Reject the null hypothesis: There is a significant difference between the means of the two groups.")
    else:
        print("Fail to reject the null hypothesis: There is not enough evidence to conclude a significant difference between the means of the two groups.")
    return ttin_sign
def ttin_main(obsM, obsF):
    # prec
    isOK, chck_info = ttin_chck(obsM, obsF)
    if isOK:
        # exec
        stat, pval = stats.ttest_ind(obsM, obsF)
        # intp
        sign_stat = ttin_intp(stat)
        sign_pval = intp_pval(pval)
        # cohe
        sign_cohn = sign_cohe(obsM, obsF)
        # exit
        return stat.round(3), pval.round(3), sign_stat, sign_pval, sign_cohn, chck_info
    else:
        return None, None, None, None, None, chck_info
# ----
# ttre
# ---- 
def ttre_chck(obsM, obsF):
    return norm_chck_yes(obsM, obsF)
def ttre_intp(pval):
    ttre_sign ='?'
    # intp
    if pval < 0.05:
        print("Reject the null hypothesis: There is a significant difference between the means of the two groups.")
    else:
        print("Fail to reject the null hypothesis: There is not enough evidence to conclude a significant difference between the means of the two groups.")
    return ttre_sign
def ttre_main(obsM, obsF):
    # prec
    isOK, chck_info = ttre_chck(obsM, obsF)
    if isOK:
        # exec
        stat, pval = stats.ttest_rel(obsM, obsF)
        # intp
        sign_stat = ttre_intp(stat)
        sign_pval = intp_pval(pval)
        # cohe
        sign_cohn = sign_cohe(obsM, obsF)
        # exit
        return stat.round(3), pval.round(3), sign_stat, sign_pval, sign_cohn, chck_info
    else:
        return None, None, None, None, None, chck_info
# ----
# Wilcoxon test
# ----
def wilc_chck(obsM, obsF):
    is_same_leng = chck_same_leng(obsM, obsF)
    is_norm_chck_not = norm_chck_not(obsM, obsF)
    return is_same_leng and is_norm_chck_not
def wilc_intp(pval): 
    wilc_sign ='?'   
    if pval < 0.05:
        print("Reject the null hypothesis: There is a significant difference between the distributions of the two groups.")
    else:
        print("Fail to reject the null hypothesis: There is not enough evidence to conclude a significant difference between the distributions of the two groups.")
    return wilc_sign
def wilc_main(obsM, obsF):
    # prec
    isOK, chck_info = wilc_chck(obsM, obsF)
    if isOK:
        # exec
        stat, pval = stats.wilcoxon(obsM, obsF)
        # intp
        sign_stat = wilc_intp(stat)
        sign_pval = intp_pval(pval)
        # cohe
        sign_cohn = sign_cohe(obsM, obsF)
        # exit
        return stat.round(3), pval.round(3), sign_stat, sign_pval, sign_cohn, chck_info
    else:
        return None, None, None, None, None, chck_info
# ----
# Mann-Whitney U test
# ----
def mann_chck(obsM, obsF):
    return norm_chck_not(obsM, obsF)
def mann_intp(pval):  
    mann_sign ='?'
    # intp
    if pval < 0.05:
        print("Reject the null hypothesis: There is a significant difference between the distributions of the two groups.")
    else:
        print("Fail to reject the null hypothesis: There is not enough evidence to conclude a significant difference between the distributions of the two groups.")
    return mann_sign 
def mann_main(obsM, obsF):
    # prec
    isOK, chck_info = mann_chck(obsM, obsF)
    if isOK:
        # exec
        stat, pval = stats.mannwhitneyu(obsM, obsF, alternative='two-sided')
        # intp
        sign_stat = mann_intp(stat)
        sign_pval = intp_pval(pval)
        # cohe
        sign_cohn = sign_cohe(obsM, obsF)
        # exit
        return stat.round(3), pval.round(3), sign_stat, sign_pval, sign_cohn, chck_info
    else:
        return None, None, None, None, None, chck_info
# ----
# Kolmogorov-Smirnov test
# ----
def kolm_chck(obsM, obsF):
    return norm_chck_not(obsM, obsF)
def kolm_intp(pval):
    kolm_sign ='?'  
    if pval < 0.05:
        print("Reject the null hypothesis: There is a significant difference between the distributions of the two groups.")
    else:
        print("Fail to reject the null hypothesis: There is not enough evidence to conclude a significant difference between the distributions of the two groups.")
    return kolm_sign
def kolm_main(obsM, obsF):
    # prec
    isOK, chck_info = kolm_chck(obsM, obsF)
    if isOK:
        # exec
        stat, pval = stats.ks_2samp(obsM, obsF)
        # intp
        sign_stat = kolm_intp(stat)
        sign_pval = intp_pval(pval)
        # cohe
        sign_cohn = sign_cohe(obsM, obsF)
        # exit
        return stat.round(3), pval.round(3), sign_stat, sign_pval, sign_cohn, chck_info
    else:
        return None, None, None, None, None, chck_info
# ----
# Permutation test
# ----
def perm_chck(obsM, obsF):
    return norm_chck_not(obsM, obsF)
def perm_exec(obsM, obsF):        
    def diff_in_means(dat1, dat2):
        return np.mean(dat1) - np.mean(dat2)
    # stat
    stat = diff_in_means(obsM, obsF)
    # pval
    combined = obsM + obsF
    nmbr_perm = 10000
    perm_diff_list = []
    for _ in range(nmbr_perm):
        np.random.shuffle(combined)
        perm_M = combined[:len(obsM)]
        perm_F = combined[len(obsM):]
        perm_diff_list.append(diff_in_means(perm_M, perm_F))
    pval = np.sum(np.abs(perm_diff_list) >= np.abs(stat)) / nmbr_perm
    # exit
    return stat.round(3), pval.round(3)
def perm_intp(pval):  
    perm_sign ='?'  
    if pval < 0.05:
        print("Reject the null hypothesis: There is a significant difference between the distributions means of the two groups.")
    else:
        print("Fail to reject the null hypothesis: There is not enough evidence to conclude a significant difference between the distributions means of the two groups.")
    return perm_sign
def perm_main(obsM, obsF):
    # prec
    isOK, chck_info = perm_chck(obsM, obsF)
    if isOK:
        # exec
        stat, pval = perm_exec(obsM, obsF)
        # intp
        sign_stat = perm_intp(stat)
        sign_pval = intp_pval(pval)
        # cohe
        sign_cohn = sign_cohe(obsM, obsF)
        # exit
        return stat.round(3), pval.round(3), sign_stat, sign_pval, sign_cohn, chck_info
    else:
        return None, None, None, None, None, chck_info
# ----
# Kruskal-Wallis H test
# ----
def krus_chck(obsM, obsF):
    return norm_chck_not(obsM, obsF)
def krus_intp(pval): 
    krus_sign ='?'   
    if pval < 0.05:
        print("Reject the null hypothesis: There is a significant difference between the distributions of the two groups.")
    else:
        print("Fail to reject the null hypothesis: There is not enough evidence to conclude a significant difference between the distributions of the two groups.")
    return krus_sign
def krus_main(obsM, obsF):
    # prec
    isOK, chck_info = krus_chck(obsM, obsF)
    if isOK:
        # exec
        stat, pval = stats.kruskal(obsM, obsF)
        # intp
        sign_stat = krus_intp(stat)
        sign_pval = intp_pval(pval)
        # cohe
        sign_cohn = sign_cohe(obsM, obsF)
        # exit
        return stat.round(3), pval.round(3), sign_stat, sign_pval, sign_cohn, chck_info
    else:
        return None, None, None, None, None, chck_info
# ----
# Bland-Altamnn test
# ----
def blan_chck(obsM, obsF):
    
    # Data
    # ----
    prec_list = []
    
    # Check for sufficient sample size
    n = len(obsM)
    if n < 30:  # Minimum sample size check
        prec_list.append('1') # "Sample size must be at least 30 for reliable Bland-Altman analysis.")
    
    # Calculate the differences
    diff_list = [l - r for l, r in zip(obsM, obsF)]
    
    # Check for outliers using the IQR method
    q1 = np.percentile(diff_list, 25)
    q3 = np.percentile(diff_list, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [d for d in diff_list if d < lower_bound or d > upper_bound]
    if outliers:
        prec_list.append('2') # print(f"Warning: Outliers detected in the differences: {outliers}")
  
    # Check for normality of differences using the Shapiro-Wilk test
    shapiro_test = stats.shapiro(diff_list)
    if shapiro_test.pvalue < 0.05:
        prec_list.append('3') # print("Warning: Differences are not normally distributed (p-value < 0.05).")

    # Check for homoscedasticity
    # Create a scatter plot of differences vs means
    means = [(l + r) / 2 for l, r in zip(obsM, obsF)]
    _, p_homoscedasticity = stats.levene(obsM, obsF)
    if p_homoscedasticity < 0.05:
        prec_list.append('4') # print("Warning: Variability of differences is not constant (p-value < 0.05).")
 
    # Exit
    if not prec_list:
        isok = True
        info = '✓'
    else:
        isok = False
        info = f"¬{','.join(prec_list)}"
    # Force
    isok = True
    return isok, info

def blan_intp(stat):
    def blan_mean_intp(mean):
        # stat
        if abs(mean) < 1:
            mean_intp = "M≈F"
        elif abs(mean) < 2:
            mean_intp = "M≠F"
        else:
            mean_intp = "M≠≠F"
        # exit
        return mean_intp
    def blan_stnd_intp(stnd):
        # stat
        if abs(stnd) < 1:
            stnd_intp = "M≈F"
        elif abs(stnd) < 2:
            stnd_intp = "M≠F"
        else:
            stnd_intp = "M≠≠F"
        # exit
        return stnd_intp
    def blan_conf_intp(conf):
        # stat
        ci_lower = conf[0]
        ci_upper = conf[1]
        if ci_lower < 0 and ci_upper < 0:
            conf_intp = "M<F"
        elif ci_lower > 0 and ci_upper > 0:
            conf_intp = "M>F"
        else:
            conf_intp = "M≈F"
        # exit
        return conf_intp
    # stat
    mean_intp = blan_mean_intp(stat['mean'])
    stnd_intp = blan_stnd_intp(stat['stnd'])
    conf_intp = blan_conf_intp(stat['conf'])
    stat_intp = f'{mean_intp}±{stnd_intp} [{conf_intp}]'
    return stat_intp
def blan_exec(obsM, obsF):

    # Calculate the differences
    diff_list = [l - r for l, r in zip(obsM, obsF)]
    
    # Calculate mean and standard deviation
    mean = np.mean(diff_list)
    stnd = np.std(diff_list, ddof=1)  # Sample standard deviation

    # Print the results
    print(f"Différence moyenne : {mean:.2f}")
    print(f"Écart-type de la différence : {stnd:.2f}")
    
    # Calculate the confidence interval for the mean difference
    conf_levl = 0.95
    n = len(diff_list)
    se = stnd / np.sqrt(n)  # Standard error
    alpha = 1 - conf_levl
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)  # t critical value for two-tailed test
    ci_lower = mean - t_critical * se
    ci_upper = mean + t_critical * se
    conf = [ci_lower, ci_upper]
    
    # Calculate p-value for one-sample t-test
    pval = stats.ttest_1samp(diff_list, 0)[1]  # One-tailed p-value
    
    # Exit
    resu_strg = f'{mean.round(3)}±{stnd.round(3)} [{conf[0].round(3)},{conf[1].round(3)}]'
    resu_dict = {
        'mean': mean,
        'stnd': stnd,
        'conf': conf
    }
    return resu_strg, resu_dict, pval

def blan_main(obsM, obsF):
    # prec
    isOK, chck_info = blan_chck(obsM, obsF)
    if isOK:
        
        # exec
        stat_strg, stat_dict, pval = blan_exec(obsM, obsF)
        # intp
        sign_stat = blan_intp(stat_dict) 
        sign_pval = intp_pval(pval)
        
        # cohe
        sign_cohn = sign_cohe(obsM, obsF)

        # exit
        return stat_strg, pval.round(3), sign_stat, sign_pval, sign_cohn, chck_info
    else:
        return None, None, None, None, None, chck_info
'''
PREC
Yes, there are some preconditions that should be considered before applying a statistical test like the Bland-Altman test:
1. **Normality**: The differences between the two measurement methods should be normally distributed. 
You can check this assumption using a normality test (e.g., Shapiro-Wilk test) or by visually inspecting 
a histogram or Q-Q plot of the differences.
2. **Independence**: The differences should be independent of each other. This means that the value of one difference 
should not depend on the value of another difference.
3. **Constant variance**: The variance of the differences should be constant across the range of measurements. 
You can check this assumption by visually inspecting a scatter plot of the differences against the mean of the two methods.
4. **Absence of proportional bias**: There should be no relationship between the differences and the average of the two methods. 
You can check this by visually inspecting a Bland-Altman plot and calculating the correlation coefficient 
between the differences and the averages.
5. **Sufficient sample size**: The sample size should be adequate to provide sufficient power to detect 
clinically relevant differences. The required sample size depends on the expected magnitude of the differences 
and the desired level of statistical significance and power.
If these preconditions are not met, the results of the Bland-Altman analysis may not be valid or may require alternative approaches, 
such as transforming the data or using non-parametric methods.

Citations:
[1] https://docs.devsamurai.com/agiletest/preconditions
[2] https://reqtest.com/en/knowledgebase/preconditions-for-successful-testing/
[3] https://en.training.qatestlab.com/blog/course-materials/how-to-correctly-write-preconditions-in-test-cases/
[4] https://docs.mendix.com/appstore/partner-solutions/ats/ht-two-use-precondition-in-test-cases/
[5] https://docs.getxray.app/display/XRAYCLOUD/Precondition
[6] https://forums.ni.com/t5/NI-TestStand/precondition-for-sequence-and-tests-within/td-p/480595
[7] https://stackoverflow.com/questions/3609204/asserting-set-up-and-pre-conditions-in-unit-tests
'''
'''
PVAL
The Bland-Altman test is used to assess the agreement between two measurement methods or instruments. 
It does not directly compute a p-value. Instead, it provides a graphical representation of the differences 
between the two methods and calculates the mean difference and limits of agreement (LOA).
The LOA is typically defined as the mean difference ± 1.96 × standard deviation of the differences. 
If the differences are normally distributed, approximately 95% of the differences should fall within the LOA.
To assess the statistical significance of the mean difference, you can perform a one-sample t-test. 
The null hypothesis would be that the mean difference is zero (i.e., there is no systematic bias between the two methods). 
The alternative hypothesis would be that the mean difference is not zero.
'''
# ----
# Pearson test
# ----
def pear_chck(obsM, obsF):
    
    # Data
    # ----
    chck_list = []
    full = False
     
    # ---------
    # Linearity
    # ---------
    if False:
        plt.scatter(df_abso_mtrx[M], df_abso_mtrx[F])
        plt.xlabel(M)
        plt.ylabel(F)
        plt.title('Scatter plot of M vs F')
        plt.show()
    if True:
        # Fit a linear regression model and calculate R-squared
        X = obsM
        Y = obsF
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        model = sm.OLS(Y, X).fit()
        r_squared = model.rsquared
        print(f'R-squared value: {r_squared}')
        # Interpretation of R-squared value
        if r_squared > 0.7:
            print('The linearity assumption is met (R-squared > 0.7)')
            if full: chck_list.append('l:y')
        else:
            print('The linearity assumption is not met (R-squared <= 0.7)')
            chck_list.append('l:n' if full else 'l')

    # ---------
    # Normality
    # ---------
    stat_m, p_m = shapiro(obsM)
    stat_f, p_f = shapiro(obsF)
    print(f'Shapiro test for M: Statistic={stat_m}, p-value={p_m}')
    print(f'Shapiro test for F: Statistic={stat_f}, p-value={p_f}')
    alpha = 0.05
    if p_m > alpha:
        print('M is normally distributed (fail to reject H0)')
        if full: chck_list.append('mn:y')
    else:
        print('M is not normally distributed (reject H0)')
        chck_list.append('mn:n' if full else 'mn')
    if p_f > alpha:
        print('F is normally distributed (fail to reject H0)')
        if full: chck_list.append('fn:y')
    else:
        print('F is not normally distributed (reject H0)')
        chck_list.append('fn:n' if full else 'fn')

    # ----------------
    # Homoscedasticity
    # ----------------
    if False:
        plt.scatter(predictions, residuals)
        plt.hlines(0, min(predictions), max(predictions), colors='r', linestyles='dashed')
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.title('Residuals plot')
        plt.show()
    if True:
        X = obsM
        Y = obsF
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        model = sm.OLS(Y, X).fit()
        predictions = model.predict(X)
        residuals = Y - predictions

        # Breusch-Pagan test for homoscedasticity
        bp_test = het_breuschpagan(residuals, X)
        print(f'Breusch-Pagan test: Lagrange Multiplier statistic={bp_test[0]}, p-value={bp_test[1]}')

        # Interpretation of Breusch-Pagan test
        if bp_test[1] > alpha:
            print('The homoscedasticity assumption is met (fail to reject H0)')
            if full: chck_list.append('h:y')
        else:
            print('The homoscedasticity assumption is not met (reject H0)')
            chck_list.append('h:n' if full else 'h')

    if not chck_list:
        isOK = True
        chck_info = '✓'   
    else:
        isOK = False
        chck_info = f'¬({",".join(map(str, chck_list))})'
    # Force
    isOK = True
    return isOK, chck_info
def pear_intp(stat):
    # Stat
    if stat < -0.7:
        stat_intp =  "2/2[-]"
    elif -0.7 <= stat < -0.3:
        stat_intp = "1/2[-]"
    elif -0.3 <= stat < 0.3:
        stat_intp = "0/2"
    elif 0.3 <= stat < 0.7:
        stat_intp =  "1/2[+]"
    else:
        stat_intp = "2/2[+]"
    return stat_intp 
def pear_main(obsM, obsF):
    
    # inpu [CHECK TODO]
    totaM = sum(obsM)
    totaF = sum(obsF)
    propM = [x / totaM for x in obsM] if totaM != 0 else obsM
    propF = [x / totaF for x in obsF] if totaF != 0 else obsF
    if True:
        listM = obsM
        listF = obsF
    else:
        listM = propM
        listF = propF
    print (listM, listF)
    
    # prec
    isOK, chck_info = pear_chck(obsM, obsF)
    if isOK:
        # exec
        stat, pval = stats.pearsonr(obsM, obsF)
        # intp
        sign_stat = pear_intp(stat)
        sign_pval = intp_pval(pval)
        # cohe
        sign_cohn = sign_cohe(obsM, obsF)
        # exit
        return stat.round(3), pval.round(3), sign_stat, sign_pval, sign_cohn, chck_info
    else:
        return None, None, None, None, None, chck_info
# ----
# Spearman test
# ----
def spea_chck(obsM, obsF):
    
    # Data
    # ----
    chck_list = []
    full = False
    
    # Monotonic Relationship
    # Visual inspection [TODO]
    if False:
        plt.scatter(M, F)
        plt.xlabel(M)
        plt.ylabel(F)
        plt.title('Scatter plot of M vs F')
        plt.show()
    if True:
        # Kendall's Tau test for monotonic relationship
        tau, p_value_tau = kendalltau(obsM, obsF)
        print(f'Kendall\'s Tau: {tau}')
        print(f'P-value (Kendall\'s Tau): {p_value_tau}')

        # Interpretation of Kendall's Tau test
        alpha = 0.05
        if p_value_tau > alpha:
            print('There is no significant monotonic relationship between M and F (fail to reject H0)')
            chck_list.append('m:n' if full else 'm')
        else:
            print('There is a significant monotonic relationship between M and F (reject H0)')
            if full: chck_list.append('m:y')

    if not chck_list:
        isOK = True
        chck_info = '✓'   
    else:
        isOK = False
        chck_info = f'¬({",".join(map(str, chck_list))})'
    # Force
    isOK = True
    return isOK, chck_info
def spea_intp(corr):
    # Stat
    if abs(corr) < 0.2:
        stat_intp =  "1/5" # "very weak"
    elif abs(corr) < 0.4:
        stat_intp =  "2/5" # "weak"
    elif abs(corr) < 0.6:
        stat_intp =  "3/5" # "moderate"
    elif abs(corr) < 0.8:
        stat_intp =  "4/5" # "strong"
    else:
        stat_intp =  "5/5" # "very strong"
    return stat_intp 
def spea_main(obsM, obsF):
    
    # inpu [CHECK TODO]
    totaM = sum(obsM)
    totaF = sum(obsF)
    propM = [x / totaM for x in obsM] if totaM != 0 else obsM
    propF = [x / totaF for x in obsF] if totaF != 0 else obsF
    if True:
        listM = obsM
        listF = obsF
    else:
        listM = propM
        listF = propF
    print (listM, listF)
    
    # prec
    isOK, chck_info = spea_chck(obsM, obsF)
    if isOK:
        # exec
        stat, pval = stats.spearmanr(obsM, obsF)
        # intp
        sign_stat = spea_intp(stat)
        sign_pval = intp_pval(pval)
        # cohe
        sign_cohn = sign_cohe(obsM, obsF)
        # exit
        return round(stat,3), round(pval,3), sign_stat, sign_pval, sign_cohn, chck_info
    else:
        return None, None, None, None, None, chck_info
# ----
# Chi2 [full,part] test
# ----
def chi2_chck(df_obsv):
        
    # Vars
    prec_list = []  
    
    # Exec
    df_clean = df_obsv.copy()
    # Check for non-zero observed frequencies
    if True:
        # Check if more than 10% of cells are zero
        total_cells = df_clean.size
        zero_cells = (df_clean == 0).sum().sum()
        if zero_cells / total_cells > 0.1:
            df_clean = df_clean[(df_clean >= 5).all(axis=1)] # Remove rows with any cell having fewer than 5 observations
            prec_list.append('1') # ">10% cells are zero."
    else:
        if (df_clean == 0).any().any():
            return None, False, "There are zero observed frequencies in the contingency table."
    # Check for minimum sample size (at least 5 observations per cell)
    if (df_clean < 5).any().any():
        df_clean = df_clean[(df_clean >= 5).all(axis=1)] # Remove rows with any cell having fewer than 5 observations
        prec_list.append('2') # "Some cells < 5 obsv"
    # Check for balanced data (optional)
    row_sums = df_clean.sum(axis=1)
    col_sums = df_clean.sum(axis=0)
    if (row_sums.min() < 5) or (col_sums.min() < 5):
        df_clean = df_clean[(df_clean >= 5).all(axis=1)] # Remove rows with any cell having fewer than 5 observations
        prec_list.append('3') # "The data is not balanced (some rows or columns have fewer than 5 observations)."
    
    # Exit
    if not prec_list:
        isok = True
        resu = '✓'
    else:
        isok = False
        resu = f"¬{','.join(prec_list)}"
    return isok, resu, df_clean
def chi2_intp(stat, pval, dof): 
    alpha = 0.05
    critical_value = chi2.ppf(1 - alpha, dof)  # Critical value for the given alpha and dof
    if stat > critical_value:
        print("The Chi-squared statistic is greater than the critical value, suggesting a significant association.")
    else:
        print("The Chi-squared statistic is less than the critical value, suggesting no significant association.")
    if False:
        krus_sign = intp_pval(pval)   
        if pval < 0.05:
            print("Reject the null hypothesis: There is a significant difference between the distributions of the two groups.")
        else:
            print("Fail to reject the null hypothesis: There is not enough evidence to conclude a significant difference between the distributions of the two groups.")
    return critical_value
def chi2_main(isOK, chck_info, df):
    obsM = df['obsM'].tolist()
    obsF = df['obsF'].tolist()
    if isOK:
        # exec
        stat, pval, dof, expected = chi2_contingency(df)
        # intp
        sign_stat = chi2_intp(stat, pval, dof)
        sign_pval = intp_pval(pval)
        # cohe
        sign_cohn = sign_cohe(obsM, obsF)
        # exit
        return stat.round(3), pval.round(3), sign_stat, sign_pval, sign_cohn, chck_info
    else:
        return None, None, None, None, None, chck_info
def ch2f_main(obsM, obsF):
    df_obsv = pd.DataFrame({'obsM': obsM,'obsF': obsF})
    isOK, chck_info, df_clea = chi2_chck(df_obsv)
    return chi2_main(isOK, chck_info, df_obsv)
def ch2p_main(obsM, obsF):
    df_obsv = pd.DataFrame({'obsM': obsM,'obsF': obsF})
    isOK, chck_info, df_clea = chi2_chck(df_obsv)
    return chi2_main(isOK, chck_info, df_clea)
# ----
# Cramer V test
# ----
'''
Explanation of the discrepancy:
The presence of many small cell values (e.g., 0, 1, etc.) in the DataFrame can lead to a significant Chi-Square test result but a weak Cramér's V and negligible Mutual Information.")
The Chi-Square test is sensitive to the overall pattern of the data, while Cramér's V and Mutual Information are more affected by the distribution of values.")
Sparse data can dilute the strength of the association and reduce the amount of information shared between the variables.")
'''
def cram_chck_ante(df_obsv):
    print(df_obsv)
    # Check if each variable has at least two unique values
    isok = not(len(df_obsv.index.unique()) < 2 or len(df_obsv.columns.unique()) < 2)
    if isok:
        isok = True
        info = '✓'
    else:
        isok = False
        info = f"¬"
    # Force
    isok = True
    return isok, info
def cram_chck_post(df_expe):
    # Check expected frequencies
    expected_freq = df_expe.flatten()
    num_cells = expected_freq.size
    num_cells_with_min_freq = np.sum(expected_freq >= 5)
    
    isok = True
    # Condition 1: No cell should have an expected frequency less than 1
    if np.any(expected_freq < 1):
        isok = False
    # Condition 2: At least 80% of cells should have expected frequencies of 5 or more
    if num_cells_with_min_freq < 0.8 * num_cells:
        isok = False
        
    if isok:
        info = '✓'
    else:
        info = f"¬"
    # Force
    isok = True
    return isok, info
def cram_intp(stat):
    if stat < 0.1:
        cram_sign = "0/3" # print(f"{what} Cramér's V indicates a negligible association.")
    elif stat < 0.3:
        cram_sign = "1/3" # print(f"{what} Cramér's V indicates a weak association.")
    elif stat < 0.5:
        cram_sign = "2/" # print(f"{what} Cramér's V indicates a moderate association.")
    else:
        cram_sign = "3/3" # print(f"{what} Cramér's V indicates a strong association.")
    return cram_sign
def cram_exec(conf_mtrx):
    print("********")
    print(conf_mtrx)
    print("********")
    stat, pval, dof, expe = chi2_contingency(conf_mtrx)
    # Total number of observations
    n = conf_mtrx.values.sum() if isinstance(conf_mtrx, pd.DataFrame) else conf_mtrx.sum()
    min_dim = min(conf_mtrx.shape) - 1
    stat = np.sqrt(stat / (n * min_dim))
    return stat, pval, expe
def cram_main(df):
    obsM = df['obsM'].tolist()
    obsF = df['obsF'].tolist()
    # prec
    isOK_ante, chck_info_ante = cram_chck_ante(df)
    if isOK_ante:
        # exec
        stat, pval, expe = cram_exec(df)
        # intp
        sign_stat = cram_intp(stat)
        sign_pval = intp_pval(pval) 
        # cohe
        sign_cohn = sign_cohe(obsM, obsF)
        # post
        isOK_post, chck_info_post = cram_chck_post(expe)
        chck_info = f'{chck_info_ante};{chck_info_post}'
        # exit
        return stat.round(3), pval.round(3), sign_stat, sign_pval, sign_cohn, chck_info
    else:
        return None, None, None, None, None, chck_info_ante
def craf_main(obsM, obsF):
            
    # chi2
    df_obsv = pd.DataFrame({'obsM': obsM,'obsF': obsF})
    isOK, chi2_chck_info, df_clea = chi2_chck(df_obsv)   
    if isOK:
        # cram
        cram_stat, cram_pval, cram_sign_stat, cram_sign_pval, cram_sign_cohn, cram_chck_info = cram_main(df_obsv)
        # fuse
        chck_info = f'{chi2_chck_info};{cram_chck_info}'
        # exit
        return cram_stat, cram_pval, cram_sign_stat, cram_sign_pval, cram_sign_cohn, chck_info 
    else:
        return None, None, None, None, None, None
    
def crap_main(obsM, obsF):
    # chi2
    df_obsv = pd.DataFrame({'obsM': obsM,'obsF': obsF})
    isOK, chi2_chck_info, df_clea = chi2_chck(df_obsv)
    if isOK:
        # cram
        print (df_clea)
        cram_stat, cram_pval, cram_sign_stat, cram_sign_pval, cram_sign_cohn, cram_chck_info = cram_main(df_clea)
        # fuse
        if cram_chck_info is not None:
            chck_info = f'{chi2_chck_info};{cram_chck_info}'
        else:
            chck_info = f'{chi2_chck_info}'
        # exit
        return cram_stat, cram_pval, cram_sign_stat, cram_sign_pval, cram_sign_cohn, chck_info 
    else:
        return None, None, None, None, None, None
def exec_mutu(df):
    
    print (f"-------------------")
    print (f"MUTUAL INFO")
    print (f"-------------------")
    # Flatten the DataFrame to create a contingency table
    flattened = df.values.flatten()
    row_labels = np.repeat(df.index, df.shape[1])
    col_labels = np.tile(df.columns, df.shape[0])

    # Check preconditions
    n_samples = len(flattened)
    n_categories = len(np.unique(row_labels)) * len(np.unique(col_labels))
    min_sample_size_per_category = 30  # Typical threshold for reliable estimates

    if n_samples < n_categories * min_sample_size_per_category:
        print("Warning: Sample size is small. Mutual Information estimates may be unreliable.")

    # Calculate Mutual Information
    mi = mutual_info_score(row_labels, col_labels)
    print(f"\nMutual Information: {mi}")

    # Calculate Normalized Mutual Information
    nmi = normalized_mutual_info_score(row_labels, col_labels)
    print(f"Normalized Mutual Information: {nmi}")

    # Interpretation of Mutual Information
    if mi < 0.1:
        print("Mutual Information indicates a negligible association.")
    elif mi < 0.3:
        print("Mutual Information indicates a weak association.")
    elif mi < 0.5:
        print("Mutual Information indicates a moderate association.")
    else:
        print("Mutual Information indicates a strong association.")
# ----
# Ztest test
# ----
'''
Z-score: The z-score indicates how many standard deviations the observed difference in proportions is from the expected difference under the null hypothesis.
p-value: The p-value indicates the probability of observing a test statistic as extreme as, or more extreme than, the one actually observed, 
assuming that the null hypothesis is true.
Reject H0: A boolean value indicating whether the null hypothesis is rejected based on the significance level (α).
If the p-value is less than or equal to the significance level (α), you reject the null hypothesis, 
concluding that there is a statistically significant difference in proportions between males and females within that age bin. 
If the p-value is greater than the significance level, you fail to reject the null hypothesis, 
concluding that there is not enough evidence to support a difference in proportions.
'''
'''
Interpretation of Results
Z-score: Compare the z-scores for individual age bins with the z-score for the total sample. Large differences in z-scores may indicate that 
certain age groups are driving the overall difference.
p-value: Compare the p-values for individual age bins with the p-value for the total sample. Significant p-values in specific age bins 
may indicate that the difference in proportions is more pronounced in those age groups.
Reject H0: Compare the decisions to reject the null hypothesis for individual age bins with the decision for the total sample. 
If the null hypothesis is rejected for the total sample but not for individual age bins, it suggests that the overall difference 
is not uniformly distributed across age groups.
Conclusion
Comparing the z-test outcomes for individual age bins with the z-test outcome for the total sample can provide valuable 
insights into the consistency and heterogeneity of the difference in proportions across different age groups. 
This comparison can help you identify specific age groups that contribute significantly to the overall difference and 
understand the underlying factors that influence the proportions of males and females.
'''
def ztes_chck(totM, totF, tota):
    if not(totM == 0 or totF == 0 or tota == 0):
        isok = True
        resu = '✓'
    else:
        isok = False
        resu = '¬'
    return isok, resu
def ztes_intp(stat): 
    # stat
    if stat < -2:
        stat_intp = "obsv<expe" #"The observed proportion is significantly lower than expected."
    elif -2 <= stat <= 2:
        stat_intp = "obsv≈expe" #"The observed proportion is close to the expected proportion."
    else:
        stat_intp = "obsv>expe" #"The observed proportion is significantly higher than expected."
    return stat_intp

def ztes_main(obsM, obsF):
    # inpu [CHECK TODO]
    totaM = sum(obsM)
    totaF = sum(obsF)
    tota = totaM + totaF
    coun = [totaM, totaF]
    nobs = [tota, tota]   
    # prec
    isOK, chck_info = ztes_chck(totaM, totaF, tota)
    if isOK:
        # exec
        stat, pval = proportions_ztest(coun, nobs)
        # intp
        sign_stat = ztes_intp(stat)
        sign_pval = intp_pval(pval)
        # cohe
        sign_cohn = sign_cohe(obsM, obsF)
        # exit
        return stat.round(3), pval.round(3), sign_stat, sign_pval, sign_cohn, chck_info
    else:
        return None, None, None, None, None, chck_info
# ----
# Zconf test
# ----
def zcon_chck(totM, totF, tota):
    if not(totM == 0 or totF == 0 or tota == 0):
        isok = True
        resu = '✓'
    else:
        isok = False
        resu = '¬'
    return isok, resu
def zcon_exec(totM, totF, tota):
    proM = totM / tota
    proF = totF / tota
    prop_diff = proM - proF
    stnd_erro = np.sqrt(proM * (1 - proM) / totM + proF * (1 - proF) / totF)
    marg_erro = norm.ppf(0.975) * stnd_erro
    conf_ = (prop_diff - marg_erro, prop_diff + marg_erro)
    conf = tuple(round(x, 4) for x in conf_)
    z_score = prop_diff / stnd_erro # this z_score has the same value as the 'stat, pval = proportions_ztest(coun, nobs)'
    pval = 2 * (1 - norm.cdf(np.abs(z_score)))
    return conf, pval # conf, pval, z_score
def zcon_intp(stat): 
    # stat
    if stat[0] <= 0 <= stat[1]:
        stat_intp = "0/1" # "Diff not sign (CIL<0<CIU)"
    else:
        stat_intp = "1/1" # "Diff sign (0<CIU,CIL or CIL,CIU<0)"  
    return stat_intp
def zcon_main(obsM, obsF):
    # inpu [CHECK TODO]
    totaM = sum(obsM)
    totaF = sum(obsF)
    tota = totaM + totaF
    coun = [totaM, totaF]
    nobs = [tota, tota]   
    # prec
    isOK, chck_info = zcon_chck(totaM, totaF, tota)
    if isOK:
        # exec
        stat, pval = zcon_exec(totaM, totaF, tota)
        # intp
        sign_stat = zcon_intp(stat)
        sign_pval = intp_pval(pval)
        # cohe
        sign_cohn = sign_cohe(obsM, obsF)
        # exit
        return tuple(round(num, 3) for num in stat), pval.round(3), sign_stat, sign_pval, sign_cohn, chck_info
    else:
        return None, None, None, None, None, chck_info
 
# ----
# Main
# ----
def main(obsM, obsF):
    resu_dict = {}
    ttin_stat, ttin_pval, ttin_sign_stat, ttin_sign_pval, ttin_sign_cohe, ttin_chck_info = ttin_main(obsM, obsF); resu_dict["ttin"] = [ttin_stat, ttin_pval, ttin_sign_stat, ttin_sign_pval, ttin_sign_cohe, ttin_chck_info]
    ttre_stat, ttre_pval, ttre_sign_stat, ttre_sign_pval, ttre_sign_cohe, ttre_chck_info = ttre_main(obsM, obsF); resu_dict["ttre"] = [ttre_stat, ttre_pval, ttre_sign_stat, ttre_sign_pval, ttre_sign_cohe, ttre_chck_info]
    wilc_stat, wilc_pval, wilc_sign_stat, wilc_sign_pval, wilc_sign_cohe, wilc_chck_info = wilc_main(obsM, obsF); resu_dict["wilc"] = [wilc_stat, wilc_pval, wilc_sign_stat, wilc_sign_pval, wilc_sign_cohe, wilc_chck_info]
    mann_stat, mann_pval, mann_sign_stat, mann_sign_pval, mann_sign_cohe, mann_chck_info = mann_main(obsM, obsF); resu_dict["mann"] = [mann_stat, mann_pval, mann_sign_stat, mann_sign_pval, mann_sign_cohe, mann_chck_info]
    kolm_stat, kolm_pval, kolm_sign_stat, kolm_sign_pval, kolm_sign_cohe, kolm_chck_info = kolm_main(obsM, obsF); resu_dict["kolm"] = [kolm_stat, kolm_pval, kolm_sign_stat, kolm_sign_pval, kolm_sign_cohe, kolm_chck_info]
    perm_stat, perm_pval, perm_sign_stat, perm_sign_pval, perm_sign_cohe, perm_chck_info = perm_main(obsM, obsF); resu_dict["perm"] = [perm_stat, perm_pval, perm_sign_stat, perm_sign_pval, perm_sign_cohe, perm_chck_info]
    krus_stat, krus_pval, krus_sign_stat, krus_sign_pval, krus_sign_cohe, krus_chck_info = krus_main(obsM, obsF); resu_dict["krus"] = [krus_stat, krus_pval, krus_sign_stat, krus_sign_pval, krus_sign_cohe, krus_chck_info]
    blan_stat, blan_pval, blan_sign_stat, blan_sign_pval, blan_sign_cohe, blan_chck_info = blan_main(obsM, obsF); resu_dict["blan"] = [blan_stat, blan_pval, blan_sign_stat, blan_sign_pval, blan_sign_cohe, blan_chck_info]
    pear_stat, pear_pval, pear_sign_stat, pear_sign_pval, pear_sign_cohe, pear_chck_info = pear_main(obsM, obsF); resu_dict["pear"] = [pear_stat, pear_pval, pear_sign_stat, pear_sign_pval, pear_sign_cohe, pear_chck_info]
    spea_stat, spea_pval, spea_sign_stat, spea_sign_pval, spea_sign_cohe, spea_chck_info = spea_main(obsM, obsF); resu_dict["spea"] = [spea_stat, spea_pval, spea_sign_stat, spea_sign_pval, spea_sign_cohe, spea_chck_info]
    ch2f_stat, ch2f_pval, ch2f_sign_stat, ch2f_sign_pval, ch2f_sign_cohe, ch2f_chck_info = ch2f_main(obsM, obsF); resu_dict["ch2f"] = [ch2f_stat, ch2f_pval, ch2f_sign_stat, ch2f_sign_pval, ch2f_sign_cohe, ch2f_chck_info]
    ch2p_stat, ch2p_pval, ch2p_sign_stat, ch2p_sign_pval, ch2p_sign_cohe, ch2p_chck_info = ch2p_main(obsM, obsF); resu_dict["ch2p"] = [ch2p_stat, ch2p_pval, ch2p_sign_stat, ch2p_sign_pval, ch2p_sign_cohe, ch2p_chck_info]
    craf_stat, craf_pval, craf_sign_stat, craf_sign_pval, craf_sign_cohe, craf_chck_info = craf_main(obsM, obsF); resu_dict["craf"] = [craf_stat, craf_pval, craf_sign_stat, craf_sign_pval, craf_sign_cohe, craf_chck_info]
    crap_stat, crap_pval, crap_sign_stat, crap_sign_pval, crap_sign_cohe, crap_chck_info = crap_main(obsM, obsF); resu_dict["crap"] = [crap_stat, crap_pval, crap_sign_stat, crap_sign_pval, crap_sign_cohe, crap_chck_info]
    ztes_stat, ztes_pval, ztes_sign_stat, ztes_sign_pval, ztes_sign_cohe, ztes_chck_info = ztes_main(obsM, obsF); resu_dict["ztes"] = [ztes_stat, ztes_pval, ztes_sign_stat, ztes_sign_pval, ztes_sign_cohe, ztes_chck_info]
    zcon_stat, zcon_pval, zcon_sign_stat, zcon_sign_pval, zcon_sign_cohe, zcon_chck_info = zcon_main(obsM, obsF); resu_dict["zcon"] = [zcon_stat, zcon_pval, zcon_sign_stat, zcon_sign_pval, zcon_sign_cohe, zcon_chck_info]
    return resu_dict

# ----
# Desc
# ----
def desc(df):

    if False:
        data = {
            'G': [0, 5, 3344, 126, 52, 3330, 33],
            'D': [43, 6, 57, 123, 45, 24, 777]
        }
        # Creating the DataFrame
        df = pd.DataFrame(data, index=pd.Index(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], name='ceap'))

    # Generating the describe() output for each column
    df_desc = df.describe()
    # Function to identify outliers
    def outl_list_exec(column):
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outl_list = column[(column < lower_bound) | (column > upper_bound)]
        return outl_list
    # Adding to df_desc
    outliers_row = {}
    for colu_name in df.columns:
        outl_list = outl_list_exec(df[colu_name])
        outl_strg = ",".join([f"{index}:{value}" for index, value in outl_list.items()])
        outliers_row[colu_name] = outl_strg
    df_desc.loc['outliers'] = outliers_row
    
    return df_desc
# ----
# Stat
# ----
def dist_2xm_stat(what, repo, df_inpu):
    # Trac
    # print(f"df_inpu:{type(df_inpu)}\n{df_inpu}\n:{df_inpu.index}")
    plug_inpu(what, repo, df_inpu.T, desc(df_inpu).T)
    # Inpu
    df_stat = df_inpu.copy()
    df_stat.columns = ['obsM', 'obsF']
    obsM = df_stat['obsM'].tolist()
    obsF = df_stat['obsF'].tolist()
    # Exec
    oupu_dict = main(obsM, obsF)
    # Oupu
    column_names = ['stat', 'pval', 'sign_stat', 'sign_pval', 'sign_cohe', 'chck_warn']
    df_oupu = pd.DataFrame(oupu_dict, index=column_names).T 
    print(f"df_oupu:{type(df_oupu)}\n{df_oupu}\n:{df_oupu.index}")
    # Trac
    plug_oupu(what, repo, df_oupu)
    pass

# Example usage
if __name__ == "__main__":
    
    repo = pgrm_init()
    #
    df1 = inpu_met2()
    #
    what, df_inpu, plot_axis = main_21_agbi_sexe(df1)
    dist_2xm_stat(what, repo, df_inpu)
    dist_2xm_plot_agbi(what, df_inpu, plot_axis)

    what, df_inpu, plot_axis = main_22_agbi_mbre(df1)
    dist_2xm_stat(what, repo, df_inpu)
    dist_2xm_plot_agbi(what, df_inpu, plot_axis)
    
    what, df_inpu, plot_axis = main_23_ceap_sexe(df1)
    dist_2xm_stat(what, repo, df_inpu)
    dist_2xm_plot_ceap(what, df_inpu, plot_axis)
    
    what, df_inpu, plot_axis = main_24_ceap_mbre(df1)
    dist_2xm_stat(what, repo, df_inpu)
    dist_2xm_plot_ceap(what, df_inpu, plot_axis)

    #
    pgrm_fini(repo, os.path.splitext(os.path.basename(__file__)))
    
    pass
'''   
stat, pval = stats.ttint_ind(obsM, obsF)
stat, pval = stats.mannwhitneyu(obsM, obsF, alternative='two-sided')
stat, pval = stats.ks_2samp(obsM, obsF)
stat, pval = stats.wilcoxon(obsM, obsF)
stat = permutation_test(obsM, obsF)
stat, pval = stats.kruskal(*groups)
stat, pval = stats.friedmanchisquare(*groups)
--- 
All these tests are used to compare samples or groups to determine if there are 
statistically significant differences between them.
Each test evaluates a null hypothesis that generally assumes there is 
no difference between the groups or samples.
---

| Name                          | Compares                                 | Parametric Data (Y/N)  |Must Same Group Length(N) | Minimum Number of Groups(2)
|-------------------------------|------------------------------------------|------------------------|
| Independent t-test            | Means of two independent samples         | Y                      |
| Mann-Whitney U test           | Distributions of two independent samples | N                      |
| Two-sample Kolmogorov-Smirnov test | Distributions of two independent samples | N                 |
| Wilcoxon test                 | Distributions of two paired samples      | N                      | Y
| Permutation test              | Distributions of two samples             | N                      |
| Kruskal-Wallis test           | Distributions of three or more independent samples | N            |
| Friedman test                 | Distributions of three or more paired samples | N                 | Y                         3
---
When "Parametric Data" is marked as "N" in the table, 
it means that the statistical test is suitable for non-parametric data. 
Non-parametric data refers to data that does not meet the assumptions 
required for parametric tests, such as normality and homogeneity of variances. 
Non-parametric tests are often used when the data is 
ordinal, ranked, or does not follow a normal distribution.
'''