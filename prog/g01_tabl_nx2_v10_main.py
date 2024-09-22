import ast
import inspect
from math import sqrt
import os
import pandas as pd
import scipy.stats as stats
import numpy as np
from scipy.stats import combine_pvalues, fisher_exact
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2_contingency
import scipy
from scipy.stats import chi2
import seaborn as sns
import matplotlib.pyplot as plt
import sympy as sp
from util_jrnl_trac import pgrm_fini, pgrm_init, plug_inpu, plug_oupu
from util_file_inou import file_inpu
from g01_tabl_xxx_v10_inpu import inpu_func, mtrx_nxn_selc, mtrx_nx2_selc
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from sklearn.metrics import cohen_kappa_score
from scipy.stats import chisquare
from statsmodels.stats.contingency_tables import SquareTable
from itertools import combinations
from scipy.stats import chi2_contingency, spearmanr, kendalltau
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm

# (nx2 tables) Stuart_maxwell test        
# (nx2 tables) Bowker test 
# (nx2 tables) Fisher's Exact test            
# (nx2 tables) McNemar's test      
# (nx2 tables) Cohen's Kappa test  

# ----
# Util
# ----
# ----
# pval
# ----
def pval_intp(pval):
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
def pval_expl(pval):
    return None
def pval_intp_list(pval, ceap_dict):
    sign_pval = pval_intp(pval)
    for ceap, value in ceap_dict.items():
        value["sign_pval_deta"] = pval_intp(value['pval_deta'])
    return sign_pval
def pval_expl_list(pval, ceap_dict):
    sign_pval = pval_expl(pval)
    for ceap, value in ceap_dict.items():
        value["expl_pval_deta"] = pval_expl(value['pval_deta'])
# ----
# Fish : Fisher : global on p_val
# ----
def fish(ceap_dict):
    if ceap_dict is None:
       sign_fish = None
    else:    
        pval_list = [entry['pval_deta'] for entry in ceap_dict.values()]
        _, pval_fish = combine_pvalues(pval_list, method='fisher')
        sign_fish = pval_intp(pval_fish)
    return sign_fish

# ----
# Comm
# ----
# ----
# Common : exec_2x2
# ----
def comm_exec_2x2(ceap_inpu, func_prec, func_exec, func_intp, func_expl, func_glob):
    if False:
        ceap_inpu = {
            'C0': np.array([[295, 43], [24, 0]]),
            'C1': np.array([[352, 5], [4, 1]]),
            'C2': np.array([[274, 47], [31, 10]]),
            'C3': np.array([[193, 43], [46, 80]]),
            'C4': np.array([[288, 22], [29, 23]]),
            'C5': np.array([[315, 18], [23, 6]]),
            'C6': np.array([[187, 62], [92, 21]])
        }
  
    # Step 1
    # ------
    ceap_dict = {}
    for ceap, cont_arra in ceap_inpu.items():
        print (ceap, cont_arra)
        # Prec
        # ----
        chck_isok, chck_info = func_prec(cont_arra)
        # Exec
        # ----
        stat, pval = func_exec(cont_arra) 
        # Intp
        # ----
        # pval
        sign_stat = func_intp(stat)
        expl_stat = func_expl(stat)
        # pval
        sign_pval = pval_intp(pval)
        expl_pval = None
   
        # Oupu
        # ----
        ceap_dict[ceap] = {
            'stat_deta': stat,
            'pval_deta': pval,
            'sign_stat_deta': sign_stat,
            'expl_stat_deta': expl_stat,
            'sign_pval_deta': sign_pval,
            'expl_pval_deta': expl_pval,
            'chck_info_deta': chck_info
        }
        
    # Glob
    # ----   
    stat_glob, pval_glob = func_glob(ceap_inpu, ceap_dict)

    # Step 2 : Fisher : global on p_val
    # ------
    fish_pval = fish(ceap_dict)
    
    # Round
    # -----
    if stat_glob is not None:
        stat_glob = f"{stat_glob:.3e}" if stat_glob < 0.001 else f"{stat_glob:.3f}" 
    if pval_glob is not None:
        pval_glob = f"{pval_glob:.3e}" if pval_glob < 0.001 else f"{pval_glob:.3f}"
    for ceap, resu in ceap_dict.items():
        stat_deta = resu['stat_deta']
        resu['stat_deta'] = f"{stat_deta:.3e}" if stat_deta < 0.001 else f"{stat_deta:.3f}" 
        pval_deta = resu['pval_deta']
        resu['pval_deta'] = f"{pval_deta:.3e}" if pval_deta < 0.001 else f"{pval_deta:.3f}" 
   
    # Exit
    resu_dict = {
        'stat': stat_glob,
        'pval': pval_glob,
        'sign_stat': None,
        'sign_pval': None,
        'chck_info': None,
        'expl_stat': None,
        'expl_pval': None,
        'ceap_dict': ceap_dict,
        'fish_pval': fish_pval
        }
    return resu_dict

# ----
# Stuart_maxwell test nx(2x2)
# ----
'''
The Stuart-Maxwell test is used to test the marginal homogeneity of a square contingency table. 
It can be used to identify significant asymmetries in the table.
The Stuart-Maxwell test is an extension of McNemar's test for larger tables.
-
Interpretation : if pval < alpha, there is evidence of asymmetry.
'''
def stu2_chck(cont_arra):
 
    # Vars
    prec_list = []
    if cont_arra.shape[0] != cont_arra.shape[1]:
        prec_list.append("1")
            
    # Exit
    if not prec_list:
        isok = True
        info = '✓'
    else:
        isok = False
        info = f"¬{','.join(prec_list)}"
    return isok, info

def stu2_comp(cont_arra):

    n = cont_arra.shape[0]
    row_sums = np.sum(cont_arra, axis=1)
    col_sums = np.sum(cont_arra, axis=0)
    total = np.sum(cont_arra)

    expected = np.outer(row_sums, col_sums) / total
    observed = cont_arra

    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = (n - 1) * (n - 1)
    p_value = 1 - chi2.cdf(chi2_stat, df)

    return chi2_stat, p_value, df

def stu2_exec(cont_arra):

    # Vars
    global_chi_square = 0
    global_df = 0  # degrees of freedom is the number of unique off-diagonal pairs
    ceap_dict = {}
    
    # Exec
    chi2_deta, pval_deta, dofr_deta = stu2_comp(cont_arra) 
    return chi2_deta, pval_deta
 
def stu2_intp(resu_dict):
    return None

def stu2_expl(resu_dict): 
    return None

def stu2_glob(ceap_inpu, ceap_dict):
    
    # Accumulate the global chi-square and degrees of freedom
    stat_glob = 0
    dofr_glob = 0   
    for (ceap1, resu1), (ceap2, resu2) in zip(ceap_inpu.items(), ceap_dict.items()):
        # Your code to process the key-value pairs from both dictionaries
        print(f"ceap1: {ceap1}, resu1: {resu1}")
        print(f"ceap2: {ceap2}, resu2: {resu2}")   
        # Accumulate the global chi-square and degrees of freedom
        stat_glob += resu2['stat_deta']
        print(resu1[0].shape)
        dofr_glob += resu1[0].shape[0] - 1
    # Calculate the global p-value
    pval_glob = 1 - chi2.cdf(stat_glob, dofr_glob)
    
    return stat_glob, pval_glob

def stu2_main(ceap_inpu):
    func_prec = stu2_chck
    func_exec = stu2_exec
    func_intp = stu2_intp
    func_expl = stu2_expl
    func_glob = stu2_glob
    return comm_exec_2x2(ceap_inpu, func_prec, func_exec, func_intp, func_expl, func_glob)

# --------------
# Bowker test (nx2 tables)
# --------------       
'''
Bowker's Test
Purpose: Bowker's test is used to test the asymmetry [not the symmetry] of a square contingency table. 
It is an extension of the McNemar test for larger (k x k) tables.
Usage: Used to determine if the off-diagonal elements of a square contingency table are symmetric.
Example: Suppose you have a 3x3 contingency table representing the changes in opinions 
before and after an intervention, and you want to test if the changes are symmetric.
---
The null hypothesis for Bowker's test is that the marginal distributions are symmetric, 
meaning there is no significant difference in the proportions of each sign between the left and right legs. 
If the p-value is less than your chosen significance level (e.g., 0.05), 
you can reject the null hypothesis and conclude that there is a significant difference 
in the proportions of the sign between the legs.
''' 
def bow2_chck(cont_arra): # ex. : [[257  66][ 39   0]] (for 'NA'))

    # Vars
    prec_list = []
    if cont_arra.shape[0] != cont_arra.shape[1]:
        prec_list.append(f"1")
    
    # Exit
    if not prec_list:
        isok = True
        info = '✓'
    else:
        isok = False
        info = f"¬{','.join(prec_list)}"
    return isok, info
def bow2_exec(cont_arra): # ex. : [[257  66][ 39   0]] (for 'NA'))
    
    # Exec
    # Ensure the table is 2x2
    rows, cols = cont_arra.shape
    if rows != 2 or cols != 2:
        raise ValueError(f"Table must be 2x2")

    O_ij = cont_arra[0, 1]
    O_ji = cont_arra[1, 0]
    if O_ij + O_ji > 0:
        chi_square_ij = (O_ij - O_ji) ** 2 / (O_ij + O_ji)
        p_value_ij = 1 - chi2.cdf(chi_square_ij, df=1)  # df=1 for each pair
    else:
        stat = None
        pval = None
    stat = chi_square_ij
    pval = p_value_ij
    
    return stat, pval
 
def bow2_intp(resu_dict):
    return None

def bow2_expl(resu_dict): 
    return None

def bow2_glob(ceap_inpu, ceap_dict):
    
    # Accumulate the global chi-square and degrees of freedom
    stat_glob = 0
    dofr_glob = 0   
    for (ceap1, resu1), (ceap2, resu2) in zip(ceap_inpu.items(), ceap_dict.items()):
        # Your code to process the key-value pairs from both dictionaries
        print(f"ceap1: {ceap1}, resu1: {resu1}")
        print(f"ceap2: {ceap2}, resu2: {resu2}")   
        # Accumulate the global chi-square and degrees of freedom
        stat_glob += resu2['stat_deta']
        print(resu1[0].shape)
        dofr_glob += resu1[0].shape[0] - 1
    # Calculate the global p-value
    pval_glob = 1 - chi2.cdf(stat_glob, dofr_glob)
    
    return stat_glob, pval_glob

def bow2_main(ceap_inpu):
    func_prec = bow2_chck
    func_exec = bow2_exec
    func_intp = bow2_intp
    func_expl = bow2_expl
    func_glob = bow2_glob
    return comm_exec_2x2(ceap_inpu, func_prec, func_exec, func_intp, func_expl, func_glob)
        
# --------------
# Fisher's Exact test
# --------------
'''
The odds ratio calculated in Fisher's Exact Test is based on the observed frequencies 
in the contingency table, rather than probabilities.
Fisher's Exact Test is used to determine if there are nonrandom associations between two categorical variables. 
It is particularly useful for small sample sizes where the chi-square test may not be appropriate.

The odds ratio is a measure of the strength of the association between two binary data values. 
In the context of a 2x2 contingency table, the odds ratio is calculated as:

\[ \text{Odds Ratio} = \frac{a/c}{b/d} = \frac{ad}{bc}]
where:
- \( a \) is the number of observations in the first cell (top-left).
- \( b \) is the number of observations in the second cell (top-right).
- \( c \) is the number of observations in the third cell (bottom-left).
- \( d \) is the number of observations in the fourth cell (bottom-right).

'''
def fish_chck(cont_arra): # ex. : [[257  66][ 39   0]] (for 'NA'))
   
    # Vars
    # ----        
    prec_list = []
    
    # Check if the contingency table is a 2x2 matrix
    if len(cont_arra) != 2 or any(len(row) != 2 for row in cont_arra):
        prec_list.append('1') #"Contingency table must be a 2x2 matrix.")
    
    # Check for non-negative counts
    for row in cont_arra:
        if any(count < 0 for count in row):
            prec_list.append('2') #"Counts in the contingency table must be non-negative.")
 
    # Exit
    if not prec_list:
        isok = True
        info = '✓'
    else:
        isok = False
        info = f"¬{','.join(prec_list)}"
    return isok, info

def fish_exec(cont_arra): # ex. : [[257  66][ 39   0]] (for 'NA'))
    
    mbre_left = cont_arra[0]  # [Absent, Present] for Left Leg
    mbre_righ = cont_arra[1]  # [Absent, Present] for Right Leg
    oddsratio, pval = fisher_exact([mbre_left, mbre_righ])
    
    return oddsratio, pval

def fish_intp(oddsratio):
    # Interpret the oddsratio coefficient
    if oddsratio > 1:
        odds_intp = "ceap:G>D"
    elif oddsratio < 1:
        odds_intp = "ceap:G<D"
    else:
        odds_intp = "ceap:D=G"
    return odds_intp

def fish_expl(oddsratio):
        # Interpret the oddsratio coefficient
    if oddsratio > 1:
        odds_expl = "ceap:G>D : The left leg has higher odds of having this clinical CEAP class signal compared to the right leg."
    elif oddsratio < 1:
        odds_expl = "ceap:G<D : The left leg has lower odds of having this clinical CEAP class signal compared to the right leg."
    else:
        odds_expl = "ceap:D=G : The odds of having this clinical CEAP class signal are the same for both legs."
    return odds_expl

def fish_glob(ceap_inpu, ceap_dict):
    return None, None

def fish_main(ceap_inpu):
    func_prec = fish_chck
    func_exec = fish_exec
    func_intp = fish_intp
    func_expl = fish_expl
    func_glob = fish_glob
    return comm_exec_2x2(ceap_inpu, func_prec, func_exec, func_intp, func_expl, func_glob)

# --------------
# McNemar's test (2x2 tables)
# --------------
'''
The McNemar test is used to determine if there is a significant difference in the proportions of cases 
that change from one category to another between two related groups (in this case, mbre left and right). 
It focuses on the discordant pairs (cases where the values differ between the two groups), 
which are represented by the off-diagonal elements ('b' and 'c') in the contingency table.

The McNemar test statistic is calculated based on the discordant pairs, and the resulting p-value indicates 
whether the observed difference in proportions is statistically significant.
'''
'''
The case where b + c = 0 in McNemar's test is a special situation that requires careful interpretation.

Meaning of b and c:
b: number of cases where left leg is negative (0) and right leg is positive (1)
c: number of cases where left leg is positive (1) and right leg is negative (0)

When b + c = 0:
This means there are no discordant pairs. In other words, for every patient, the left and right legs have 
the same status (both positive or both negative) for that particular CEAP class.
Interpretation:

Perfect agreement: This indicates perfect agreement between the left and right legs for this CEAP class.
Cannot compute traditional stat: The standard McNemar's stat (b-c)^2 / (b+c) is undefined when b+c=0.
P-value interpretation: In this case, the p-value would be 1, indicating no evidence against 
the null hypothesis of marginal homogeneity.
Correlation: This situation suggests a perfect positive correlation between left and right legs for this class.
'''
'''
Here's a brief explanation of the interpretation:

No association: The left and right leg conditions are completely independent for this CEAP class.
Negligible association: There's a very slight, probably meaningless relationship.
Weak association: There's a small but potentially meaningful relationship.
Moderate association: There's a considerable relationship.
Strong association: There's a strong relationship between left and right leg conditions.
Perfect association: The left and right leg conditions are perfectly correlated for this CEAP class.
'''
'''
Interpretation of resu_dict:

The McNemar's stat indicates the strength of disagreement between left and right legs.
The p-value indicates the statistical significance of this disagreement.
A small p-value (typically < 0.05) suggests a significant difference between left and right legs for that CEAP class.
'''
def nema_chck(cont_arra): # ex. : [[257  66][ 39   0]] (for 'NA'))
    
    # Vars
    prec_list = []
    m00, m01 = cont_arra[0][0], cont_arra[0][1]
    m10, m11 = cont_arra[1][0], cont_arra[1][1]
    
    # Check for sufficient sample size (at least 10 discordant pairs)
    discordant_pairs = m01 + m10
    if discordant_pairs < 10:
        prec_list.append('1') # "Insufficient sample size: There must be at least 10 discordant pairs.")
    # Check for marginal homogeneity
    line_tota = np.array([m00+m01, m10+m11])
    colu_tota = np.array([m00+m10, m01+m11])
    if not np.isclose(line_tota[0], line_tota[1]) or not np.isclose(colu_tota[0], colu_tota[1]):
        prec_list.append('2') # "Marginal homogeneity condition not satisfied: The row and column totals must be approximately equal.")
    
    print (f'a:{m00} b:{m01}')
    print (f'a:{m10} b:{m11}')
    print (f'line_tota:{line_tota} colu_tota:{colu_tota}')

    # Exit
    if not prec_list:
        isok = True
        info = '✓'
    else:
        isok = False
        info = f"¬{','.join(prec_list)}"
    return isok, info

def nema_exe1(cont_arra): # ex. : [[257  66][ 39   0]] (for 'NA'))
    # Assuming cont_tabl is a 2x2 contingency table
    m00, m01, m10, m11 = cont_arra[0][0], cont_arra[0][1], cont_arra[1][0], cont_arra[1][1]
    is_perf_agre = (m01 + m10) == 0
    if not is_perf_agre:
        stat = (abs(m01 - m10) - 1) ** 2 / (m01 + m10)
        pval = 1 - 2 * np.min([np.exp(-stat / 2), 1])  # Simplified p-value calculation
        n = np.sum(cont_arra)
        phix = sqrt(stat / n)
    else:
        stat = None
        pval = 0
        phix = 1
    return phix, pval

def nema_exe2(cont_arra): # ex. : [[257  66][ 39   0]] (for 'NA'))

    # Assuming cont_arra is a 2x2 contingency table
    # Perform McNemar's test using scipy
    result = mcnemar(cont_arra, exact=True)  # Set exact=False for large samples
    phix = result.statistic  # Get the test statistic
    pval = result.pvalue      # Get the p-value

    # Calculate phi coefficient
    n = np.sum(cont_arra)
    if (cont_arra[0][1] + cont_arra[1][0]) > 0:  # Ensure no division by zero
        phix = np.sqrt(result.statistic / n)
    else:
        phix = 0  # If there are no discordant pairs

    return phix, pval

def nema_exec(cont_arra): # ex. : [[257  66][ 39   0]] (for 'NA'))
    if False:
        return nema_exe1(cont_arra)
    else:
        return nema_exe2(cont_arra)

# H0  of McNemar's test states that there is no difference in the proportions of paired observations in a 2x2 contingency table
# If the test results in a significant p-value, the null hypothesis is rejected,
# indicating that there is a statistically significant difference in the proportions of the outcomes
def nema_intp(phix):
    
    # Interpret the phi coefficient
    if abs(phix) == 1:
        sign_phix = "5/5" # Asso:Perfect
    elif abs(phix) >= 0.5:
        sign_phix = "4/5" # Asso:Strong
    elif abs(phix) >= 0.3:
        sign_phix = "3/5" # Asso:Moderate
    elif abs(phix) >= 0.1:
        sign_phix = "2/5" # Asso:Weak
    else:
        sign_phix = "1/5" # Asso:Negligible  
        
    # Determine the direction of the association
    if phix < 0:
        sign_phix = f'-{sign_phix}' 
  
    return sign_phix

def nema_expl(phix):

    # Interpret the phi coefficient
    if abs(phix) == 1:
        expl_phix = "association:perfect"
    elif abs(phix) >= 0.5:
        sign_phix = "association:strong"
    elif abs(phix) >= 0.3:
        sign_phix = "association:moderate"
    elif abs(phix) >= 0.1:
        sign_phix = "association:weak"
    else:
        sign_phix = "association:negligible" 
        
    # Determine the direction of the association
    if phix < 0:
        sign_phix = f'{sign_phix}(negative)' 
    else:
        sign_phix = f'{sign_phix}(positive)' 
  
    return sign_phix

def nema_glob(ceap_inpu, ceap_dict):
    return None, None

def nema_main(ceap_inpu):
    func_prec = nema_chck
    func_exec = nema_exec
    func_intp = nema_intp
    func_expl = nema_expl
    func_glob = nema_glob
    return comm_exec_2x2(ceap_inpu, func_prec, func_exec, func_intp, func_expl, func_glob)

# ------------------
# Cohen's Kappa test nx2
# ------------------
'''
Concept:

The Cohen's Kappa test is used to evaluate the consistency of CEAP classifications between the left and right legs. 
For each CEAP class (C0 through C6), the test compares the observed frequencies of the classifications 
made for the left leg with those made for the right leg. The Kappa value quantifies the level of agreement 
between the two legs, adjusting for the agreement that would be expected by chance.

Interpretation:

Kappa < 0: Indicates no agreement, suggesting systematic disagreement between the left and right legs in the classification of the CEAP class.
Kappa < 0.20: Indicates slight agreement, a very low level of agreement between the left and right legs.
Kappa < 0.40: Indicates fair agreement, a low level of agreement between the left and right legs.
Kappa < 0.60: Indicates moderate agreement, a moderate level of agreement between the left and right legs.
Kappa < 0.80: Indicates substantial agreement, a high level of agreement between the left and right legs.
Kappa < 1.00: Indicates almost perfect agreement, a very high level of agreement between the left and right legs.
Kappa = 1.00: Indicates perfect agreement, complete agreement between the left and right legs.
Purpose:

The purpose of applying the Cohen's Kappa test in this context is to determine 
the reliability and consistency of CEAP classifications between the left and right legs.
A high Kappa value indicates that the classifications are consistent and reliable, 
while a low Kappa value suggests that there is significant variability or disagreement in the classifications.

'''
def cok2_chck(cont_arra): # ex. : [[257  66][ 39   0]] (for 'NA'))
    
    # Vars
    prec_list = []
    
    # Check if the contingency table is a 2x2 matrix
    if len(cont_arra) != 2 or any(len(row) != 2 for row in cont_arra):
        prec_list.append('1') # raise ValueError("Contingency table must be a 2x2 matrix.")
    
    # Check if the categories are consistent
    expected_categories = {0, 1}  # Assuming categories are 0 and 1 for binary classification
    actual_categories = {0 if cont_arra[0][0] > 0 or cont_arra[0][1] > 0 else None,
                         1 if cont_arra[1][0] > 0 or cont_arra[1][1] > 0 else None}
    actual_categories = {cat for cat in actual_categories if cat is not None}
    if actual_categories != expected_categories:
        prec_list.append('2') # raise ValueError("Both raters must use the same set of categories: {0, 1}.")

    # Exit
    if not prec_list:
        isok = True
        info = '✓'
    else:
        isok = False
        info = f"¬{','.join(prec_list)}"
    return isok, info

def cok2_util(cont_arra): # ex. : [[257  66][ 39   0]] (for 'NA'))

    # Extract counts from the contingency table
    n00, n01 = cont_arra[0]  # True Negatives, False Positives
    n10, n11 = cont_arra[1]  # False Negatives, True Positives
    # Total observations
    total = n00 + n01 + n10 + n11    
    # Observed agreement
    p_o = (n00 + n11) / total     
    # Expected agreement
    p_e = ((n00 + n01) / total) * ((n00 + n10) / total) + ((n10 + n11) / total) * ((n01 + n11) / total)      
    # Cohen's Kappa calculation
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 0  # Avoid division by zero      
    return kappa

def cok2_exec(cont_arra, num_permutations=500): # 10000 is ideal  # ex. : [[257  66][ 39   0]] (for 'NA'))

    # Step 1
    # ------
    observed_kappa = cok2_util(cont_arra)
    
    # Step 2
    # ------
    # Flatten the contingency table to create the original ratings
    ratings1 = np.concatenate([np.repeat(0, cont_arra[0, 0] + cont_arra[0, 1]), np.repeat(1, cont_arra[1, 0] + cont_arra[1, 1])])  
    ratings2 = np.concatenate([np.repeat(0, cont_arra[0, 0] + cont_arra[1, 0]), np.repeat(1, cont_arra[0, 1] + cont_arra[1, 1])])
    
    # Store permuted kappa values
    permuted_kappas = []
    for _ in range(num_permutations):
        # Shuffle the second set of ratings
        np.random.shuffle(ratings2)
        # Create a new contingency table based on shuffled ratings
        new_cont_tabl = np.array([[np.sum((ratings1 == 0) & (ratings2 == 0)), np.sum((ratings1 == 0) & (ratings2 == 1))],
                                  [np.sum((ratings1 == 1) & (ratings2 == 0)), np.sum((ratings1 == 1) & (ratings2 == 1))]])  
        # Calculate Kappa for the permuted contingency table
        permuted_kappa = cok2_util(new_cont_tabl)
        permuted_kappas.append(permuted_kappa)
    # Calculate the p-value
    p_value = np.sum(np.array(permuted_kappas) >= observed_kappa) / num_permutations
    return observed_kappa, p_value # Reject = the observed agreement between the two raters is statistically significant and is unlikely to have occurred by chance

def cok2_intp(kapp):

    if kapp < 0:
        kapp_intp = '0/5' #"Agre:0/5 No"
    elif kapp < 0.20:
        kapp_intp = '1/5' #"Agre:1/5 Slight"
    elif kapp < 0.40:
        kapp_intp = '2/5' #"Agre:2/5 Fair"
    elif kapp < 0.60:
        kapp_intp = '3/5' #"Agre:3/5 Moderate"
    elif kapp < 0.80:
        kapp_intp = '4/5' #"Agre:4a/5 Substantial"
    elif kapp < 1.00:
        kapp_intp = '4/5' #"Agre:4b/5 Almost perfect"
    else:
        kapp_intp = '5/5' #"Agre:5/5 Perfect"
    
    return kapp_intp

def cok2_expl(kapp):

    if kapp < 0:
        kapp_expl = "Kappa < 0: No agreement, indicating a systematic disagreement between the raters."
    elif kapp < 0.20:
        kapp_expl = "Kappa < 0.20: Slight agreement, indicating a very low level of agreement between the raters."
    elif kapp < 0.40:
        kapp_expl = "Kappa < 0.40: Fair agreement, indicating a low level of agreement between the raters."
    elif kapp < 0.60:
        kapp_expl = "Kappa < 0.60: Moderate agreement, indicating a moderate level of agreement between the raters."
    elif kapp < 0.80:
        kapp_expl = "Kappa < 0.80: Substantial agreement, indicating a high level of agreement between the raters."
    elif kapp < 1.00:
        kapp_expl = "Kappa < 1.00: Almost perfect agreement, indicating a very high level of agreement between the raters."
    else:
        kapp_expl = "Kappa = 1.00: Perfect agreement, indicating complete agreement between the raters."

    return kapp_expl

def cok2_glob(ceap_inpu, ceap_dict):
    return None, None

def cok2_main(ceap_inpu):
    func_prec = cok2_chck
    func_exec = cok2_exec
    func_intp = cok2_intp
    func_expl = cok2_expl
    func_glob = cok2_glob
    return comm_exec_2x2(ceap_inpu, func_prec, func_exec, func_intp, func_expl,func_glob)

# ----
# Call
# ----
'''
df_inpu:<class 'pandas.core.frame.DataFrame'>
                cont_tabl  pati_nmbr
NA   [[257, 66], [39, 0]]        362
C0   [[295, 43], [24, 0]]        362
C1     [[352, 5], [4, 1]]        362
C2  [[274, 47], [31, 10]]        362
C3  [[193, 43], [46, 80]]        362
C4  [[288, 22], [29, 23]]        362
C5   [[315, 18], [23, 6]]        362
C6  [[187, 62], [92, 21]]        362
:Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], dtype='object')
'''
def call(df_nx2):
    
    cont_dict_arra = {indx: np.array(item) for indx, item in df_nx2['cont_tabl'].items()}
    resu_dict = {}

    if True:
                
        resu_dict["stu2"] = stu2_main(cont_dict_arra)
        resu_dict["bow2"] = bow2_main(cont_dict_arra)
        resu_dict["fish"] = fish_main(cont_dict_arra)
        resu_dict["nema"] = nema_main(cont_dict_arra)
        resu_dict["cok2"] = cok2_main(cont_dict_arra)
        
    print (resu_dict)    
    return resu_dict

# ----
# Orch
# ----
def desc(df):
    return ''
def orch(what, repo, df_nx2):
    # Inpu
    plug_inpu(what, repo, df_nx2, desc(df_nx2))
    # Exec
    resu_dict = call(df_nx2)
    # Oupu
    flat_data_list = []
    for key, value in resu_dict.items():
        #print (key, value['ceap_dict'])
        if value['ceap_dict'] == {}:
           row = {
                    'key': key,
                    **{k: v for k, v in value.items() if k != 'ceap_dict'}
                } 
           flat_data_list.append(row)
        else:
            for sub_key, sub_value in value['ceap_dict'].items():
                row = {
                    'key': key,
                    'sub_key': sub_key,
                    **{k: v for k, v in sub_value.items()},
                    **{k: v for k, v in value.items() if k != 'ceap_dict'}
                }
                flat_data_list.append(row)
    df_oupu = pd.DataFrame(flat_data_list)
    df_oupu.fillna('', inplace=True)
    print(f"df_oupu:{type(df_oupu)}\n{df_oupu}\n:{df_oupu.index}")
    # Trac
    plug_oupu(what, repo, df_oupu)
    pass

# ----
# Main
# ----
def ke43_ceap_unbi_deta_NAC6(df_nxn, df_nx2, sexe):
    df_nxn = mtrx_nxn_selc(df_nxn, 'NA', 'C6')
    df_nx2 = mtrx_nx2_selc(df_nx2, 'NA', 'C6')
    what = inspect.currentframe().f_code.co_name
    what = f'{what} sexe:{sexe}'
    return what, df_nxn, df_nx2
def ke43_ceap_unbi_deta_C0C6(df_nxn, df_nx2, sexe):
    df_nxn = mtrx_nxn_selc(df_nxn, 'C0', 'C6')
    df_nx2 = mtrx_nx2_selc(df_nx2, 'C0', 'C6')
    what = inspect.currentframe().f_code.co_name
    what = f'{what} sexe:{sexe}'
    return what, df_nxn, df_nx2
def ke43_ceap_unbi_deta_NAC2(df_nxn, df_nx2, sexe):
    df_nxn = mtrx_nxn_selc(df_nxn, 'NA', 'C2')
    df_nx2 = mtrx_nx2_selc(df_nx2, 'NA', 'C2')
    what = inspect.currentframe().f_code.co_name
    what = f'{what} sexe:{sexe}'
    return what, df_nxn, df_nx2
def ke43_ceap_unbi_deta_C0C2(df_nxn, df_nx2, sexe):
    df_nxn = mtrx_nxn_selc(df_nxn, 'C0', 'C2')
    df_nx2 = mtrx_nx2_selc(df_nx2, 'C0', 'C2')
    what = inspect.currentframe().f_code.co_name
    what = f'{what} sexe:{sexe}'
    return what, df_nxn, df_nx2
def ke43_ceap_unbi_deta_C3C6(df_nxn, df_nx2, sexe):
    df_nxn = mtrx_nxn_selc(df_nxn, 'C3', 'C6')
    df_nx2 = mtrx_nx2_selc(df_nx2, 'C3', 'C6')
    what = inspect.currentframe().f_code.co_name
    what = f'{what} sexe:{sexe}'
    return what, df_nxn, df_nx2

if __name__ == "__main__":
    
    # Stat
    # ----
    if True:

        df_A_nxn, df_A_nx2, df_M_nxn, df_M_nx2, df_F_nxn, df_F_nx2 = inpu_func() 
        print(f"df_inpu:{type(df_A_nxn)}\n{df_A_nxn}\n:{df_A_nxn.index}")
        print(f"df_inpu:{type(df_A_nx2)}\n{df_A_nx2}\n:{df_A_nx2.index}")
        
        repo = pgrm_init()
        #
        def exec(df_nxn, df_nx2, sexe):
            what, df_nxn_NAC6, df_nx2_NAC6 = ke43_ceap_unbi_deta_NAC6(df_nxn, df_nx2, sexe)
            orch(what, repo, df_nx2_NAC6)
            if False:
                what, df_nxn_C0C0, df_nx2_C0C6 = ke43_ceap_unbi_deta_C0C6(df_nxn, df_nx2, sexe)
                orch(what, repo, df_nx2_C0C6)
                what, df_nxn_NAC2, df_nx2_NAC2 = ke43_ceap_unbi_deta_NAC2(df_nxn, df_nx2, sexe)
                orch(what, repo, df_nx2_NAC2)
                what, df_nxn_C0C2, df_nx2_C0C2 = ke43_ceap_unbi_deta_C0C2(df_nxn, df_nx2, sexe)
                orch(what, repo, df_nx2_C0C2)
                what, df_nxn_C3C6, df_nx2_C3C6 = ke43_ceap_unbi_deta_C3C6(df_nxn, df_nx2, sexe)
                orch(what, repo, df_nx2_C3C6)
        #
        exec(df_A_nxn, df_A_nx2, 'A') # sexe M+F
        # exec(df_M_nxn, df_M_nx2, 'M') # sexe M
        # exec(df_F_nxn, df_F_nx2, 'F') # sexe F
        #
        pgrm_fini(repo, os.path.splitext(os.path.basename(__file__)))
        pass