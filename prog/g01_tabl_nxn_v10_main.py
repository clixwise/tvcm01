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

# (nxn tables) Fre1,2,3,4         
# (nxn tables) Marginal symmetry (McNemar-Bowker) test
# (nxn tables) Marginal Homogeneity (Stuart-Maxwell) test
# (nxn tables) Cramer V test
# (nxn tables) Kendall tau test
# (nxn tables) Spearman test
# (nxn tables) Pearson test
# (nxn tables) Chi2 test   
# (nxn tables) Residual test         
# (nxn tables) Stuart_Maxwell test 
# (nxn tables) Bowker test                
# (nxn tables) Permutation test    
# (nxn tables) Log linear test 
# (nxn tables) Cohen's Kappa test 

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
# Common : exec_nxn
# ----
def comm_exec_nxn(ceap_labl, ceap_arra, func_prec, func_exec, func_intp, func_expl):

    # Step 1
    # ------
    
    # Prec
    # ----
    chck_isok, chck_info = func_prec(ceap_arra)
    # Exec
    # ----
    resu_dict = func_exec(ceap_labl, ceap_arra) 
    stat = resu_dict['stat']
    pval = resu_dict['pval']
    ceap_dict = resu_dict['ceap_dict']
    # Intp
    # ----
    # stat
    sign_stat = func_intp(resu_dict)
    expl_stat = func_expl(resu_dict)
    # pval
    sign_pval = pval_intp_list(pval, ceap_dict)
    expl_pval = pval_expl_list(pval, ceap_dict) 
        
    # Step 2 : Fisher : global on p_val
    # ------
    fish_pval = fish(ceap_dict)
    
    # Exit
    if isinstance(stat, float): 
        stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    else:
        stat_form = stat
    if isinstance(pval, float): 
        pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    else:
        stat_form = pval
    resu_dict = {
        'stat': stat_form,
        'pval': pval_form,
        'sign_stat': sign_stat,
        'sign_pval': sign_pval,
        'chck_info': chck_info,
        'expl_stat': expl_stat,
        'expl_pval': expl_pval,
        'ceap_dict': ceap_dict,
        'fish_pval': fish_pval
    }
    return resu_dict

# ----
# Stat
# ----

# ----
# Highest frequency test : percentages : fre2onal, lower triangle, higher triangle
# ----
def fre1_chck(df_inpu):
    return None
def fre1_intp(stat): 
    return None
def fre1_expl(sign_stat): 
    return None
def fre1_main(df_inpu):
    #print(f"df_inpu:{type(df_inpu)} len:{len(df_inpu)}\n{df_inpu}\n:{df_inpu.index}\n:{df_inpu.columns}")
    if False:
        data = {
            'C3': [80, 16, 3, 14],
            'C4': [17, 23, 7, 5],
            'C5': [4, 2, 6, 8],
            'C6': [9, 5, 7, 21]
        }
        df_inpu = pd.DataFrame(data, index=['C3', 'C4', 'C5', 'C6'])
    # prec
    chck_info = fre1_chck(df_inpu)
    # exec
    total_sum = df_inpu.values.sum()
    diagonal_sum = np.diag(df_inpu.values).sum()
    lower_left_sum = np.tril(df_inpu.values, k=-1).sum()
    upper_right_sum = np.triu(df_inpu.values, k=1).sum()
    fre2onal_percentage = (diagonal_sum / total_sum) * 100
    lower_left_percentage = (lower_left_sum / total_sum) * 100
    upper_right_percentage = (upper_right_sum / total_sum) * 100
    stat = f'totl:{total_sum} fre2:{diagonal_sum} tri_low:{lower_left_sum} tri_upp:{upper_right_sum} ; fre2:{fre2onal_percentage:.2f}% tri_low:{lower_left_percentage:.2f}% tri_upp{upper_right_percentage:.2f}%'
    pval = None
    # intp
    sign_stat = fre1_intp(stat)
    expl_stat = fre1_expl(sign_stat)
    sign_pval = None
    expl_pval = None
    # cohe
    sign_cohn = None
    # exit
    resu_dict = {
        'stat': stat,
        'pval': pval,
        'sign_stat': sign_stat,
        'sign_pval': sign_pval,
        'sign_cohn': sign_cohn,
        'expl_stat': expl_stat,
        'expl_pval': expl_pval,
        'chck_info': chck_info,
        'ceap_dict': {}
        }
    return resu_dict

# ----
# Diagonal test
# ----
def fre2_chck(df_inpu):
    return None
def fre2_intp(same_class_percentage): 
    if same_class_percentage > 75:
        fre2_sign = "3/3" # print("   Very high symmetry: The vast majority of patients have the same CEAP classification in both legs.")
    elif same_class_percentage > 50:
        fre2_sign = "2/3" # print("   High symmetry: Most patients have the same CEAP classification in both legs.")
    elif same_class_percentage > 25:
        fre2_sign = "1/3" # print("   Moderate symmetry: A substantial portion of patients have the same CEAP classification in both legs.")
    else:
        fre2_sign = "0/3" # print("   Low symmetry: Most patients have different CEAP classifications in their left and right legs.")
    return fre2_sign
def fre2_expl(fre2_sign): 
    match fre2_sign:
        case "3/3":
            fre2_expl = "Very high symmetry: The vast majority of patients have the same CEAP classification in both legs."
        case "2/3":
            fre2_expl = "High symmetry: Most patients have the same CEAP classification in both legs."
        case "1/3":
            fre2_expl = "Moderate symmetry: A substantial portion of patients have the same CEAP classification in both legs."
        case "0/3":
            fre2_expl = "Low symmetry: Most patients have different CEAP classifications in their left and right legs."
        case _:
            raise Exception()
    return fre2_expl
def fre2_main(df_inpu):
    #print(f"df_inpu:{type(df_inpu)} len:{len(df_inpu)}\n{df_inpu}\n:{df_inpu.index}\n:{df_inpu.columns}")
    # prec
    chck_info = fre2_chck(df_inpu)
    # exec
    ceap_fre2 = np.trace(df_inpu.values) # somme des cellules diagonales
    ceap_tota = df_inpu.values.sum() # total des CEAP
    stat = same_class_percentage = (ceap_fre2 / ceap_tota) * 100
    pval = None
    # intp
    sign_stat = fre2_intp(stat)
    expl_stat = fre2_expl(sign_stat)
    sign_pval = None
    expl_pval = None
    # cohe
    sign_cohn = None
    # exit
    resu_dict = {
        'stat': stat,
        'pval': pval,
        'sign_stat': sign_stat,
        'sign_pval': sign_pval,
        'sign_cohn': sign_cohn,
        'expl_stat': expl_stat,
        'expl_pval': expl_pval,
        'chck_info': chck_info,
        'ceap_dict': {}
        }
    return resu_dict
# ----
# Highest frequency test : highest value overall
# ----
def fre3_chck(df_inpu):
    return None
def fre3_intp(stat): 
    return None
def fre3_expl(sign_stat): 
    return None
def fre3_main(df_inpu):
    #print(f"df_inpu:{type(df_inpu)} len:{len(df_inpu)}\n{df_inpu}\n:{df_inpu.index}\n:{df_inpu.columns}")
    # prec
    chck_info = fre3_chck(df_inpu)
    # exec
    max_value = df_inpu.values.max()
    #max_location = np.unravel_index(df_inpu.values.argmax(), df_inpu.values.shape)
    max_locations = df_inpu.stack().loc[lambda x: x == max_value].index.tolist()
    percentage = (max_value / df_inpu.sum().sum()) * 100
    #line_0 = df_inpu.index[max_location[0]]
    #colu_1 = df_inpu.columns[max_location[1]]
    stat = f'{max_value}({round(percentage)}%){max_locations}'
    pval = None
    # intp
    sign_stat = fre3_intp(stat)
    expl_stat = fre3_expl(sign_stat)
    sign_pval = None
    expl_pval = None
    # cohe
    sign_cohn = None
    # exit
    resu_dict = {
        'stat': stat,
        'pval': pval,
        'sign_stat': sign_stat,
        'sign_pval': sign_pval,
        'sign_cohn': sign_cohn,
        'expl_stat': expl_stat,
        'expl_pval': expl_pval,
        'chck_info': chck_info,
        'ceap_dict': {}
        }
    return resu_dict

# ----
# Asymmetry test
# ----
def fre4_chck(df_inpu):
    return None
def fre4_intp(max_value, max_locations, min_value,min_locations):
    sign_stat = None
    return sign_stat
def fre4_expl(max_value, max_locations, min_value,min_locations):
    fre4_expl = None
    return fre4_expl
def fre4_main(df_inpu):
    print(f"df_inpu:{type(df_inpu)} len:{len(df_inpu)}\n{df_inpu}\n:{df_inpu.index}\n:{df_inpu.columns}")
    # prec
    chck_info = fre4_chck(df_inpu)
    if False:
        df_inpu.index = ['L_' + idx for idx in df_inpu.index]
        df_inpu.columns = ['R_' + col for col in df_inpu.columns]
        print (df_inpu)
    # exec
    if False:
        data = {
            'R_C3': [0, -1, -1, 5],
            'R_C4': [1, 0, 5, 0],
            'R_C5': [1, -5, 0, 1],
            'R_C6': [-5, 0, -1, 0]
        }
        df_asym = pd.DataFrame(data, index=['L_C3', 'L_C4', 'L_C5', 'L_C6'])
    df_asym = df_inpu - df_inpu.T
    df_asym.index = ['L_' + idx for idx in df_asym.index]
    df_asym.columns = ['R_' + col for col in df_asym.columns]
    #print (df_asym)
    mask = np.tril(np.ones(df_asym.shape), k=-1).astype(bool)
    df_asym[mask] = 0
    #print(df_asym)
    #
    max_value = df_asym.max().max()
    max_locations = df_asym.stack().loc[lambda x: x == max_value].index.tolist()
    min_value = df_asym.min().min()
    min_locations = df_asym.stack().loc[lambda x: x == min_value].index.tolist()
    print (f'max_locations:{max_locations};min_locations:{min_locations}')
    #
    max = f'[' + ', '.join([f"('{row}', '{col}')" for row, col in max_locations]).replace(' ', '') + ']'
    min = f'[' + ', '.join([f"('{row}', '{col}')" for row, col in min_locations]).replace(' ', '') + ']'
    stat = f'diff L>R:{max_value}@{max};diff L<R:{min_value}@{min}'
    pval = None
    # intp
    sign_stat = fre4_intp(max_value, max_locations, min_value,min_locations)
    expl_stat = fre4_expl(max_value, max_locations, min_value,min_locations)
    sign_pval = None
    expl_pval = None
    # cohe
    sign_cohn = None
    # exit
    resu_dict = {
        'stat': stat,
        'pval': pval,
        'sign_stat': sign_stat,
        'sign_pval': sign_pval,
        'sign_cohn': sign_cohn,
        'expl_stat': expl_stat,
        'expl_pval': expl_pval,
        'chck_info': chck_info,
        'ceap_dict': {}
        }
    return resu_dict

# ----
# Marginal Symmetry (McNemar-Bowker) test 
# ----
def symm_chck(df_inpu):
    return None
def symm_pval_expl(pval): 
    if pval < 0.05:
        pval_expl = "There is evidence of an asymmetry between left and right leg CEAP classifications"
        pval_expl += "This suggests that the overall pattern of CEAP classifications differs between legs"
    else:
        pval_expl = "There is no evidence of an asymmetry between left and right leg CEAP classifications"
        pval_expl += "This suggests that the overall pattern of CEAP classifications is similar for both legs"
    return pval_expl
def symm_main(df_inpu):
    print(f"df_inpu:{type(df_inpu)} len:{len(df_inpu)}\n{df_inpu}\n:{df_inpu.index}\n:{df_inpu.columns}")
    # prec
    chck_info = symm_chck(df_inpu)
    # exec
    st = SquareTable(df_inpu)
    symmetry_result = st.symmetry()
    stat = symmetry_result.statistic
    pval = symmetry_result.pvalue
    # intp
    sign_stat = None
    expl_stat = None
    sign_pval = pval_intp(pval)
    expl_pval = symm_pval_expl(pval)
    # cohe
    sign_cohn = None
    # exit
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    # exit
    resu_dict = {
        'stat': stat_form,
        'pval': pval_form,
        'sign_stat': sign_stat,
        'sign_pval': sign_pval,
        'sign_cohn': sign_cohn,
        'expl_stat': expl_stat,
        'expl_pval': expl_pval,
        'chck_info': chck_info,
        'ceap_dict': {}
        }
    return resu_dict

# ----
# Marginal Homogeneity (Stuart-Maxwell) test
# ----
def homo_chck(df_inpu):
    return None
def homo_pval_expl(pval): 
    if pval < 0.05:
        pval_expl = "There is evidence of a difference between the marginal distributions of the rows and columns"
        pval_expl += "This suggests that the overall pattern of CEAP classifications differs between legs"
    else:
        pval_expl = "There is no evidence of a difference between the marginal distributions of the rows and columns"
        pval_expl += "This suggests that the overall pattern of CEAP classifications is similar for both legs"
    return pval_expl
def homo_main(df_inpu):
    print(f"df_inpu:{type(df_inpu)} len:{len(df_inpu)}\n{df_inpu}\n:{df_inpu.index}\n:{df_inpu.columns}")
    # prec
    chck_info = homo_chck(df_inpu)
    # exec
    st = SquareTable(df_inpu)
    homogeneity_result = st.homogeneity()
    stat = homogeneity_result.statistic
    pval = homogeneity_result.pvalue
    # intp
    sign_stat = None
    expl_stat = None
    sign_pval = pval_intp(pval)
    expl_pval = homo_pval_expl(pval)
    # cohe
    sign_cohn = None
    # exit
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    resu_dict = {
        'stat': stat_form,
        'pval': pval_form,
        'sign_stat': sign_stat,
        'sign_pval': sign_pval,
        'sign_cohn': sign_cohn,
        'expl_stat': expl_stat,
        'expl_pval': expl_pval,
        'chck_info': chck_info,
        'ceap_dict': {}
        }
    return resu_dict    


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
    
    # Vars
    prec_list = []  
    
    #
    isok = not(len(df_obsv.index.unique()) < 2 or len(df_obsv.columns.unique()) < 2)
    prec_list.append('8') # Check if each variable has at least two unique values
    
    # Exit
    if not prec_list:
        isok = True
    else:
        isok = False
    return isok, prec_list

def cram_chck_post(df_expe, prec_list):
    if not prec_list:
        isok = True
        resu = '✓'
    else:
        isok = False
        resu = f"¬{','.join(prec_list)}"
    return isok, resu
def cram_stat_intp(stat):
    if stat < 0.1:
        cram_sign = "0/3" # print(f"{what} Cramér's V indicates a negligible association.")
    elif stat < 0.3:
        cram_sign = "1/3" # print(f"{what} Cramér's V indicates a weak association.")
    elif stat < 0.5:
        cram_sign = "2/3" # print(f"{what} Cramér's V indicates a moderate association.")
    else:
        cram_sign = "3/3" # print(f"{what} Cramér's V indicates a strong association.")
    return cram_sign
def cram_stat_expl(stat):
    if stat < 0.1:
        cram_expl = f"Cramér's V indicates a negligible association [Note : chi2 conditions and explanations apply too]"
    elif stat < 0.3:
        cram_expl = f"Cramér's V indicates a weak association [Note : chi2 conditions and explanations apply too]"
    elif stat < 0.5:
        cram_expl = f"Cramér's V indicates a moderate association [Note : chi2 conditions and explanations apply too]"
    else:
        cram_expl = f"Cramér's V indicates a strong association [Note : chi2 conditions and explanations apply too]"
    return cram_expl
def cram_exec(df_inpu):
    chi2, pval, dof, expe = chi2_contingency(df_inpu)
    n = df_inpu.values.sum() if isinstance(df_inpu, pd.DataFrame) else df_inpu.sum()
    min_dim = min(df_inpu.shape) - 1
    stat = np.sqrt(chi2 / (n * min_dim))
    return stat, chi2, pval, dof, expe 
def cram_main(df_inpu):
    # prec
    isOK_ante, chck_prec_list = cram_chck_ante(df_inpu)
    # exec
    stat, chi2, pval, dof, expe = cram_exec(df_inpu)
    # intp
    sign_stat = cram_stat_intp(stat)
    expl_stat = cram_stat_expl(stat)
    sign_pval = pval_intp(pval)
    expl_pval = None
    # cohe
    sign_cohn = None
    # post
    isOK_post, chck_info_post = cram_chck_post(expe, chck_prec_list)
    # exit
    stat_form = f"{stat:.3e}" if stat < 0.001 else f"{stat:.3f}"
    pval_form = f"{pval:.3e}" if pval < 0.001 else f"{pval:.3f}"
    resu_dict = {
        'stat': stat_form,
        'pval': pval_form,
        'sign_stat': sign_stat,
        'sign_pval': sign_pval,
        'sign_cohn': sign_cohn,
        'expl_stat': expl_stat,
        'expl_pval': expl_pval,
        'chck_info': chck_info_post,
        'ceap_dict': {}
        }
    return resu_dict

# ----
# kendall tau test (global) [MOVED TO 'CONT_TABL']
# ----
'''
Kendall's Tau Test
Kendall's Tau is a non-parametric statistical test used to measure 
the ordinal association (correlation) between two measured quantities. 
It assesses how well the order of data in one variable predicts the order of data in another variable. 
The test returns two main values:

tau (stat): The Kendall Tau correlation coefficient, which ranges from -1 to 1.

A value of 1 indicates a perfect positive correlation.
A value of -1 indicates a perfect negative correlation.
A value of 0 indicates no correlation.
p-value (pval): The p-value associated with the test statistic.

A small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis, 
suggesting a significant correlation.
A large p-value (> 0.05) indicates weak evidence against the null hypothesis, 
suggesting no significant correlation.
'''
def keng_chck(ceap_arra):
    # Vars
    prec_list = []
    
    # Exec
    
    # Exit
    if not prec_list:
        isok = True
        info = '✓'
    else:
        isok = False
        info = f"¬{','.join(prec_list)}"
    return isok, info
def keng_stat_intp(resu_dict):
    stat = resu_dict['stat']
    if stat > 0.7:
        sign_stat = f"stat:{stat};corr:+1" #print("There is a strong positive correlation between the sums of occurrences for left and right clinical signs.")
    elif stat < -0.7:
        sign_stat = f"stat:{stat};corr:-1" #print("There is a strong negative correlation between the sums of occurrences for left and right clinical signs.")
    else:
        sign_stat = f"stat:{stat};corr:0"  #print("There is a weak or no correlation between the sums of occurrences for left and right clinical signs.")
        return sign_stat
def keng_stat_expl(resu_dict):
    return None
def keng_exec(ceap_labl, ceap_arra):

    # Create lists of left leg (L) and right leg (R) CEAP class ranks
    rank_list_left = []
    rank_list_righ = []
    #
    for i in range(ceap_arra.shape[0]):
        for j in range(ceap_arra.shape[1]):
            frequency = ceap_arra[i,j]
            if frequency > 0:
                # print (i, j, frequency)
                rank_list_left.extend([i] * frequency)
                rank_list_righ.extend([j] * frequency)
    #        
    stat, pval = kendalltau(rank_list_left, rank_list_left)
    # Exit
    ceap_dict = {}
    resu_dict = {
        'stat': stat,
        'pval': pval,
        'dofr': None,
        'ceap_dict': ceap_dict
    }
    return resu_dict

def keng_main(ceap_labl, ceap_inpu):
    func_prec = keng_chck
    func_exec = keng_exec
    func_intp = keng_stat_intp
    func_expl = keng_stat_expl
    return comm_exec_nxn(ceap_labl, ceap_inpu, func_prec, func_exec, func_intp, func_expl)

# ----
# kendall tau test (detail) [MOVED TO 'CONT_TABL']
# ----
def kend_chck(df_inpu):
    return None
def kend_stat_intp(tau):
    abs_tau = abs(tau)
    if abs_tau < 0.25:
        intp = "0/3"
    elif 0.25 <= abs_tau < 0.50:
        intp = "1/3"
    elif 0.50 <= abs_tau < 0.75:
        intp = "2/3"
    elif 0.75 <= abs_tau <= 1.00:
        intp = "3/3"
    else:
        raise Exception()
    return intp
def kend_main(df_inpu):
    print(f"df_inpu:{type(df_inpu)} len:{len(df_inpu)}\n{df_inpu}\n:{df_inpu.index}\n:{df_inpu.columns}")
    # prec
    chck_info = kend_chck(df_inpu)
    # exec
    results = []
    for left in df_inpu.index:
        for right in df_inpu.columns:
            stat, pval = kendalltau(df_inpu.loc[left], df_inpu[right])
            stat_intp = kend_stat_intp(stat)
            results.append({
                'L': left,
                'R': right,
                'stat': stat,
                'pval': pval,
                'intp': stat_intp
            })
    df_stat = pd.DataFrame(results)
    df_stat = df_stat.sort_values(by=['intp', 'L', 'R'], ascending=[False, True, True])
    df_stat = df_stat[['intp','L','R','stat','pval']]
    df_stat[['stat','pval']] = df_stat[['stat','pval']].round(2)
    stat = df_stat
    #print (df_stat)
    pval = None
    # intp
    sign_stat = None
    expl_stat = None
    sign_pval = None
    expl_pval = None
    # cohe
    sign_cohn = None
    # exit
    return stat, pval, sign_stat, sign_pval, sign_cohn, chck_info, expl_stat, expl_pval

# ----
# Spearman's rank correlation test
# ----
def spea_chck(ceap_arra):
    """
    Preconditions:
    1. Ordinal or continuous data
    2. Monotonic relationship between variables
    """
    # Vars
    prec_list = []
    
    # Exec
    
    # Exit
    if not prec_list:
        isok = True
        info = '✓'
    else:
        isok = False
        info = f"¬{','.join(prec_list)}"
    return isok, info
def spea_stat_intp(resu_dict):
    stat = resu_dict['stat']
    #
    if abs(stat) < 0.3:
        sign_stat = '1/3' # print("The correlation is weak.")
    elif abs(stat) < 0.7:
        sign_stat = '2/3' # print("The correlation is moderate.")
    else:
        sign_stat = '3/3' # print("The correlation is strong.")
    #
    if stat > 0:
        sign_stat = f'+{sign_stat}'
    else:
        sign_stat = f'-{sign_stat}'
    #
    return sign_stat
def spea_stat_expl(resu_dict):
    stat = resu_dict['stat']
    #
    if abs(stat) < 0.3:
        stat_expl = "weak correlation" # print("The correlation is weak.")
    elif abs(stat) < 0.7:
        stat_expl = "moderate correlation" # print("The correlation is moderate.")
    else:
        stat_expl = "strong correlation" # print("The correlation is strong.")
    #
    if stat > 0:
        stat_expl = f'+ {stat_expl}'
    else:
        stat_expl = f'- {stat_expl}'
        
    return stat_expl
def spea_exec(ceap_labl, ceap_arra):

    # Create lists of left leg (L) and right leg (R) CEAP class ranks
    rank_list_left = []
    rank_list_righ = []
    #
    for i in range(ceap_arra.shape[0]):
        for j in range(ceap_arra.shape[1]):
            frequency = ceap_arra[i,j]
            if frequency > 0:
                # print (i, j, frequency)
                rank_list_left.extend([i] * frequency)
                rank_list_righ.extend([j] * frequency)
    #        
    corr, pval = stats.spearmanr(rank_list_left, rank_list_left) # corr = correlation coefficient
    # Exit
    ceap_dict = {}
    resu_dict = {
        'stat': corr,
        'pval': pval,
        'dofr': None,
        'ceap_dict': ceap_dict
    }
    return resu_dict

def spea_main(ceap_labl, ceap_inpu):
    func_prec = spea_chck
    func_exec = spea_exec
    func_intp = spea_stat_intp
    func_expl = spea_stat_expl
    return comm_exec_nxn(ceap_labl, ceap_inpu, func_prec, func_exec, func_intp, func_expl)
# ----
# Spearman's rank correlation [MOVED TO 'CONT_TABL']
# ----
def pear_chck(ceap_arra):
    
    # Data
    # ----
    prec_list = []
    full = False
    
    # Create lists of left leg (L) and right leg (R) CEAP class ranks
    rank_list_left = []
    rank_list_righ = []
    #
    for i in range(ceap_arra.shape[0]):
        for j in range(ceap_arra.shape[1]):
            frequency = ceap_arra[i,j]
            if frequency > 0:
                # print (i, j, frequency)
                rank_list_left.extend([i] * frequency)
                rank_list_righ.extend([j] * frequency)
    
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
        X = rank_list_left
        Y = rank_list_righ
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        model = sm.OLS(Y, X).fit()
        r_squared = model.rsquared
        print(f'R-squared value: {r_squared}')
        # Interpretation of R-squared value
        if r_squared > 0.7:
            print('The linearity assumption is met (R-squared > 0.7)')
            if full: prec_list.append('l:y')
        else:
            print('The linearity assumption is not met (R-squared <= 0.7)')
            prec_list.append('l:n' if full else 'l')

    # ---------
    # Normality
    # ---------
    stat_m, p_m = shapiro(rank_list_left)
    stat_f, p_f = shapiro(rank_list_righ)
    print(f'Shapiro test for M: Statistic={stat_m}, p-value={p_m}')
    print(f'Shapiro test for F: Statistic={stat_f}, p-value={p_f}')
    alpha = 0.05
    if p_m > alpha:
        print('M is normally distributed (fail to reject H0)')
        if full: prec_list.append('mn:y')
    else:
        print('M is not normally distributed (reject H0)')
        prec_list.append('mn:n' if full else 'mn')
    if p_f > alpha:
        print('F is normally distributed (fail to reject H0)')
        if full: prec_list.append('fn:y')
    else:
        print('F is not normally distributed (reject H0)')
        prec_list.append('fn:n' if full else 'fn')

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
        X = rank_list_left
        Y = rank_list_righ
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
            if full: prec_list.append('h:y')
        else:
            print('The homoscedasticity assumption is not met (reject H0)')
            prec_list.append('h:n' if full else 'h')

    # Exit
    if not prec_list:
        isok = True
        info = '✓'
    else:
        isok = False
        info = f"¬{','.join(prec_list)}"
        
    return isok, info

def pear_stat_intp(resu_dict):
    stat = resu_dict['stat']

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

def pear_stat_expl(resu_dict):
    stat = resu_dict['stat']
    #
    if abs(stat) < 0.3:
        stat_expl = "weak correlation" # print("The correlation is weak.")
    elif abs(stat) < 0.7:
        stat_expl = "moderate correlation" # print("The correlation is moderate.")
    else:
        stat_expl = "strong correlation" # print("The correlation is strong.")
    #
    if stat > 0:
        stat_expl = f'+ {stat_expl}'
    else:
        stat_expl = f'- {stat_expl}'
        
    return stat_expl
def pear_exec(ceap_labl, ceap_arra):

    # Create lists of left leg (L) and right leg (R) CEAP class ranks
    rank_list_left = []
    rank_list_righ = []
    #
    for i in range(ceap_arra.shape[0]):
        for j in range(ceap_arra.shape[1]):
            frequency = ceap_arra[i,j]
            if frequency > 0:
                # print (i, j, frequency)
                rank_list_left.extend([i] * frequency)
                rank_list_righ.extend([j] * frequency)
    #        
    corr, pval = stats.pearsonr(rank_list_left, rank_list_left) # corr = correlation coefficient
    # Exit
    ceap_dict = {}
    resu_dict = {
        'stat': corr,
        'pval': pval,
        'dofr': None,
        'ceap_dict': ceap_dict
    }
    return resu_dict

def pear_main(ceap_labl, ceap_inpu):
    func_prec = pear_chck
    func_exec = pear_exec
    func_intp = pear_stat_intp
    func_expl = pear_stat_expl
    return comm_exec_nxn(ceap_labl, ceap_inpu, func_prec, func_exec, func_intp, func_expl)

# ----
# Chi2 [full,part] test
# ----

def chi1_chck(ceap_arra):
        
    # Vars
    prec_list = [] 
    
    # Exit
    if not prec_list:
        isok = True
        resu = '✓'
    else:
        isok = False
        resu = f"¬{','.join(prec_list)}"
    return isok, resu # , df_clean
def chi1_exec(ceap_labl, ceap_arra):

    # Exec
    stat, pval, dof, expected = chi2_contingency(ceap_arra)
    print (f'ceap_labl:{ceap_labl} ; stat:{stat} ; pval:{pval}')
    # Exit
    ceap_dict = {}
    return {
        'stat': stat,
        'pval': pval,
        'dofr': dof,
        'expe':expected,
        'ceap_dict':ceap_dict
    } 
'''
if p_value < 0.05:
    print("The p-value is less than 0.05, indicating a significant relationship between the signs on the left and right legs.")
    print("This suggests that the distribution of signs is not independent between the legs.")
else:
    print("The p-value is greater than 0.05, suggesting no significant relationship between the signs on the left and right legs.")
    print("This implies that the distribution of signs might be independent between the legs.")
'''
def chi1_intp(resu_dict): 
    stat = resu_dict['stat']
    pval = resu_dict['pval']
    dofr = resu_dict['dofr']
    alpha = 0.05
    critical_value = chi2.ppf(1 - alpha, dofr)  # Critical value for the given alpha and dof
    if stat > critical_value:
        intp = "sign diff : yes"
    else:
        intp = "sign diff : not"  
    return intp
def chi1_expl(resu_dict): 
    stat = resu_dict['stat']
    pval = resu_dict['pval']
    dofr = resu_dict['dofr']
    alpha = 0.05
    critical_value = chi2.ppf(1 - alpha, dofr)  # Critical value for the given alpha and dof
    if stat > critical_value:
        expl = "The Chi-squared statistic is greater than the critical value, suggesting a significant association."
    else:
        expl = "The Chi-squared statistic is less than the critical value, suggesting no significant association."
    return expl
def chi1_main(ceap_labl, ceap_arra):
    func_prec = chi1_chck
    func_exec = chi1_exec
    func_intp = chi1_intp
    func_expl = chi1_expl
    return comm_exec_nxn(ceap_labl, ceap_arra, func_prec, func_exec, func_intp, func_expl)

# ----
# Residuals
# ----

def res1_chck(ceap_arra):
 
    # Exit
    isok = True
    resu = '✓'
    return isok, resu 

def res1_exec(ceap_labl, ceap_arra):
    if False:
        ceap_arra = np.array([
            [0, 0, 0, 2, 18, 5, 0, 22],
            [0, 0, 0, 1, 8, 2, 2, 14],
            [0, 0, 1, 0, 3, 2, 1, 0],
            [3, 1, 1, 10, 14, 3, 3, 18],
            [20, 9, 0, 15, 80, 16, 3, 14],
            [9, 2, 1, 3, 17, 23, 7, 5],
            [7, 5, 1, 5, 4, 2, 6, 8],
            [39, 29, 3, 33, 9, 5, 7, 21]
        ])
    # Row and column totals
    row_totals = ceap_arra.sum(axis=1)
    col_totals = ceap_arra.sum(axis=0)
    total = ceap_arra.sum()
    
    # Expected values
    # ---------------
    ceap_arra_expd = np.outer(row_totals, col_totals) / total
    df_expd = pd.DataFrame(ceap_arra_expd, index=ceap_labl, columns=ceap_labl)
    df_expd.columns = ['D_' + col for col in df_expd.columns]
    df_expd.index = ['G_' + str(i) for i in df_expd.index]
    df_expd = df_expd.round(3)
    print (f'df_expected\n{df_expd}\n')
    
    # Standardized residuals
    # ----------------------
    standardized_residuals = (ceap_arra - ceap_arra_expd) / np.sqrt(ceap_arra_expd)
    # Create a dataframe for easier interpretation
    df_standardized_residuals = pd.DataFrame(standardized_residuals, index=ceap_labl, columns=ceap_labl)
    df_standardized_residuals.columns = ['D_' + col for col in df_standardized_residuals.columns]
    df_standardized_residuals.index = ['G_' + str(i) for i in df_standardized_residuals.index]
    df_standardized_residuals = df_standardized_residuals.round(3)
    print (f'df_standardized_residuals\n{df_standardized_residuals}\n')
    
    # Exit
    ceap_dict = {}
    return {
        'stat': None,
        'pval': None,
        'dofr': None,
        'expe':None,
        'ceap_dict':df_standardized_residuals
    } 

def res1_intp(ceap_labl, resu_dict): 
    residuals_df = resu_dict['ceap_dict']
    print (residuals_df)
    res1_intp_1(ceap_labl, residuals_df)
    res1_intp_2(ceap_labl, residuals_df)
    res1_intp_3(ceap_labl, residuals_df)
def res1_intp_3(ceap_labl, residuals_df):
    index_and_columns = ceap_labl
    residuals_df.index = index_and_columns
    residuals_df.columns = index_and_columns
    residuals_dfT = residuals_df.T 
    dfZ = residuals_df - residuals_dfT
    print (residuals_df)
    print (residuals_dfT)
    print (dfZ)
    pass
# Function to create a DataFrame with "L", "R", "LL", "RR", or " " based on standardized residuals
def res1_intp_1(ceap_labl, residuals_df): 
    synthesis = {}
    for row_label in residuals_df.index:
        synthesis[row_label] = {}
        for col_label in residuals_df.columns:
            residual = residuals_df.loc[row_label, col_label]
            if residual > 2.58: # 0.01
                synthesis[row_label][col_label] = f"{row_label}_{col_label}:++:{residual}"  
            elif residual > 1.94: # 0.05
                synthesis[row_label][col_label] = f"{row_label}_{col_label}:+:{residual}"  
            elif residual < -2.58:
                synthesis[row_label][col_label] = f"{row_label}_{col_label}:--:{residual}"    
            elif residual < -1.94:
                synthesis[row_label][col_label] = f"{row_label}_{col_label}:-:{residual}"       
            else:
                synthesis[row_label][col_label] = f"{row_label}_{col_label}:.:{residual}"  
    df = pd.DataFrame(synthesis)
    df = df.T
    print (df)
def res1_intp_2(ceap_labl, residuals_df):
    synthesis = {}
    for row_label in residuals_df.index:
        synthesis[row_label] = {}
        for col_label in residuals_df.columns:
            residual = residuals_df.loc[row_label, col_label]
            if residual > 2.58: # 0.01
                synthesis[row_label][col_label] = f"++"  
            elif residual > 1.94: # 0.05
                synthesis[row_label][col_label] = f"+"  
            elif residual < -2.58:
                synthesis[row_label][col_label] = f"--"    
            elif residual < -1.94:
                synthesis[row_label][col_label] = f"-"       
            else:
                synthesis[row_label][col_label] = f"."            
    df = pd.DataFrame(synthesis)
    df = df.T
    print (df)
    #df.columns = df.index
    #df.index = ['G_' + str(i) for i in df.index]
    #df.columns = ['D_' + str(col) for col in df.columns]
    #print (df)
    return df

def res1_expl(resu_dict): 
    return None
def res1_main(ceap_labl, ceap_arra):
    print (ceap_arra)
    func_prec = res1_chck
    func_exec = res1_exec
    func_intp = res1_intp
    func_expl = res1_expl
    
    # Step 1
    # ------
    
    # Prec
    # ----
    chck_isok, chck_info = func_prec(ceap_arra)
    # Exec
    # ----
    resu_dict = func_exec(ceap_labl, ceap_arra) 
    ceap_dict = resu_dict['ceap_dict']
    # Intp
    # ----
    sign_stat = func_intp(ceap_labl, resu_dict)
    
    # Exit
    resu_dict = {
        'stat': None, # round(stat, 3),
        'pval': None, # round(pval, 3),
        'sign_stat': sign_stat,
        'sign_pval': None,
        'chck_info': None,
        'expl_stat': None,
        'expl_pval': None,
        'ceap_dict': ceap_dict,
        'fish_pval': None
    }
    return resu_dict
 
# ----
#
# ----
    
def res2_merg(df, dfM, dfF):
    # 
    dfA = df.copy()
    def colu_name(df, name):
        for col in df.columns:
            df[col] = df[col].apply(lambda x: f"{name}{x}" if x != '' else x)
    colu_name(dfA, 'A')
    colu_name(dfM, 'M')
    colu_name(dfF, 'F')
    print (f'dfA\n{dfA}\n')
    print (f'dfM\n{dfM}\n')
    print (f'dfF\n{dfF}\n')
    # Merging cell by cell into dftarget
    for i in range(dfA.shape[0]):  # Iterate over rows
        for j in range(dfA.shape[1]):  # Iterate over columns
            existing_value = dfA.iat[i, j]  # Get existing value
            new_value1 = dfM.iat[i, j] 
            new_value2 = dfF.iat[i, j]
            values_to_merge = [existing_value, new_value1, new_value2]
            merged_value = ' '.join(filter(bool, values_to_merge))
            dfA.iat[i, j] = merged_value
    return dfA

def res2_func(df_nxn):
    ceap_arra = df_nxn.to_numpy()
    ceap_labl = df_nxn.index
    resu_dict = res1_main(ceap_labl, ceap_arra)
    df = resu_dict['sign_stat']
    return df
    
def res2_main(df_nxn_A, df_nxn_M, df_nxn_F):

    df_A = res2_func(df_nxn_A)
    df_M = res2_func(df_nxn_M)
    df_F = res2_func(df_nxn_F)
    #
    print (f'dfA\n{df_A}\n')
    print (f'dfM\n{df_M}\n')
    print (f'dfF\n{df_F}\n')
    df_A.replace('.', '', inplace=True)
    df_M.replace('.', '', inplace=True)
    df_F.replace('.', '', inplace=True)
    print (f'dfA\n{df_A}\n')
    df_T = res2_merg(df_A, df_M, df_F)
    print (f'dfT\n{df_T}\n')
    df_A.replace('', '.', inplace=True)
    df_M.replace('', '.', inplace=True)
    df_F.replace('', '.', inplace=True)
    df_T.replace('', '.', inplace=True)
    print (f'dfA\n{df_A}\n')
    print (f'dfM\n{df_M}\n')
    print (f'dfF\n{df_F}\n')
    print (f'dfT\n{df_T}\n')
    pass

# ----
# Stuart_Maxwell test nxn
# ----
'''
The Stuart-Maxwell test is used to test the marginal homogeneity of a square contingency table. 
It can be used to identify significant asymmetries in the table.
The Stuart-Maxwell test is an extension of McNemar's test for larger tables.
-
Interpretation : if pval < alpha, there is evidence of asymmetry.
-
if pval < 0.05:
    print("   The Stuart-Maxwell test is significant (p < 0.05).")
    print("   This suggests that there are significant differences in the marginal distributions")
    print("   of CEAP disease levels between the left and right legs.")
    print("   In other words, the overall severity of CEAP disease is not symmetrically")
    print("   distributed between the left and right legs across all patients.")
else:
    print("   The Stuart-Maxwell test is not significant (p >= 0.05).")
    print("   This suggests that there are no significant differences in the marginal distributions")
    print("   of CEAP disease levels between the left and right legs.")
    print("   The overall severity of CEAP disease appears to be similarly")
    print("   distributed between the left and right legs across all patients.") 
'''
def stu1_chck(ceap_arra):

    # Vars
    prec_list = []
    
    # Exec
    if ceap_arra.shape[0] != ceap_arra.shape[1]:
        prec_list.append("1")
        
    # Check for sufficient sample size (typically > 10 in each off-diagonal cell)
    off_diagonal_elements = ceap_arra[~np.eye(ceap_arra.shape[0], dtype=bool)]
    if np.any(off_diagonal_elements < 10):
        prec_list.append("2") # print("Warning: Some off-diagonal elements are less than 10. Results may be unreliable.")
    
    # Check for zeros in diagonal (can cause issues in calculations)
    if np.any(np.diag(ceap_arra) == 0):
        prec_list.append("3") # print("Warning: Zeros present in the diagonal. This may affect test results.")
    
    # Exit
    if not prec_list:
        isok = True
        info = '✓'
    else:
        isok = False
        info = f"¬{','.join(prec_list)}"
    return isok, info

def stu1_exe1(ceap_labl, ceap_arra): # [openai:valid] 
    
    # Exec
    # stat
    n = ceap_arra.shape[0]
    stat = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ceap_arra[i, j] + ceap_arra[j, i] > 0:
                stat += (ceap_arra[i, j] - ceap_arra[j, i]) ** 2 / (ceap_arra[i, j] + ceap_arra[j, i])
    # dofr
    dofr = (n * (n - 1)) // 2
    # pval
    pval = 1 - chi2.cdf(stat, dofr)
    
    # Print the results
    print("Stuart-Maxwell test statistic:", stat)
    print("Degrees of freedom:", dofr)
    print("p-value:", pval)
    
    # Exit
    ceap_dict = {}
    return {
        'stat': stat,
        'pval': pval,
        'dofr': dofr,
        'ceap_dict': ceap_dict
    }
def stu1_exe2(ceap_labl, ceap_arra): # [claude:valid] 

    # Exec
    # stat
    n = ceap_arra.shape[0]
    
    # Initialize d and V
    d = np.zeros(n-1)
    V = np.zeros((n-1, n-1))
    
    # Compute d and V
    for i in range(n-1):
        d[i] = np.sum(ceap_arra[i, i+1:]) - np.sum(ceap_arra[i+1:, i])
        V[i, i] = np.sum(ceap_arra[i, :]) + np.sum(ceap_arra[:, i]) - 2 * ceap_arra[i, i]
        
        for j in range(i+1, n-1):
            V[i, j] = V[j, i] = -ceap_arra[i, j] - ceap_arra[j, i]  
    # Inverse of V
    V_inv = np.linalg.inv(V)  
    # Compute test statistic
    stat= np.dot(np.dot(d.T, V_inv), d)  
     # Degrees of freedom
    dofr = n - 1   
    # Compute p-value
    pval = 1 - stats.chi2.cdf(stat, dofr)
    
    # Print the results
    print (f'ceap_labl:{ceap_labl} ; stat:{stat} ; pval:{pval} ; dofr:{dofr}')
    
    # Exit
    ceap_dict = {}
    resu_dict = {
        'stat': stat,
        'pval': pval,
        'dofr': dofr,
        'ceap_dict': ceap_dict
    }
    return resu_dict

def stu1_exe3(ceap_labl, ceap_arra):
    n = ceap_arra.sum()
    k = ceap_arra.shape[0]  # number of categories
    
    # Calculate marginal proportions
    p1 = ceap_arra.sum(axis=1) / n
    p2 = ceap_arra.sum(axis=0) / n
    
    # Calculate differences in marginal proportions
    d = p1 - p2
    
    # Calculate the covariance matrix
    V = np.diag(p1) - np.outer(p1, p1)
    V = V[:-1, :-1]  # remove last row and column
    
    # Calculate the test statistic
    d = d[:-1]  # remove last element
    stat = chi2 = n * d.T @ np.linalg.inv(V) @ d
    
    # Degrees of freedom
    df = k - 1
    
    # P-value
    pval = 1 - stats.chi2.cdf(chi2, df)

    # Output the result of the chi-square test
    print(f"\chi2:{stat} ; pval:{pval}")
    
    if True:
        print("\nMarginal proportions:")
        categories = ceap_labl
        left_proportions = ceap_arra.sum(axis=1) / ceap_arra.sum()
        right_proportions = ceap_arra.sum(axis=0) / ceap_arra.sum()
        for cat, left, right in zip(categories, left_proportions, right_proportions):
            print(f"{cat}: Left - {left:.4f}, Right - {right:.4f}, Difference - {left-right:.4f}")
        print("\nLargest discrepancies:")
        differences = left_proportions - right_proportions
        sorted_indices = np.argsort(np.abs(differences))[::-1]
        for i in sorted_indices[:3]:
            print(f"{categories[i]}: {differences[i]:.4f}")
            
    # Exit
    ceap_dict = {}
    resu_dict = {
        'stat': stat,
        'pval': pval,
        'dofr': None,
        'ceap_dict': ceap_dict
    }
    return resu_dict

def stu1_exec(ceap_labl, ceap_arra):
    if False:
        return stu1_exe1(ceap_labl, ceap_arra)
    if False:
        return stu1_exe2(ceap_labl, ceap_arra)
    if True:
        return stu1_exe3(ceap_labl, ceap_arra)
    
def stu1_intp(resu_dict):
    pval = resu_dict['pval']
    alpha = 0.05
    if pval < alpha:
        print(f"The p-value ({pval:.4f}) is less than the significance level ({alpha}).")
        print("Reject the null hypothesis of marginal homogeneity.")
        print("There is evidence of a significant difference in the distribution of CEAP classes between left and right legs.")
    else:
        print(f"The p-value ({pval:.4f}) is greater than or equal to the significance level ({alpha}).")
        print("Fail to reject the null hypothesis of marginal homogeneity.")
        print("There is not enough evidence to conclude a significant difference in the distribution of CEAP classes between left and right legs.")  
    return None

def stu1_expl(resu_dict):
    return None
 
def stu1_main(ceap_labl, ceap_inpu):
    func_prec = stu1_chck
    func_exec = stu1_exec
    func_intp = stu1_intp
    func_expl = stu1_expl
    return comm_exec_nxn(ceap_labl, ceap_inpu, func_prec, func_exec, func_intp, func_expl)

# ----
# Bowker test (nxn tables) test 
# ----
'''
The Bowker test is used to compare the marginal distributions of a square contingency table, 
similar to the Stuart-Maxwell test, but it is specifically designed for symmetric tables. 
The Bowker test checks whether the off-diagonal elements are symmetric.
---
Chi-square statistic: 61.14718086429752
P-value: 8.549645711397602e-06
The Bowker test results suggest that there are significant differences in the CEAP classifications 
between the left and right legs. This implies that the observed frequencies of CEAP classes 
do not match between the left and right legs, and this difference is statistically significant.
-
Interpretation :
-
if pval < 0.05:
        print("   The Bowker test is significant (p < 0.05).")
        print("   This indicates that there is significant asymmetry in the CEAP disease levels")
        print("   between the left and right legs.")
        print("   It suggests that for some CEAP levels, there's a tendency for one leg")
        print("   to have a higher or lower level compared to the other leg.")
    else:
        print("   The Bowker test is not significant (p >= 0.05).")
        print("   This suggests that there is no significant asymmetry in the CEAP disease levels")
        print("   between the left and right legs.")
        print("   It indicates that the pattern of CEAP levels is generally symmetric")
        print("   when comparing left and right legs across patients.")
---
To perform subgroup analysis and understand where the most significant differences lie, you can follow a systematic approach. Here are some suggested steps and methods to conduct this analysis:

1. Identify Subgroups of Interest
First, identify the specific subgroups or classes you want to analyze. 
In your case, these could be the individual CEAP classes (C0 through C6).
2. Extract Subgroup Data
Extract the data for each subgroup from the contingency table. 
This will allow you to perform focused analyses on each subgroup.
3. Perform Statistical Tests
For each subgroup, perform statistical tests to compare the distributions between the left and right legs. 
You can use tests like the chi-square test, Fisher's exact test, or other appropriate tests depending on the nature of your data.
4. Visualize the Results
Create visualizations to help interpret the results. Heatmaps, bar plots, or other types of plots 
can be useful for comparing the distributions.
5. Interpret the Results
Interpret the results of the statistical tests and visualizations to understand where the most significant differences lie.
---
Interpretation
Chi-Square Statistic: A large chi-square statistic indicates a significant difference between the observed and expected frequencies.
P-Value: A small p-value (typically less than 0.05) indicates that the difference is statistically significant.
By performing this subgroup analysis, you can identify which CEAP classes have the most significant differences between the left and right legs. This information can help you 
understand the nature of the discrepancies and guide further investigation or clinical interpretation.
'''
'''
NAC6
----
The result you obtained from the Bowker's test shows a chi-square statistic 
(`stat = 87.67`) and a very small p-value (`pval ≈ 4.57e-08`). 
This suggests strong evidence against the null hypothesis of marginal homogeneity, 
meaning there is likely a significant difference between the distribution of disease severity (CEAP classes) 
in the left leg and the right leg.

### Breakdown of the Code and Test:

- **Bowker's Test**: This is a generalization of McNemar's test for an \( n \times n \) contingency table. It tests the symmetry of the table, which in this case would mean testing whether the distribution of CEAP classes in the left leg is the same as in the right leg.
- **Null Hypothesis**: The disease severity (CEAP class) distribution is symmetric between the left and right legs, meaning that for any disease severity \( i \), the number of patients with severity \( i \) in the left leg and severity \( j \) in the right leg should equal the number with severity \( j \) in the left leg and severity \( i \) in the right leg.
- **Test Statistic**: The test compares the off-diagonal elements of the contingency table. The difference between the pairs of off-diagonal elements is squared, and the result is divided by the sum of the pair (or adjusted with Yates correction if specified). The chi-square statistic is the sum of these values.
- **Degrees of Freedom**: For an \( n \times n \) contingency table, the degrees of freedom for Bowker's test is \( \frac{n(n-1)}{2} \), where \( n \) is the number of categories. In this case, with 8 categories (NA, C0, C1, C2, C3, C4, C5, C6), the degrees of freedom are:
  \[
  \text{dof} = \frac{8(8-1)}{2} = 28
  \]
  
### Interpretation of Results:
- **Test Statistic**: The computed chi-square test statistic of 87.67 is quite large, indicating significant asymmetry in the contingency table.
- **p-value**: The p-value is extremely small (4.57e-08), which indicates that there is a highly significant difference between the left and right legs in terms of disease severity distribution. This means the null hypothesis of symmetry is rejected with strong confidence, suggesting that disease severity is distributed differently between the legs.

### Conclusion:
The Bowker's test indicates that the distribution of CEAP classes between the left and right legs is not symmetric. This could suggest that one leg tends to have a different distribution of disease severity than the other, which may be important for understanding the progression or treatment of venous disease in your study.

C0C6
----
stat:70.81829197540864 pval:2.5963738958800775e-07
C3C6
----
stat:9.48982683982684 pval:0.14784637552584878
'''
def bow1_chck(ceap_arra):

    # Vars
    prec_list = []
    
    # Exec
    if ceap_arra.shape[0] != ceap_arra.shape[1]:
        prec_list.append("1")
        
    # Check for sufficient sample size (typically > 10 in each off-diagonal cell)
    off_diagonal_elements = ceap_arra[~np.eye(ceap_arra.shape[0], dtype=bool)]
    if np.any(off_diagonal_elements < 10):
        prec_list.append("2") # print("Warning: Some off-diagonal elements are less than 10. Results may be unreliable.")
    
    # Check for zeros in diagonal (can cause issues in calculations)
    if np.any(np.diag(ceap_arra) == 0):
        prec_list.append("3") # print("Warning: Zeros present in the diagonal. This may affect test results.")
    
    # Exit
    if not prec_list:
        isok = True
        info = '✓'
    else:
        isok = False
        info = f"¬{','.join(prec_list)}"
    return isok, info

def bow1_exec_glob(ceap_labl, ceap_arra, yates_correction=False):
    
    # Exec
    n = ceap_arra.shape[0]
    if n != ceap_arra.shape[1]:
        raise ValueError("The contingency table must be square.")
    # Extract the off-diagonal elements
    off_diagonal = ceap_arra[np.triu_indices_from(ceap_arra, k=1)]
    off_diagonal_transposed = ceap_arra[np.tril_indices_from(ceap_arra, k=-1)]
    print (off_diagonal)
    print (off_diagonal_transposed)
    # Calculate the chi-square statistic
    numerator = (off_diagonal - off_diagonal_transposed) ** 2
    if yates_correction:
        numerator = (np.abs(off_diagonal - off_diagonal_transposed) - 0.5) ** 2
    denominator = off_diagonal + off_diagonal_transposed
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2_stat = np.sum(np.where(denominator != 0, numerator / denominator, 0))
    # Degrees of freedom
    dofr = n * (n - 1) // 2
    # Calculate the p-value
    pval = 1 - chi2.cdf(chi2_stat, dofr)
    
    print (f'ceap_labl:{ceap_labl} ; stat:{chi2_stat} ; pval:{pval} ; dofr:{dofr}')
    # Exit
    return chi2_stat, pval

def bow1_exec_deta(ceap_labl, ceap_arra):

    # Function to perform chi-square test on a 2x2 contingency table
    def chi_square_test(table):
        chi2, p, dof, ex = chi2_contingency(table)
        return chi2, p
    # Perform subgroup analysis
    ceap_dict = {}
    ceap_trac = {}
    for i in range(ceap_arra.shape[0]):
        # Extract counts for the current CEAP class
        left_leg_counts = ceap_arra[i, i]
        right_leg_counts = ceap_arra[i, :].sum() - left_leg_counts
        other_counts = ceap_arra[:, i].sum() - left_leg_counts
        total_counts = ceap_arra.sum() - left_leg_counts - right_leg_counts - other_counts
        # Create a 2x2 contingency table
        table = np.array([[left_leg_counts, right_leg_counts], [other_counts, total_counts]])
        # Perform chi-square test
        chi2, p = chi_square_test(table)
        ceap_dict[ceap_labl[i]] = {'chi2_deta': chi2, 'pval_deta': p} # ceap_dict[df.index[i]] = {'chi2_deta': chi2, 'pval_deta': p}
        ceap_trac[ceap_labl[i]] = {'stat:' : round(chi2,3), 'pval': round(p,3)}
        # Print results
        print(f"{ceap_labl[i]}:{ceap_trac[ceap_labl[i]]}")
            
    # exit
    return ceap_dict

def bow1_exec(ceap_labl, ceap_arra):
    chi2_stat, pval = bow1_exec_glob(ceap_labl, ceap_arra)
    ceap_dict = bow1_exec_deta(ceap_labl, ceap_arra)
    print (ceap_dict)
    # Exit
    resu_dict = {
        'stat': chi2_stat,
        'pval': pval,
        'dofr': None,
        'ceap_dict': ceap_dict
    }
    return resu_dict

def bow1_intp(resu_dict):
    return None

def bow1_expl(resu_dict): 
    return None
def bow1_main(ceap_labl, ceap_inpu):
    func_prec = bow1_chck
    func_exec = bow1_exec
    func_intp = bow1_intp
    func_expl = bow1_expl
    return comm_exec_nxn(ceap_labl, ceap_inpu, func_prec, func_exec, func_intp, func_expl)

# ----
# Permutation test nxn
# ----
'''
if pval < 0.05:
    print("   The permutation test indicates significant asymmetry in CEAP disease levels")
    print("   between left and right legs (p < 0.05).")
    print("   This suggests that the observed pattern of CEAP levels is unlikely")
    print("   to occur by chance if there were true symmetry between legs.")
else:
    print("   The permutation test does not provide evidence of significant asymmetry")
    print("   in CEAP disease levels between left and right legs (p >= 0.05).")
    print("   This suggests that the observed pattern of CEAP levels could plausibly")
    print("   occur even if there were true symmetry between legs.")
'''
def perm_chck(cont_tabl):
    print (cont_tabl)
    print (type(cont_tabl))
    # Vars
    prec_list = []
    
    # Exec
    
    # Exit
    if not prec_list:
        isok = True
        info = '✓'
    else:
        isok = False
        info = f"¬{','.join(prec_list)}"
    return isok, info

def perm_exec(ceap_labl, ceap_arra, num_permutations=10000): # [claude:valid] 10000 is ideal

    '''
    contingency_matrix = np.array([
        [0, 0, 1, 8, 2, 2, 14],
        [0, 1, 0, 3, 2, 1, 0],
        [1, 1, 10, 14, 3, 3, 18],
        [9, 0, 15, 80, 16, 3, 14],
        [2, 1, 3, 17, 23, 7, 5],
        [5, 1, 5, 4, 2, 6, 8],
        [29, 3, 33, 9, 5, 7, 21]
    ])
    '''
    # Perform a permutation test for symmetry in the contingency table.
    n = ceap_arra.shape[0]
    stat = observed_statistic = np.sum((ceap_arra - ceap_arra.T)**2)
    
    count = 0
    for _ in range(num_permutations):
        # Create a permuted matrix
        permuted = np.zeros_like(ceap_arra)
        for i in range(n):
            for j in range(i+1, n):
                total = ceap_arra[i, j] + ceap_arra[j, i]
                split = np.random.binomial(total, 0.5)
                permuted[i, j] = split
                permuted[j, i] = total - split
        
        permuted_statistic = np.sum((permuted - permuted.T)**2)
        if permuted_statistic >= observed_statistic:
            count += 1
    
    pval = count / num_permutations
    
    print (f'stat:{stat} pval:{pval}')
    # Exit
    ceap_dict = {}
    resu_dict = {
        'stat': stat,
        'pval': pval,
        'dofr': None,
        'ceap_dict': ceap_dict
    }
    return resu_dict

def perm_intp(resu_dict):
    return None

def perm_expl(resu_dict):
    return None
 
def perm_main(ceap_labl, ceap_inpu):
    func_prec = perm_chck
    func_exec = perm_exec
    func_intp = perm_intp
    func_expl = perm_expl
    return comm_exec_nxn(ceap_labl, ceap_inpu, func_prec, func_exec, func_intp, func_expl)

# ----
# Log linear test nxn
# ----
'''
if pvalue < 0.05:
        print("1. The test is statistically significant (p < 0.05).")
        print("   This suggests strong evidence against the null hypothesis of symmetry.")
        print("   In the context of CEAP disease levels, this indicates significant")
        print("   asymmetry between left and right leg disease patterns.")
    else:
        print("1. The test is not statistically significant (p >= 0.05).")
        print("   This suggests insufficient evidence to reject the null hypothesis of symmetry.")
        print("   In the context of CEAP disease levels, this indicates no significant")
        print("   asymmetry detected between left and right leg disease patterns.")
'''
def loli_chck(ceap_arra):

    # Vars
    prec_list = []
    
    # Exec
    
    # Exit
    if not prec_list:
        isok = True
        info = '✓'
    else:
        isok = False
        info = f"¬{','.join(prec_list)}"
    return isok, info

def loli_exec(ceap_labl, ceap_arra): # [claude:valid]
    
    # Vars
    ceap_dict = {}
    '''
    contingency_matrix = np.array([
        [0, 0, 1, 8, 2, 2, 14],
        [0, 1, 0, 3, 2, 1, 0],
        [1, 1, 10, 14, 3, 3, 18],
        [9, 0, 15, 80, 16, 3, 14],
        [2, 1, 3, 17, 23, 7, 5],
        [5, 1, 5, 4, 2, 6, 8],
        [29, 3, 33, 9, 5, 7, 21]
    ])
    '''
    # Fit a log-linear model to test for symmetry in the contingency table.
    n = ceap_arra.shape[0]
    data = []
    for i in range(n):
        for j in range(n):
            data.append({
                'mbrG': i,
                'mbrR': j,
                'count': ceap_arra[i, j]
            })
    df = pd.DataFrame(data)
    
    # Fit the symmetry model
    formula = 'count ~ C(mbrG) + C(mbrR)'
    model = GLM.from_formula(formula, data=df, family=families.Poisson())
    results = model.fit()
    # Fit the quasi-symmetry model
    formula_quasi = 'count ~ C(mbrG) + C(mbrR) + C(mbrG):C(mbrR)'
    model_quasi = GLM.from_formula(formula_quasi, data=df, family=families.Poisson())
    results_quasi = model_quasi.fit()
    # Likelihood ratio test
    stat = lr_statistic = -2 * (results.llf - results_quasi.llf)
    dofr = lr_df = results_quasi.df_model - results.df_model
    pval = lr_pvalue = 1 - stats.chi2.cdf(lr_statistic, lr_df)
    
    print (f'stat:{stat} dofr:{dofr} pval:{pval}')
    print (f'symmetry_aic      :{results.aic}')
    print (f'quasi_symmetry_aic:{results_quasi.aic}')

    # Exit
    resu_dict = {
        'stat': stat,
        'dofr': dofr,
        'pval': pval,
        'ceap_dict': ceap_dict,
        'symmetry_aic': results.aic,
        'quasi_symmetry_aic': results_quasi.aic
    }
    return resu_dict

def loli_intp(resu_dict):
    symmetry_aic = resu_dict['symmetry_aic']
    quasi_symmetry_aic = resu_dict['quasi_symmetry_aic']
    pval = resu_dict['pval'] 
    aic_difference = symmetry_aic - quasi_symmetry_aic
    
    # Modl    
    if pval < 0.05 and aic_difference > 2:
        modl = "[asym++]"
    elif pval < 0.05 and aic_difference <= 2:
        modl = "[asym,symm]"
    else:
        modl = "[symm++]"
    # Aic
    if aic_difference > 10:
        aic = "[modl_quas_symm : mix asym++,symm+]"
    elif aic_difference > 2:
        aic = "[modl_quas_symm : mix asym+,symm+]"
    else:
        aic = "[modl_symm]"
    # Aic
    aic_form = f"{aic_difference:.3e}" if aic_difference < 0.001 else f"{aic_difference:.3f}" 
    
    intp = f'pval:{pval} ; aic_dif:{aic_form} ; modl: {modl} & aic: {aic}'
    return intp

def loli_expl(resu_dict):
    symmetry_aic = resu_dict['symmetry_aic']
    quasi_symmetry_aic = resu_dict['quasi_symmetry_aic']
    pvalue = resu_dict['pval'] 
    aic_difference = symmetry_aic - quasi_symmetry_aic
    
    print(f"\n2. AIC Comparison:")
    print(f"   The difference in AIC (Symmetry - Quasi-Symmetry) is {aic_difference:.4f}")
    if aic_difference > 10:
        print("   This large difference (> 10) strongly favors the quasi-symmetry model.")
        print("   It suggests that while there is asymmetry, it's not complete - some")
        print("   symmetric patterns exist alongside asymmetric ones in the CEAP data.")
    elif aic_difference > 2:
        print("   This moderate difference (2-10) suggests some support for the quasi-symmetry model.")
        print("   It indicates a mix of symmetric and asymmetric patterns in the CEAP data,")
        print("   but the evidence for asymmetry is not as strong.")
    else:
        print("   The small difference (<= 2) suggests little improvement of the quasi-symmetry model.")
        print("   This indicates that the simpler symmetry model might be adequate to describe")
        print("   the CEAP disease level patterns between left and right legs.")
    
    print("\n3. Clinical Implications:")
    if pvalue < 0.05 and aic_difference > 2:
        print("   The results suggest significant asymmetry in CEAP disease levels between")
        print("   left and right legs. This could indicate:")
        print("   - Differential progression of venous disease between legs")
        print("   - Potential unilateral risk factors affecting one leg more than the other")
        print("   - Need for leg-specific treatment approaches in some patients")
    elif pvalue < 0.05 and aic_difference <= 2:
        print("   While there's statistical evidence of asymmetry, the practical significance")
        print("   may be limited. Consider:")
        print("   - Whether the detected asymmetry is clinically meaningful")
        print("   - If certain CEAP levels or patient subgroups drive this asymmetry")
    else:
        print("   The results suggest largely symmetric CEAP disease patterns between legs.")
        print("   This could imply:")
        print("   - Systemic factors affecting both legs similarly")
        print("   - Potentially similar treatment approaches for both legs in most patients")
    
    print("\n4. Further Considerations:")
    print("   - Examine raw data and contingency tables to identify specific asymmetric patterns")
    print("   - Consider patient-specific factors that might explain any observed asymmetry")
    print("   - Assess if the results align with clinical observations and existing literature")
    print("   - Consider additional analyses to explore specific types of asymmetry if needed")
    
    return None
 
def loli_main(ceap_labl, ceap_inpu):
    func_prec = loli_chck
    func_exec = loli_exec
    func_intp = loli_intp
    func_expl = loli_expl
    return comm_exec_nxn(ceap_labl, ceap_inpu, func_prec, func_exec, func_intp, func_expl)

# ----
# Cohen's Kappa test nxn
# ----
'''
'''
def cok1_chck(cont_tabl):

    # Vars
    prec_list = []
    
    # Exec
    
    # Exit
    if not prec_list:
        isok = True
        info = '✓'
    else:
        isok = False
        info = f"¬{','.join(prec_list)}"
    return isok, info

def cok1_exec(ceap_labl, ceap_arra, num_permutations=10000): # [claude:valid] 10000 is ideal

    '''
    contingency_matrix = np.array([
        [0, 0, 1, 8, 2, 2, 14],
        [0, 1, 0, 3, 2, 1, 0],
        [1, 1, 10, 14, 3, 3, 18],
        [9, 0, 15, 80, 16, 3, 14],
        [2, 1, 3, 17, 23, 7, 5],
        [5, 1, 5, 4, 2, 6, 8],
        [29, 3, 33, 9, 5, 7, 21]
    ])
    '''
    ceap_labl_shap = len(ceap_labl)  # Number of classes (should match matrix dimensions)
    if ceap_arra.shape[0] != ceap_labl_shap or ceap_arra.shape[1] != ceap_labl_shap:
        raise ValueError("Matrix dimensions do not match the length of ceap_labl")
    
    # Determine observed kappas ->Flatten the matrix for kappa calculation
    left_leg = np.repeat(np.arange(ceap_labl_shap), np.sum(ceap_arra, axis=1))
    right_leg = np.concatenate([np.repeat(np.arange(ceap_labl_shap), ceap_arra[i, :]) for i in range(ceap_labl_shap)])
    # Check lengths of left_leg and right_leg
    if len(left_leg) != len(right_leg):
        raise ValueError(f"Length mismatch: left_leg ({len(left_leg)}) and right_leg ({len(right_leg)})")
    observed_kappa = stat = cohen_kappa_score(left_leg, right_leg)
    # Print intermediate results for verification
    #print("Left Leg:", left_leg)
    #print("Right Leg:", right_leg)
    #print(f"Observed Kappa: {observed_kappa}")
    
    # Determine permuted kappas -> p-value to help determine if the observed agreement is statistically significant.
    permuted_kappas = []
    for _ in range(num_permutations):
        permuted_right_leg = np.random.permutation(right_leg)
        permuted_kappa = cohen_kappa_score(left_leg, permuted_right_leg)
        permuted_kappas.append(permuted_kappa)
    permuted_kappas = np.array(permuted_kappas)
    
    # pval
    pval = np.mean(np.abs(permuted_kappas) >= np.abs(observed_kappa))
    
    print(f'stat:{observed_kappa} pval:{pval}')
 
    # Exit
    ceap_dict = {}
    resu_dict = {
        'stat': stat,
        'pval': pval,
        'dofr': None,
        'ceap_dict': ceap_dict
    }
    return resu_dict

def cok1_intp(resu_dict):
    
    kapp = resu_dict['stat']
       
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

def cok1_expl(resu_dict):
    
    kapp = resu_dict['stat']

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

 
def cok1_main(ceap_labl, ceap_inpu):
    func_prec = cok1_chck
    func_exec = cok1_exec
    func_intp = cok1_intp
    func_expl = cok1_expl
    return comm_exec_nxn(ceap_labl, ceap_inpu, func_prec, func_exec, func_intp, func_expl)

# ----
# Call
# ----
'''
df.type:<class 'pandas.core.frame.DataFrame'>
    NA  C0  C1  C2  C3  C4  C5  C6
NA   0   0   0   2  18   5   0  22
C0   0   0   0   1   8   2   2  14
C1   0   0   1   0   3   2   1   0
C2   3   1   1  10  14   3   3  18
C3  20   9   0  15  80  16   3  14
C4   9   2   1   3  17  23   7   5
C5   7   5   1   5   4   2   6   8
C6  39  29   3  33   9   5   7  21
:Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], dtype='object')
:Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], dtype='object')
:542
---
    C0  C1  C2  C3  C4  C5  C6
C0   0   0   1   8   2   2  14
C1   0   1   0   3   2   1   0
C2   1   1  10  14   3   3  18
C3   9   0  15  80  16   3  14
C4   2   1   3  17  23   7   5
C5   5   1   5   4   2   6   8
C6  29   3  33   9   5   7  21
:Index(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], dtype='object')
:Index(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], dtype='object')
:417
'''
def call(df_nxn):
    
    cont_arra_arra = df_nxn.to_numpy()
    resu_dict = {}

    if True:
        
        resu_dict["fre1"] = fre1_main(df_nxn)
        resu_dict["fre2"] = fre2_main(df_nxn)  
        resu_dict["fre3"] = fre3_main(df_nxn)
        resu_dict["fre4"] = fre4_main(df_nxn)
        resu_dict["homo"] = homo_main(df_nxn)
        resu_dict["symm"] = symm_main(df_nxn)
        resu_dict["cram"] = cram_main(df_nxn)
        resu_dict["keng"] = keng_main(df_nxn.index, cont_arra_arra)
        resu_dict["spea"] = spea_main(df_nxn.index, cont_arra_arra)
        resu_dict["pear"] = pear_main(df_nxn.index, cont_arra_arra)
        
        resu_dict["chi1"] = chi1_main(df_nxn.index, cont_arra_arra)
        resu_dict["res1"] = res1_main(df_nxn.index, cont_arra_arra)
        resu_dict["stu1"] = stu1_main(df_nxn.index, cont_arra_arra)
        resu_dict["bow1"] = bow1_main(df_nxn.index, cont_arra_arra)
        resu_dict["perm"] = perm_main(df_nxn.index, cont_arra_arra)
        resu_dict["loli"] = loli_main(df_nxn.index, cont_arra_arra)
        resu_dict["cok1"] = cok1_main(df_nxn.index, cont_arra_arra)
        
    print (resu_dict)    
    return resu_dict

# ----
# Orch
# ----
def desc(df):
    return ''
def orch(what, repo, df_nxn):
    # Trac
    print(f"df_inpu:{type(df_nxn)}\n{df_nxn}\n:{df_nxn.index}")
    plug_inpu(what, repo, df_nxn, desc(df_nxn))
    # Exec
    resu_dict = call(df_nxn)
    # Example
    if False:
        resu_dict = {
            'nema': {'xyz': 'abc', 'ceap_dict': {'C0': {'stat': 0.25, 'pval': 0.02}, 'C1': {'stat': 0.10, 'pval': 1.0} } },
            'bowk': {'xyz': 'def', 'ceap_dict': {'C0': {'stat': 0.35, 'pval': 0.32}, 'C1': {'stat': 0.11, 'pval': 1.0} } }
        }
    # Flatten the nested dictionary into a list of dictionaries
    flat_data_list = []
    for key, value in resu_dict.items():
        if key == 'res1':
            row = {
                    'key': key,
                    'resi_tabl' : value['ceap_dict']
                } 
            flat_data_list.append(row)
        else:
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

        # Etude des tests statistiques
        repo = pgrm_init()
        #
        def exec(df_nxn, df_nx2, sexe):
            what, df_nxn_NAC6, df_nx2_NAC6 = ke43_ceap_unbi_deta_NAC6(df_nxn, df_nx2, sexe)
            orch(what, repo, df_nxn_NAC6)
            if True:
                what, df_nxn_C0C0, df_nx2_C0C6 = ke43_ceap_unbi_deta_C0C6(df_nxn, df_nx2, sexe)
                orch(what, repo, df_nxn_C0C0)
                what, df_nxn_NAC2, df_nx2_NAC2 = ke43_ceap_unbi_deta_NAC2(df_nxn, df_nx2, sexe)
                orch(what, repo, df_nxn_NAC2)
                what, df_nxn_C0C2, df_nx2_C0C2 = ke43_ceap_unbi_deta_C0C2(df_nxn, df_nx2, sexe)
                orch(what, repo, df_nxn_C0C2)
                what, df_nxn_C3C6, df_nx2_C3C6 = ke43_ceap_unbi_deta_C3C6(df_nxn, df_nx2, sexe)
                orch(what, repo, df_nxn_C3C6)
        #
        exec(df_A_nxn, df_A_nx2, 'A') # sexe M+F
        # exec(df_M_nxn, df_M_nx2, 'M') # sexe M
        # exec(df_F_nxn, df_F_nx2, 'F') # sexe F
        #
        pgrm_fini(repo, os.path.splitext(os.path.basename(__file__)))
        pass