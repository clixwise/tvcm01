
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
# ---
# chi2
# ----
'''
NAC6
----
The new table includes an additional category 'NA' to represent cases where there is no disease in the leg. 
This table now compares the CEAP disease levels in the left leg (rows) versus the right leg (columns), 
including the 'NA' category, for the same 362 patients.

The results of the chi-square test with the 'NA' category included are:
- Chi-square statistic (`stat`): 286.8287045463585
- p-value (`pval`): 2.3580464104728916e-35

### Interpretation of the Results
1. **Chi-square Statistic**: The chi-square statistic has increased to 286.8287045463585, 
indicating an even greater deviation from the expected frequencies under the assumption of independence when the 'NA' category is included.
2. **p-value**: The p-value is extremely small (2.3580464104728916e-35), which is even smaller than the previous p-value. 
This further strengthens the evidence against the null hypothesis of independence.

### Conclusion
Given the extremely small p-value, we can confidently reject the null hypothesis of independence. 
This means that there is a significant association between the CEAP disease levels in the left leg and the right leg, 
even when the 'NA' category is included. The inclusion of the 'NA' category has not only maintained 
but also strengthened the evidence of the association between the CEAP disease levels in the left and right legs for the 362 patients in this dataset.
In summary, the CEAP disease levels in the left leg are not independent of the CEAP disease levels in the right leg, 
and this association is statistically significant.
'''
'''
C0C6
----
The table you provided is a contingency table that compares the CEAP disease levels in the left leg (rows) 
versus the right leg (columns) for 362 patients. Each cell in the table represents the count of patients 
with a specific combination of CEAP disease levels in the left and right legs.
The chi-square test is used to determine if there is a significant association between the CEAP disease levels 
in the left leg and the right leg. The results of the chi-square test are:
- Chi-square statistic (`stat`): 217.74949340196423
- p-value (`pval`): 7.342945128757637e-28

### Interpretation of the Results

1. **Chi-square Statistic**: The chi-square statistic is a measure of how much the observed frequencies deviate 
from the expected frequencies under the assumption of independence. A large chi-square statistic indicates 
a significant deviation from independence.
2. **p-value**: The p-value is the probability of observing a chi-square statistic as extreme as, or more extreme than, 
the one calculated under the null hypothesis of independence. A very small p-value (less than 0.05) indicates 
strong evidence against the null hypothesis, suggesting that the CEAP disease levels in the left and right legs are not independent.

### Conclusion
Given the very small p-value (7.342945128757637e-28), we can reject the null hypothesis of independence. 
This means that there is a significant association between the CEAP disease levels in the left leg and the right leg. 
In other words, the CEAP disease levels in the left leg are not independent of the CEAP disease levels 
in the right leg for the 362 patients in this dataset.
'''
'''
C3C6
----
86.91626201704861 
6.753681026326215e-15
'''
def chi1_chck_WAIT_PART_CLEAN(df_obsv):
        
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
    return isok, resu # , df_clean

# ----
# resi
# ----
        
# ----
# Play
# ----
def res1_play_comp():
    # Observed contingency table
    observed = np.array([
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
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    total = observed.sum()
    # Compute expected values
    expected = np.outer(row_totals, col_totals) / total
    # Compute standardized residuals
    standardized_residuals = (observed - expected) / np.sqrt(expected)
    # Create a dataframe for easier interpretation
    index_labels = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    df_standardized_residuals = pd.DataFrame(standardized_residuals, index=index_labels, columns=index_labels)
    print (df_standardized_residuals)
    return df_standardized_residuals

# Function to create a DataFrame with "L", "R", "LL", "RR", or " " based on standardized residuals
def res1_play_intp(residuals_df):
    synthesis = {}
    for row_label in residuals_df.index:
        synthesis[row_label] = {}
        for col_label in residuals_df.columns:
            residual = residuals_df.loc[row_label, col_label]
            if residual > 2.58: # 0.01
                synthesis[row_label][col_label] = "LL"  # Strong overrepresentation of left leg
            elif residual > 1.94: # 0.05
                synthesis[row_label][col_label] = "L"   # Moderate overrepresentation of left leg
            elif residual < -2.58:
                synthesis[row_label][col_label] = "RR"  # Strong overrepresentation of right leg
            elif residual < -1.94:
                synthesis[row_label][col_label] = "R"   # Moderate overrepresentation of right leg
            else:
                synthesis[row_label][col_label] = "_"   # No significant difference
    return pd.DataFrame(synthesis)

def res1_play_main():
    df = res1_play_comp()
    print (df)
    # Example: Assuming df_standardized_residuals is the dataframe containing standardized residuals
    df_synthesized = res1_play_intp(df)
    print(df_synthesized)
    pass

def res2_main():
    pass # stub to replace
def res2_play():   
    # Sample DataFrames with string content
    dfA = pd.DataFrame({'A': ['existing1', 'existing2', ''], 'B': ['existing3', '', 'existing5']}, index=[0, 1, 2])
    dfM = pd.DataFrame({'A': ['', '', 'banana'], 'B': ['cat', 'dog', '']}, index=[0, 1, 2])
    dfF = pd.DataFrame({'A': ['orange', '', 'grape'], 'B': ['fish', 'hamster', '']}, index=[0, 1, 2])
    res2_main(dfA, dfM, dfF)
# ----
# Stuart_maxwell test nxn
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
'''
NAC6
----
The results you obtained indicate the following:
- **Stuart-Maxwell test statistic**: 36.147
- **Degrees of freedom**: 28
- **p-value**: 0.1389
### Interpretation
1. **Test Statistic**: The Stuart-Maxwell test statistic is 36.147. 
This value indicates the degree of discrepancy between the row and column marginal distributions.
2. **Degrees of Freedom**: The degrees of freedom for the test are 28, which is calculated as 
\( \frac{n(n-1)}{2} \), where \( n \) is the number of rows (or columns) in the table.
3. **p-value**: The p-value is 0.1389. This value is greater than the common significance level of 0.05.
### Conclusion
A p-value greater than 0.05 indicates that there is not enough evidence to reject the null hypothesis of marginal homogeneity. 
In other words, the CEAP disease levels in the left leg are not significantly different 
from the CEAP disease levels in the right leg when considering the 'NA' category and applying the Yates correction.
This suggests that the distribution of CEAP disease levels in the left leg is not statistically different from 
the distribution in the right leg for the 362 patients in your dataset.
C0C6
----
Stuart-Maxwell test statistic: 22.961237891841062
Degrees of freedom: 21
p-value: 0.3460486194277286
C3C6
----
Stuart-Maxwell test statistic: 4.1045611393437476
Degrees of freedom: 6
p-value: 0.6625287858064197

****
To test for marginal homogeneity using the Stuart-Maxwell test, the null hypothesis (H0) is:
H0: The marginal distributions of the row and column variables are identical
In other words, the null hypothesis states that the marginal probabilities for each row category are equal to the corresponding marginal probabilities for each column category. This implies that the row and column variables have the same underlying probability distribution.
Mathematically, for a square contingency table with k categories, the null hypothesis can be expressed as:
H0: P(row = i) = P(col = i), for all i = 1, 2, ..., k
Where P(row = i) represents the marginal probability of the row variable being in category i, and P(col = i) represents the marginal probability of the column variable being in category i.
If the null hypothesis is rejected, it suggests that the row and column variables have different underlying probability distributions, i.e., the marginal distributions are not homogeneous.

Citations:
https://statkat.com/stat-tests/marginal-homogeneity-stuart-maxwell-test.php 
[1] https://www.richardsonhealthcare.com/ceap-classification/
[2] https://pubmed.ncbi.nlm.nih.gov/32113854/
[3] https://pubmed.ncbi.nlm.nih.gov/23488310/
[4] https://www.jvsvenous.org/article/S2213-333X%2820%2930063-9/fulltext
[5] https://www.sigvaris.com/en-gb/expertise/basics/ceap-classification
[6] https://emedicine.medscape.com/article/1085412-guidelines
[7] https://www.dynamed.com/condition/venous-insufficiency
'''

# ----
# Stuart_maxwell play
# ----        
def stu1_play_mai1():

    # Your contingency table
    data = {
        'NA': [0, 0, 0, 3, 20, 9, 7, 39],
        'C0': [0, 0, 0, 1, 9, 2, 5, 29],
        'C1': [0, 0, 1, 1, 0, 1, 1, 3],
        'C2': [2, 1, 0, 10, 15, 3, 5, 33],
        'C3': [18, 8, 3, 14, 80, 17, 4, 9],
        'C4': [5, 2, 2, 3, 16, 23, 2, 5],
        'C5': [0, 2, 1, 3, 3, 7, 6, 7],
        'C6': [22, 14, 0, 18, 14, 5, 8, 21]
    }
    cont_tabl = pd.DataFrame(data, index=['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'])
    # Convert the DataFrame to a NumPy array
    table = cont_tabl.to_numpy()
    print (table)
    # Calculate the Stuart-Maxwell test statistic
    n = table.shape[0]
    chi2_stat = 0
    for i in range(n):
        for j in range(i + 1, n):
            if table[i, j] + table[j, i] > 0:
                chi2_stat += (table[i, j] - table[j, i]) ** 2 / (table[i, j] + table[j, i])
    # Degrees of freedom
    df = (n * (n - 1)) // 2
    # p-value
    p_value = 1 - chi2.cdf(chi2_stat, df)
    # Print the results
    print("Stuart-Maxwell test statistic:", chi2_stat)
    print("Degrees of freedom:", df)
    print("p-value:", p_value)
    pass

def stu1_play_mai2_test(data):
    n = data.sum()
    k = data.shape[0]  # number of categories
    
    # Calculate marginal proportions
    p1 = data.sum(axis=1) / n
    p2 = data.sum(axis=0) / n
    
    # Calculate differences in marginal proportions
    d = p1 - p2
    
    # Calculate the covariance matrix
    V = np.diag(p1) - np.outer(p1, p1)
    V = V[:-1, :-1]  # remove last row and column
    
    # Calculate the test statistic
    d = d[:-1]  # remove last element
    chi2 = n * d.T @ np.linalg.inv(V) @ d
    
    # Degrees of freedom
    df = k - 1
    
    # P-value
    p_value = 1 - stats.chi2.cdf(chi2, df)
    
    return chi2, df, p_value

def stu1_play_mai2_intp(data, chi2, df, p_value, alpha=0.05):
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {df}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < alpha:
        print(f"The p-value ({p_value:.4f}) is less than the significance level ({alpha}).")
        print("Reject the null hypothesis of marginal homogeneity.")
        print("There is evidence of a significant difference in the distribution of CEAP classes between left and right legs.")
    else:
        print(f"The p-value ({p_value:.4f}) is greater than or equal to the significance level ({alpha}).")
        print("Fail to reject the null hypothesis of marginal homogeneity.")
        print("There is not enough evidence to conclude a significant difference in the distribution of CEAP classes between left and right legs.")
    
    print("\nMarginal proportions:")
    categories = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    left_proportions = data.sum(axis=1) / data.sum()
    right_proportions = data.sum(axis=0) / data.sum()
    
    for cat, left, right in zip(categories, left_proportions, right_proportions):
        print(f"{cat}: Left - {left:.4f}, Right - {right:.4f}, Difference - {left-right:.4f}")
    
    print("\nLargest discrepancies:")
    differences = left_proportions - right_proportions
    sorted_indices = np.argsort(np.abs(differences))[::-1]
    for i in sorted_indices[:3]:
        print(f"{categories[i]}: {differences[i]:.4f}")

def stu1_play_mai2():
    data = np.array([
        [0, 0, 0, 2, 18, 5, 0, 22],
        [0, 0, 0, 1, 8, 2, 2, 14],
        [0, 0, 1, 0, 3, 2, 1, 0],
        [3, 1, 1, 10, 14, 3, 3, 18],
        [20, 9, 0, 15, 80, 16, 3, 14],
        [9, 2, 1, 3, 17, 23, 7, 5],
        [7, 5, 1, 5, 4, 2, 6, 8],
        [39, 29, 3, 33, 9, 5, 7, 21]
    ])

    # Perform Stuart-Maxwell test
    chi2, df, p_value = stu1_play_mai2_test(data)

    # Interpret the results
    stu1_play_mai2_intp(data, chi2, df, p_value)
    pass  
# ----
# Bowker
# ----

def bow1_play_func(contingency_table):
    """
    Perform the Bowker test to compare the marginal distributions of a symmetric contingency table.

    Parameters:
    contingency_table (np.ndarray): A square contingency table.

    Returns:
    tuple: (chi2_stat, p_value)
    """
    n = contingency_table.shape[0]
    if n != contingency_table.shape[1]:
        raise ValueError("The contingency table must be square.")

    # Extract the off-diagonal elements
    off_diagonal = contingency_table[np.triu_indices_from(contingency_table, k=1)]
    off_diagonal_transposed = contingency_table[np.tril_indices_from(contingency_table, k=-1)]

    # Calculate the chi-square statistic
    numerator = (off_diagonal - off_diagonal_transposed) ** 2
    denominator = off_diagonal + off_diagonal_transposed

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2_stat = np.sum(np.where(denominator != 0, numerator / denominator, 0))

    # Degrees of freedom
    df = n * (n - 1) // 2

    # Calculate the p-value
    p_value = 1 - chi2.cdf(chi2_stat, df)

    return chi2_stat, p_value
def bow1_play_fish(resu_dict):
    # Combine p-values using Fisher's method
    pval_list = [entry['pval'] for entry in resu_dict.values()]
    stat_fish, pval_fish = combine_pvalues(pval_list, method='fisher')
    sign_glob = pval_fish < 0.05
    return stat_fish, pval_fish
def bow1_play_plo1(df):
    # Create a heatmap of the contingency table
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Contingency Table of CEAP Classes")
    plt.xlabel("Right Leg")
    plt.ylabel("Left Leg")
    #plt.show()
    pass
def bow1_play_subg(df):

    # Convert DataFrame to NumPy array
    contingency_table = df.to_numpy()
    # Function to perform chi-square test on a 2x2 contingency table
    def chi_square_test(table):
        chi2, p, dof, ex = chi2_contingency(table)
        return chi2, p
    # Perform subgroup analysis
    results = {}
    for i in range(contingency_table.shape[0]):
        # Extract counts for the current CEAP class
        left_leg_counts = contingency_table[i, i]
        right_leg_counts = contingency_table[i, :].sum() - left_leg_counts
        other_counts = contingency_table[:, i].sum() - left_leg_counts
        total_counts = contingency_table.sum() - left_leg_counts - right_leg_counts - other_counts
        # Create a 2x2 contingency table
        table = np.array([[left_leg_counts, right_leg_counts], [other_counts, total_counts]])
        # Perform chi-square test
        chi2, p = chi_square_test(table)
        results[df.index[i]] = {'chi2': chi2, 'p_value': p}
        # Print results
        for key, value in results.items():
            print(f"CEAP Class {key}: Chi-square statistic = {value['chi2']}, P-value = {value['p_value']}")

        # Interpretation
        significant_classes = []
        non_significant_classes = []

        for key, value in results.items():
            if value['p_value'] < 0.05:
                significant_classes.append(key)
            else:
                non_significant_classes.append(key)

        print("\nInterpretation:")
        print("CEAP Classes with statistically significant differences between left and right legs:")
        print(significant_classes)
        print("CEAP Classes with no statistically significant differences between left and right legs:")
        print(non_significant_classes)

    # Visualize the results
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Contingency Table of CEAP Classes")
    plt.xlabel("Right Leg")
    plt.ylabel("Left Leg")
    #plt.show()
    
    return results
    pass
def bow1_play_plo2(df):
    # Create a bar plot for each CEAP class
    for ceap_class in df.index:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=['Left Leg', 'Right Leg'], y=[df.loc[ceap_class, ceap_class], df.loc[ceap_class, :].sum() - df.loc[ceap_class, ceap_class]], palette='viridis')
        plt.title(f'CEAP Class {ceap_class} Distribution')
        plt.xlabel('Leg')
        plt.ylabel('Count')
        #plt.show()
        pass
def bow1_play_main():
    # Example DataFrame
    data = {
        'C0': [0, 0, 1, 9, 2, 5, 29],
        'C1': [0, 1, 0, 3, 2, 1, 3],
        'C2': [1, 1, 10, 15, 3, 5, 33],
        'C3': [8, 3, 14, 80, 17, 4, 9],
        'C4': [2, 2, 3, 16, 23, 7, 5],
        'C5': [2, 1, 3, 3, 7, 6, 7],
        'C6': [14, 0, 18, 14, 5, 8, 21]
    }

    index = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    df = pd.DataFrame(data, index=index)

    # Convert DataFrame to NumPy array
    contingency_table = df.to_numpy()

    # Apply Bowker test
    chi2_stat, p_value = bow1_play_func(contingency_table)
    print(f"Chi-square statistic: {chi2_stat}")
    print(f"P-value: {p_value}")

    bow1_play_plo1(df)
    resu_dict = bow1_play_subg(df)
    stat_fish, pval_fish = bow1_play_fish(resu_dict)
    bow1_play_plo2(df)
# ----
# Bowker 2 test (nx2 tables) play
# ----
'''
Backend TkAgg is interactive backend. Turning interactive mode on.
Bowker's Chi-Square: 16.297, p-value: 0.023, df: 7
Global Interpretation:
The global p-value (0.0225368543590303) is less than the significance level (0.05). There is evidence of asymmetry in the contingency tables.
Detailed Results:
{'key': 'C0', 'i': 0, 'j': 1, 'O_ij': 43, 'O_ji': 24, 'chi_square_ij': 5.388, 'p_value_ij': 0.02}
{'key': 'C1', 'i': 0, 'j': 1, 'O_ij': 5, 'O_ji': 4, 'chi_square_ij': 0.111, 'p_value_ij': 0.739}
{'key': 'C2', 'i': 0, 'j': 1, 'O_ij': 47, 'O_ji': 31, 'chi_square_ij': 3.282, 'p_value_ij': 0.07}
{'key': 'C3', 'i': 0, 'j': 1, 'O_ij': 43, 'O_ji': 46, 'chi_square_ij': 0.101, 'p_value_ij': 0.75}
{'key': 'C4', 'i': 0, 'j': 1, 'O_ij': 22, 'O_ji': 29, 'chi_square_ij': 0.961, 'p_value_ij': 0.327}
{'key': 'C5', 'i': 0, 'j': 1, 'O_ij': 18, 'O_ji': 23, 'chi_square_ij': 0.61, 'p_value_ij': 0.435}
{'key': 'C6', 'i': 0, 'j': 1, 'O_ij': 62, 'O_ji': 92, 'chi_square_ij': 5.844, 'p_value_ij': 0.016}
Detailed Interpretation:
The p-value (0.02) for pair (0, 1) in table C0 is significant. There is evidence of asymmetry.
The p-value (0.739) for pair (0, 1) in table C1 is not significant. There is no evidence of asymmetry, but this does not prove symmetry.
The p-value (0.07) for pair (0, 1) in table C2 is not significant. There is no evidence of asymmetry, but this does not prove symmetry.
The p-value (0.75) for pair (0, 1) in table C3 is not significant. There is no evidence of asymmetry, but this does not prove symmetry.
The p-value (0.327) for pair (0, 1) in table C4 is not significant. There is no evidence of asymmetry, but this does not prove symmetry.
The p-value (0.435) for pair (0, 1) in table C5 is not significant. There is no evidence of asymmetry, but this does not prove symmetry.
The p-value (0.016) for pair (0, 1) in table C6 is significant. There is evidence of asymmetry.
Significant Summary:
Significant pairs contributing to asymmetry:
- Pair (0, 1) in table C0: p-value = 0.02, chi-square = 5.388
- Pair (0, 1) in table C6: p-value = 0.016, chi-square = 5.844
'''
def bow2_play_func(cont_tabl, alpha=0.05):
    # Initialize variables for chi-square calculation
    global_chi_square = 0
    global_df = 0  # degrees of freedom is the number of unique off-diagonal pairs
    detailed_results = []

    # Iterate over each key in the dictionary
    for key, table in cont_tabl.items():
        # Ensure the table is 2x2
        rows, cols = table.shape
        if rows != 2 or cols != 2:
            raise ValueError(f"Table for key {key} must be 2x2")

        # Calculate chi-square for the current table
        chi_square = 0
        df = 0
        for i in range(rows):
            for j in range(i + 1, cols):
                O_ij = table[i, j]
                O_ji = table[j, i]
                if O_ij + O_ji > 0:
                    chi_square_ij = (O_ij - O_ji) ** 2 / (O_ij + O_ji)
                    chi_square += chi_square_ij
                    df += 1
                    p_value_ij = 1 - chi2.cdf(chi_square_ij, df=1)  # df=1 for each pair
                    detailed_results.append({
                        'key': key,
                        'i': i,
                        'j': j,
                        'O_ij': O_ij,
                        'O_ji': O_ji,
                        'chi_square_ij': round(chi_square_ij,3),
                        'p_value_ij': round(p_value_ij, 3)
                    })

        # Accumulate the global chi-square and degrees of freedom
        global_chi_square += chi_square
        global_df += df

    # Calculate the global p-value from the chi-square distribution
    global_p_value = 1 - chi2.cdf(global_chi_square, global_df)

    # Interpret the global p-value
    if global_p_value < alpha:
        global_interpretation = f"The global p-value ({global_p_value}) is less than the significance level ({alpha}). There is evidence of asymmetry in the contingency tables."
    else:
        global_interpretation = f"The global p-value ({global_p_value}) is greater than or equal to the significance level ({alpha}). There is no evidence of asymmetry in the contingency tables."

    # Interpret the detailed results
    detailed_interpretation = []
    significant_pairs = []
    for detail in detailed_results:
        if detail['p_value_ij'] < alpha:
            interpretation = f"The p-value ({detail['p_value_ij']}) for pair ({detail['i']}, {detail['j']}) in table {detail['key']} is significant. There is evidence of asymmetry."
            significant_pairs.append(detail)
        else:
            interpretation = f"The p-value ({detail['p_value_ij']}) for pair ({detail['i']}, {detail['j']}) in table {detail['key']} is not significant. There is no evidence of asymmetry, but this does not prove symmetry."
        detailed_interpretation.append(interpretation)

    # Summarize significant pairs
    if significant_pairs:
        significant_summary = "Significant pairs contributing to asymmetry:"
        for pair in significant_pairs:
            significant_summary += f"\n- Pair ({pair['i']}, {pair['j']}) in table {pair['key']}: p-value = {pair['p_value_ij']}, chi-square = {pair['chi_square_ij']}"
    else:
        significant_summary = "No significant pairs contributing to asymmetry."

    return {
        'chi_square': round(global_chi_square, 3),
        'p_value': round(global_p_value, 3),
        'df': global_df,
        'global_interpretation': global_interpretation,
        'detailed_results': detailed_results,
        'detailed_interpretation': detailed_interpretation,
        'significant_summary': significant_summary
    }

def bow2_play_plot():
    # Example contingency table
    data = np.array([[295, 43], [24, 0]])
    labels = ['Category 1', 'Category 2']

    # Create a DataFrame
    df = pd.DataFrame(data, index=labels, columns=labels)

    # Create a heatmap
    sns.heatmap(df, annot=True, cmap='coolwarm', fmt='d')
    plt.title("Contingency Table Heatmap")
    plt.show()
    
def bow2_play_main():
    # Example contingency table dictionary
    cont_tabl = {
        'C0': np.array([[295, 43], [24, 0]]),
        'C1': np.array([[352, 5], [4, 1]]),
        'C2': np.array([[274, 47], [31, 10]]),
        'C3': np.array([[193, 43], [46, 80]]),
        'C4': np.array([[288, 22], [29, 23]]),
        'C5': np.array([[315, 18], [23, 6]]),
        'C6': np.array([[187, 62], [92, 21]])
    }

    # Perform Bowker's test
    results = bow2_play_func(cont_tabl)
    print(f"Bowker's Chi-Square: {results['chi_square']}, p-value: {results['p_value']}, df: {results['df']}")
    print("Global Interpretation:")
    print(results['global_interpretation'])
    print("Detailed Results:")
    for detail in results['detailed_results']:
        print(detail)
    print("Detailed Interpretation:")
    for interpretation in results['detailed_interpretation']:
        print(interpretation)
    print("Significant Summary:")
    print(results['significant_summary'])
    #
    bow2_play_plot()
    pass

# ----
# Bowker 3
# ----
def bow3_play_intp(df_pval_adjusted):
    # Create an interpretation DataFrame based on p-value significance
    df_interpretation = pd.DataFrame("", index=df_pval_adjusted.index, columns=df_pval_adjusted.columns)

    # Interpretation thresholds
    for i in df_pval_adjusted.index:
        for j in df_pval_adjusted.columns:
            if pd.notna(df_pval_adjusted.loc[i, j]):
                pval = df_pval_adjusted.loc[i, j]
                if pval < 0.01:
                    if i < j:
                        df_interpretation.loc[i, j] = "RR"
                    else:
                        df_interpretation.loc[i, j] = "LL"
                elif pval < 0.05:
                    if i < j:
                        df_interpretation.loc[i, j] = "R"
                    else:
                        df_interpretation.loc[i, j] = "L"
                else:
                    df_interpretation.loc[i, j] = " "  # Non-significant

    # Display the final interpretation DataFrame
    print(df_interpretation)
    pass

def bow3_play_main():
    # Define your 8x8 contingency table
    contingency_table = np.array([
        [0, 0, 0, 2, 18, 5, 0, 22],
        [0, 0, 0, 1, 8, 2, 2, 14],
        [0, 0, 1, 0, 3, 2, 1, 0],
        [3, 1, 1, 10, 14, 3, 3, 18],
        [20, 9, 0, 15, 80, 16, 3, 14],
        [9, 2, 1, 3, 17, 23, 7, 5],
        [7, 5, 1, 5, 4, 2, 6, 8],
        [39, 29, 3, 33, 9, 5, 7, 21]
    ])

    # Initialize the result matrix
    pval_matrix = np.full((8, 8), np.nan)
    alpha = 0.05
    n_tests = 28  # Number of pairwise comparisons (8x8 matrix, but only off-diagonal pairs)

    # Perform pairwise Bowker tests for all off-diagonal pairs (i, j) and (j, i)
    for i in range(8):
        for j in range(i + 1, 8):  # Only off-diagonal pairs (i != j)
            # Extract the counts for the pairs (i, j) and (j, i)
            n_ij = contingency_table[i, j]
            n_ji = contingency_table[j, i]

            # Use Fisher's exact test if any of the observed values are 0
            if (n_ij == 0 or n_ji == 0) or (n_ij + n_ji) < 5:
                observed = np.array([[n_ij, n_ji], [n_ji, n_ij]])
                oddsratio, pval = fisher_exact(observed) # fisher copes with 0 observed values
            else:
                observed = np.array([[n_ij, n_ji]])
                chi2_stat, pval, dof, _ = chi2_contingency(observed, correction=False)

            # Store the p-value in the matrix
            pval_matrix[i, j] = pval
            pval_matrix[j, i] = pval  # Mirror for symmetry

    # Apply Bonferroni correction
    pval_adjusted = np.minimum(pval_matrix * n_tests, 1)  # Bonferroni correction: multiply p-values by the number of comparisons

    # Convert to a DataFrame for easier visualization
    df_pval_adjusted = pd.DataFrame(pval_adjusted, columns=["NA", "C0", "C1", "C2", "C3", "C4", "C5", "C6"],
                                                index=["NA", "C0", "C1", "C2", "C3", "C4", "C5", "C6"])
    bow3_play_intp(df_pval_adjusted)

    # Visualize using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pval_adjusted, annot=True, cmap="coolwarm", cbar=True, fmt=".3f",
                linewidths=.5, linecolor='black')

    plt.title("Heatmap of Adjusted P-values (Bonferroni correction)")
    plt.show()
    pass
# ----
# Bowker 4 'power' test for Bowker 1 (nx2 tables) play [for information]
# ----
def bow4_play_power_func(table):
    chi_square = 0
    df = 0
    for i in range(table.shape[0]):
        for j in range(i + 1, table.shape[1]):
            O_ij = table[i, j]
            O_ji = table[j, i]
            if O_ij + O_ji > 0:
                chi_square_ij = (O_ij - O_ji) ** 2 / (O_ij + O_ji)
                chi_square += chi_square_ij
                df += 1
    p_value = 1 - chi2.cdf(chi_square, df)
    return chi_square, p_value, df

def bow4_play_power_analysis(sample_size, asymmetry_magnitude, alpha=0.05, num_simulations=1000):
    power = 0
    for _ in range(num_simulations):
        # Simulate a 2x2 contingency table with asymmetry
        table = np.array([[sample_size // 2 + asymmetry_magnitude, sample_size // 2 - asymmetry_magnitude],
                         [sample_size // 2 - asymmetry_magnitude, sample_size // 2 + asymmetry_magnitude]])
        _, p_value, _ = bow4_play_power_func(table)
        if p_value < alpha:
            power += 1
    power /= num_simulations
    return power
def bow4_play_power_main():
    # Example contingency table dictionary
    cont_tabl = {
        'C0': np.array([[295, 43], [24, 0]]),
        'C1': np.array([[352, 5], [4, 1]]),
        'C2': np.array([[274, 47], [31, 10]]),
        'C3': np.array([[193, 43], [46, 80]]),
        'C4': np.array([[288, 22], [29, 23]]),
        'C5': np.array([[315, 18], [23, 6]]),
        'C6': np.array([[187, 62], [92, 21]])
    }

    # Extract the off-diagonal elements
    off_diagonal_elements = []
    for key, table in cont_tabl.items():
        off_diagonal_elements.append(table[0, 1])
        off_diagonal_elements.append(table[1, 0])

    # Calculate summary statistics
    mean_off_diagonal = np.mean(off_diagonal_elements)
    std_off_diagonal = np.std(off_diagonal_elements)
    min_off_diagonal = np.min(off_diagonal_elements)
    max_off_diagonal = np.max(off_diagonal_elements)

    print(f"Mean of off-diagonal elements: {mean_off_diagonal}")
    print(f"Standard deviation of off-diagonal elements: {std_off_diagonal}")
    print(f"Minimum of off-diagonal elements: {min_off_diagonal}")
    print(f"Maximum of off-diagonal elements: {max_off_diagonal}")

    # Choose asymmetry magnitudes based on the observed data
    asymmetry_magnitudes = [mean_off_diagonal - std_off_diagonal, mean_off_diagonal, mean_off_diagonal + std_off_diagonal]

    # Print the chosen asymmetry magnitudes
    print(f"Chosen asymmetry magnitudes: {asymmetry_magnitudes}")

    # Example power analysis with chosen asymmetry magnitudes
    sample_sizes = [50, 100, 200, 500]
    alpha = 0.05

    print("Power Analysis Results:")
    for sample_size in sample_sizes:
        for asymmetry_magnitude in asymmetry_magnitudes:
            power = bow4_play_power_analysis(sample_size, asymmetry_magnitude, alpha)
            print(f"Sample Size: {sample_size}, Asymmetry Magnitude: {asymmetry_magnitude}, Power: {power:.2f}")      
# ----
# Mc Nemar play
# ----

def nema_play_func(cont_tabl):
    
    # Inpu
    m00, m01 = cont_tabl[0][0], cont_tabl[0][1]
    m10, m11 = cont_tabl[1][0], cont_tabl[1][1]
    
    # Check for sufficient sample size (at least 10 discordant pairs)
    discordant_pairs = m01 + m10
    if discordant_pairs < 10:
        print ("Insufficient sample size: There must be at least 10 discordant pairs.")
    # Check for marginal homogeneity
    line_tota = np.array([m00+m01, m10+m11])
    colu_tota = np.array([m00+m10, m01+m11])
    if not np.isclose(line_tota[0], line_tota[1]) or not np.isclose(colu_tota[0], colu_tota[1]):
        print("Marginal homogeneity condition not satisfied: The row and column totals must be approximately equal.")
    
    print (f'a:{m00} b:{m01}')
    print (f'a:{m10} b:{m11}')
    print (f'line_tota:{line_tota} colu_tota:{colu_tota}')
    
    # Exec
    stat = (m01 - m10) ** 2 / (m01 + m10)
    p_value = stats.chi2.sf(stat, 1)
    
    return stat, p_value

def nema_play_main():
    # Sample contingency table
    cont_tabl = np.array([
                [30, 10],  # a, b
                [5,  25]]) # c, d
    df = pd.DataFrame(cont_tabl, columns=['Column 1', 'Column 2'], index=['Row 1', 'Row 2'])
    print (cont_tabl)
    print (df)
    try:
        stat, p_value = nema_play_func(cont_tabl)
        print(f"McNemar's test statistic: {stat:.4f}")
        print(f"P-value: {p_value:.4f}")
    except ValueError as e:
        print(e)

# ----
# Cochran [for information]
# ----
'''
It is an extension of the McNemar test for more than two related samples.
Purpose: Cochran's Q test is used to determine whether there are differences in 
the proportion of successes across multiple groups or conditions. 
Usage: Typically used in repeated measures designs where the same subjects are tested under different conditions.
Example: Suppose you have a group of subjects who are tested under three different conditions, 
and you want to see if the proportion of successes differs significantly across these conditions.
# Example data: 10 subjects tested under 3 conditions
data = np.array([
    [1, 0, 1],
    ...
    [1, 1, 0],
    [1, 0, 1]
])

# Perform Cochran's Q test
result = cok2rans_q(data)
'''

# ----
# Diaconis-Sturmfels test [pour information]
# ----
'''
Based on the search results, I would suggest using the Diaconis-Sturmfels algorithm to test for symmetry in contingency tables. This approach can be seen as complementary to the Bowker test, as it provides an alternative way to assess symmetry.
The key advantages of using the Diaconis-Sturmfels algorithm are:
It avoids relying on the χ2 approximation of the test statistic distribution, which may be inappropriate for sparse tables or tables with small expected values.
It allows for exact testing of the null hypothesis of symmetry, rather than an approximate test.
It provides a way to generate the exact distribution of the test statistic under the null hypothesis, enabling precise p-value calculations.
By rejecting the null hypothesis using the Diaconis-Sturmfels algorithm, you would be concluding that the data is asymmetric based on an exact test. This is in contrast to the Bowker test, which relies on an approximate χ2 distribution and may not be accurate for certain table configurations.
Rejecting the null hypothesis using the Diaconis-Sturmfels algorithm would provide strong evidence of asymmetry in the data, complementing the conclusions drawn from the Bowker test. The two approaches together can provide a more comprehensive assessment of symmetry in contingency tables.
'''

def diaconis_sturmfels_test(cont_tabl):
    """
    Perform the Diaconis-Sturmfels test for symmetry on a dictionary of 2x2 contingency tables.

    Parameters:
    - cont_tabl (dict): A dictionary where keys are identifiers and values are 2x2 numpy arrays.

    Returns:
    - dict: A dictionary containing the test results, including p-value and interpretation.
    """
    results = {}
    
    # Iterate over each key in the dictionary
    for key, table in cont_tabl.items():
        rows, cols = table.shape
        if rows != 2 or cols != 2:
            raise ValueError(f"Table for key {key} must be 2x2")

        # Create variables for the entries of the contingency table
        a, b, c, d = sp.symbols('a b c d')
        # Define the contingency table as a matrix
        matrix = np.array([[a, b], [c, d]])
        
        # Create equations based on the symmetry condition
        equations = [
            sp.Eq(a + d, b + c),  # Symmetry condition
            sp.Eq(a, c),          # First pair symmetry
            sp.Eq(b, d)           # Second pair symmetry
        ]
        
        # Solve the equations
        solution = sp.solve(equations, (a, b, c, d))
        
        # Check if the solution exists
        if solution:
            # Calculate the expected frequencies
            expected = np.array([[solution[a], solution[b]], [solution[c], solution[d]]])
            observed = table
            
            # Compute the test statistic (chi-square)
            chi_square_stat = np.sum((observed - expected) ** 2 / expected)
            df = 1  # Degrees of freedom for 2x2 table
            
            # Calculate p-value from the chi-square distribution
            p_value = 1 - sp.stats.chi2.cdf(chi_square_stat, df)
            
            # Store results
            results[key] = {
                'chi_square': chi_square_stat,
                'p_value': p_value,
                'interpretation': "Reject H0: Evidence of asymmetry." if p_value < 0.05 else "Fail to reject H0: No evidence of asymmetry."
            }
        else:
            results[key] = {
                'error': "No solution found for the symmetry equations."
            }
    
    return results

def diaconis_sturmfels_main():
    # Example contingency table dictionary
    cont_tabl = {
        'C0': np.array([[295, 43], [24, 0]]),
        'C1': np.array([[352, 5], [4, 1]]),
        'C2': np.array([[274, 47], [31, 10]]),
        'C3': np.array([[193, 43], [46, 80]]),
        'C4': np.array([[288, 22], [29, 23]]),
        'C5': np.array([[315, 18], [23, 6]]),
        'C6': np.array([[187, 62], [92, 21]])
    }

    # Perform the Diaconis-Sturmfels test
    results = diaconis_sturmfels_test(cont_tabl)
    for key, result in results.items():
        print(f"Results for table {key}:")
        if 'error' in result:
            print(result['error'])
        else:
            print(f"Chi-Square: {result['chi_square']:.3f}, p-value: {result['p_value']:.3f}")
            print(result['interpretation'])
        print()
     
if __name__ == "__main__":
    
    # Play
    # ----
    if False:
        #stu1_play_main()
        #nema_play_main()
        #stu1_play_main()
        #bow1_play_main()
        bow2_play_main()
        res1_play_main()
        bow3_play_main()