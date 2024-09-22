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

# ####
# Inpu
# ####
    
# ----
# Inpu : Meth 2 : see 'GR04_zzzz_pand_ceap : pa43_ceap_unbi_deta()' : algorithm
# ----
def inpu_met2():
    
    # Step 1
    # ------
    # dire_path = "C:/tate01/grph01/gr05/inpu"
    # file_path = "InpuFile04.a.3a6_full.c4.UB.csv.oupu.csv"
    dire_path = os.path.dirname(os.path.abspath(__file__))
    file_path = "InpuFile04.a.3a6_full.c4.UB.csv.oupu.csv"
    df = file_inpu(f"{dire_path}/../inpu/{file_path}", deli="|")
    print(f"df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
        
    # Step 2
    # ------
    ordr_ceap = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    df = df[df['ceap'].isin(ordr_ceap)]
    df_inpu = df[['doss', 'age', 'age_bin', 'sexe', 'unbi', 'mbre', 'ceap']] 
    print(f"df_inpu.size:{len(df_inpu)} df_inpu.type:{type(df_inpu)}\n{df_inpu}\n:{df_inpu.index}\n:{df_inpu.columns}")

    # Step 3
    # ------
    def line_fill(df, memG, memD):
        df1 = df.copy()
        df1 = df1[(df1['unbi'] == 'U') & (df1['mbre'] == memG)]
        print(f"df1.size:{len(df1)} df1.type:{type(df1)}\n{df1}\n:{df1.index}\n:{df1.columns}")
        df2 = df1.groupby(['doss', 'age', 'age_bin', 'sexe', 'unbi']).size().reset_index(name='coun')
        #df2 = df2[(df2['coun'] > 1)]
        df2['mbre'] = memD
        df2['ceap'] = 'NA'
        print(f"df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
        return df2
    dfD = line_fill(df_inpu, 'G', 'D')
    dfG = line_fill(df_inpu, 'D', 'G')
    df_oupu = pd.concat([df_inpu, dfD, dfG], ignore_index=True)
    print(f"df_oupu.size:{len(df_oupu)} df_oupu.type:{type(df_oupu)}\n{df_oupu}\n:{df_oupu.index}\n:{df_oupu.columns}")
    
    # Exit
    return df_oupu

# ####
# Part 10
# ####
'''
ke41 : ceap_unbi_glob
ceap_D  NA  C0  C1  C2  C3  C4  C5  C6
ceap_G
NA       0   0   0   2  18   5   0  22
C0       0   0   0   1   8   2   2  14
C1       0   0   1   0   3   2   1   0
C2       3   1   1  10  14   3   3  18
C3      20   9   0  15  80  16   3  14
C4       9   2   1   3  17  23   7   5
C5       7   5   1   5   4   2   6   8
C6      39  29   3  33   9   5   7  21
:Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], dtype='object', name='ceap_G')
:Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], dtype='object', name='ceap_D')
 df2.sum:542
'''
def exec_10(df0, indx_name, indx_list, colu_name, colu_list):

    if False:
        data_extended = {
            'doss': ['D1', 'D1', 'D2', 'D2', 'D1', 'D2', 'D3', 'D3', 'D1', 'D3', 'D3'],
            'ceap': ['C0', 'C1', 'C1', 'C2', 'C2', 'C3', 'C1', 'C4', 'C5', 'C6', 'C1'],
            'mbre': ['G', 'G', 'G', 'G', 'D', 'D', 'D', 'D', 'G', 'G', 'D']
        }
        df_extended = pd.DataFrame(data_extended)
    # Separate 'G' and 'D' groups
    df_g = df0[df0['mbre'] == 'G']
    df_d = df0[df0['mbre'] == 'D']
    # Create all combinations of 'G' and 'D'
    combinations = pd.merge(df_g, df_d, on='doss', suffixes=('_G', '_D'))
    # Cross-tabulation to count occurrences
    df2 = pd.crosstab(
        pd.Categorical(combinations['ceap_G'], categories=indx_list),
        pd.Categorical(combinations['ceap_D'], categories=colu_list)
    ).reindex(index=indx_list, columns=colu_list, fill_value=0)
    df2.index.name = 'ceap_G'
    df2.columns.name = 'ceap_D'
    print(f"df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}\n df2.sum:{df2.sum().sum()}")
    return df2

def main_11_ceap_ceap(df1):  
    indx_name = 'ceap'
    colu_name = 'ceap'
    indx_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    colu_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    df = exec_10(df1, indx_name, indx_list, colu_name, colu_list)
    what = inspect.currentframe().f_code.co_name
    desc = ["ceap","ceap"]
    return what, df, desc
    
# ####
# Part 20
# ####
'''
ke15 : agbi_sexe::<class 'pandas.core.frame.DataFrame'>
         M    F
10-19    8    4
20-29   15   16
30-39   23   64
40-49   61   74
50-59  102  105
60-69   83  114
70-79   70   92
80-89   16   26
90-99    0    3
:Index(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'], dtype='object')
:Index(['M', 'F'], dtype='object')
df2.sum:876
'''
# ----
#
# ----
'''
ke19 : agbi_mbre::<class 'pandas.core.frame.DataFrame'>
         G    D
10-19    6    6
20-29   14   17
30-39   43   44
40-49   64   71
50-59  101  106
60-69   99   98
70-79   80   82
80-89   21   21
90-99    1    2
:Index(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'], dtype='object')
:Index(['G', 'D'], dtype='object')
df2.sum:876
'''
# ----
#
# ----
'''
ke37 : ceap_sexe:<class 'pandas.core.frame.DataFrame'>
     M    F
NA  52   53
C0  31   36
C1   5    6
C2  44   54
C3  93  156
C4  38   59
C5  18   35
C6  97   99
:Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], dtype='object')
:Index(['M', 'F'], dtype='object')
df2.sum:876
'''
# ----
#
# ----
'''
ke39 : ceap_mbre:<class 'pandas.core.frame.DataFrame'>
      G    D
NA   39   66
C0   24   43
C1    5    6
C2   41   57
C3  126  123
C4   52   45
C5   29   24
C6  113   83
:Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], dtype='object')
:Index(['G', 'D'], dtype='object')
sum:876
'''
# ----
#
# ----
'''
kexx : ceap_agbi:<class 'pandas.core.frame.DataFrame'>
        10-19  20-29  30-39  40-49  50-59  60-69  70-79  80-89  90-99
NA      3      5      8     24     25     15     17      8      0
C0      1      2      9      7     17     16     11      4      0
C1      0      1      0      2      3      4      1      0      0
C2      1      5     15      9     22     22     18      6      0
C3      5      8     21     38     66     54     46      9      2
C4      0      3      8     16     12     38     18      2      0
C5      1      1      4      7     12     15     11      2      0
C6      1      6     22     32     50     33     40     11      1
:Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], dtype='object')
:Index(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'], dtype='object')
df2.sum:876
'''
def exec_20(df0, indx_name, indx_list, colu_name, colu_list):

    # df1
    # ---
    df1 = df0.copy()
    df1 = df1[df1[indx_name].isin(indx_list)]
    df1 = df1[df1[colu_name].isin(colu_list)]
    # df2
    # ---
    df2 = pd.DataFrame(index=indx_list, columns=colu_list)
    df2 = df2.fillna(0)
    for index, row in df1.iterrows():
        valu_indx = row[indx_name]
        valu_colu = row[colu_name]
        df2.at[valu_indx, valu_colu] += 1
    print(f"df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}\n df2.sum:{df2.sum().sum()}")
    # exit
    # ----
    return df2

def main_21_agbi_sexe(df1):    
    indx_name = 'age_bin'
    colu_name = 'sexe'
    indx_list = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
    colu_list = ['M', 'F']
    df = exec_20(df1, indx_name, indx_list, colu_name, colu_list)
    what = inspect.currentframe().f_code.co_name
    desc = ["age_bin","sexe"]
    return what, df, desc
def main_22_agbi_mbre(df1): 
    indx_name = 'age_bin'
    colu_name = 'mbre'
    indx_list = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
    colu_list = ['G', 'D']
    df = exec_20(df1, indx_name, indx_list, colu_name, colu_list)
    what = inspect.currentframe().f_code.co_name
    desc = ["age_bin","mbre"]
    return what, df, desc
def main_23_ceap_sexe(df1):
    indx_name = 'ceap'
    colu_name = 'sexe'
    indx_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    colu_list = ['M', 'F']
    df = exec_20(df1, indx_name, indx_list, colu_name, colu_list)
    what = inspect.currentframe().f_code.co_name
    desc = ["ceap","sexe"]
    return what, df, desc
def main_24_ceap_mbre(df1):
    indx_name = 'ceap'
    colu_name = 'mbre'
    indx_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    colu_list = ['G', 'D']
    df = exec_20(df1, indx_name, indx_list, colu_name, colu_list)  
    what = inspect.currentframe().f_code.co_name
    desc = ["ceap","mbre"]
    return what, df, desc
def main_25_ceap_agbi(df1):
    indx_name = 'ceap'
    colu_name = 'age_bin'
    indx_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    colu_list = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
    df = exec_20(df1, indx_name, indx_list, colu_name, colu_list)
    what = inspect.currentframe().f_code.co_name
    desc = ["ceap","age_bin"]
    return what, df, desc    

# ####
# Part 30
# ####
'''
ke49 : ceap_sexe_agbi_abso
ceap   NA      C0    C1     C2      C3      C4     C5      C6    
sexe    M   F   M  F  M  F   M   F   M   F   M   F  M   F   M   F
10-19   2   1   1  0  0  0   1   0   2   3   0   0  1   0   1   0
20-29   2   3   1  1  1  0   4   1   4   4   0   3  0   1   3   3
30-39   5   3   1  8  0  0   1  14   2  19   2   6  2   2  10  12
40-49  11  13   4  3  1  1   6   3  13  25   6  10  1   6  19  13
50-59  16   9  11  6  0  3  14   8  19  47   4   8  5   7  33  17
60-69   9   6   7  9  2  2   9  13  22  32  16  22  4  11  14  19
70-79   5  12   4  7  1  0   7  11  26  20  10   8  4   7  13  27
80-89   2   6   2  2  0  0   2   4   5   4   0   2  1   1   4   7
90-99   0   0   0  0  0  0   0   0   0   2   0   0  0   0   0   1
:Index(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89',
       '90-99'],
      dtype='object')
:MultiIndex([('NA', 'M'),
            ('NA', 'F'),
            ('C0', 'M'),
            ('C0', 'F'),
            ('C1', 'M'),
            ('C1', 'F'),
            ('C2', 'M'),
            ('C2', 'F'),
            ('C3', 'M'),
            ('C3', 'F'),
            ('C4', 'M'),
            ('C4', 'F'),
            ('C5', 'M'),
            ('C5', 'F'),
            ('C6', 'M'),
            ('C6', 'F')],
           names=['ceap', 'sexe'])
 df2.sum:876
'''
'''
ke51 ceap_sexe_mbre_abso
sexe   M       F
mbre   G   D   G   D
NA    18  34  21  32
C0    11  20  13  23
C1     2   3   3   3
C2    21  23  20  34
C3    48  45  78  78
C4    22  16  30  29
C5     8  10  21  14
C6    58  39  55  44
:Index(['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'], dtype='object')
:MultiIndex([('M', 'G'),
            ('M', 'D'),
            ('F', 'G'),
            ('F', 'D')],
           names=['sexe', 'mbre'])
 df2.sum:876
'''
def exec_30(df0, indx_name, indx_list, col1_name, col1_list, col2_name, col2_list):
    
    # df1
    # ---
    df1 = df0.copy()
    df1 = df1[df1[indx_name].isin(indx_list)]
    df1 = df1[df1[col1_name].isin(col1_list)]
    df1 = df1[df1[col2_name].isin(col2_list)]

    # df2
    # ---
    columns = pd.MultiIndex.from_product([col1_list, col2_list], names=[col1_name, col2_name])
    df2 = pd.DataFrame(index=indx_list, columns=columns)
    df2 = df2.fillna(0)

    for index, row in df1.iterrows():
        valu_indx = row[indx_name]
        valu_col1 = row[col1_name]
        valu_col2 = row[col2_name]
        df2.loc[valu_indx, (valu_col1, valu_col2)] += 1

    print(f"df2.size:{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}\n df2.sum:{df2.sum().sum()}")

    # exit
    # ----
    return df2

def main_31_agbi_ceap_sexe(df1):
    indx_name = 'age_bin'
    col1_name = 'ceap'
    col2_name = 'sexe'
    indx_list = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
    col1_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    col2_list = ['M', 'F']
    df = exec_30(df1, indx_name, indx_list, col1_name, col1_list, col2_name, col2_list)
    what = inspect.currentframe().f_code.co_name
    desc = ["age_bin","sexe"]
    return what, df, desc
def main_32_ceap_sexe_mbre(df1):
    indx_name = 'ceap'
    col1_name = 'sexe'
    col2_name = 'mbre'
    indx_list = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    col1_list = ['M', 'F']
    col2_list = ['G', 'D']
    df = exec_30(df1, indx_name, indx_list, col1_name, col1_list, col2_name, col2_list)
    what = inspect.currentframe().f_code.co_name
    desc = ["age_bin","sexe"]
    return what, df, desc

def main():

    df1 = inpu_met2()
    
    main_11_ceap_ceap(df1)
    
    main_21_agbi_sexe(df1)
    main_22_agbi_mbre(df1)
    main_23_ceap_sexe(df1)
    main_24_ceap_mbre(df1)
    main_25_ceap_agbi(df1)
    
    main_31_agbi_ceap_sexe(df1)
    main_32_ceap_sexe_mbre(df1)
    
# ----
#
# ----
if __name__ == "__main__":
    main()
