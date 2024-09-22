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
'''
ke43 : ceap_unbi_deta_abso:<class 'pandas.core.frame.DataFrame'>
           doss unbi                L_array                R_array
0        D10077    U  [0, 0, 0, 0, 0, 0, 0]  [0, 0, 0, 1, 0, 0, 0]
1        D10103    U  [0, 0, 0, 0, 0, 0, 0]  [0, 0, 0, 1, 0, 0, 0]
2        D10120    U  [0, 0, 0, 0, 0, 0, 1]  [0, 0, 0, 0, 0, 0, 0]
'''

# ----
# Inpu : Meth 1 : test
# ----
def inpu_met1():
    data = {
        'doss': ['D10077', 'D10103', 'D10120'],
        'sexe': ['F', 'M', 'M'],
        'L_array': [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]],
        'R_array': [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    }
    df = pd.DataFrame(data)
    print(f"df.size {len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    
    return df
    
# ----
# Inpu : Meth 2 : see 'GR04_zzzz_pand_ceap : pa43_ceap_unbi_deta()' : algorithm
# ----
def inpu_met2():
    # dire_path = "C:/tate01/grph01/gr05/inpu"
    # file_path = "InpuFile04.a.3a6_full.c4.UB.csv.oupu.csv"
    dire_path = os.path.dirname(os.path.abspath(__file__))
    file_path = "InpuFile04.a.3a6_full.c4.UB.csv.oupu.csv"
    df = file_inpu(f"{dire_path}/../inpu/{file_path}", deli="|")
    print(f"df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
        
    # Step 1
    # ------
    ordr_ceap = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    df = df[df['ceap'].isin(ordr_ceap)]
    #df = df.loc[df['unbi'] == 'B']
    df = df[['doss', 'sexe', 'mbre', 'ceap', 'unbi']] 
    print(f"df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    
    # Step 2: Nemar : create binary indicators for each CEAP class
    # ------
    df = df.copy() # avoids warning
    for ceap_class in ordr_ceap:
        df.loc[:, ceap_class] = df['ceap'].apply(lambda x: 1 if x == ceap_class else 0)
    df = df.head(10000)
    # Group by 'doss' and create separate arrays for L and R legs
    df = df.groupby(['doss', 'sexe', 'unbi']).agg(
        L_array=('mbre', lambda x: df.loc[x.index[x == 'G'], ordr_ceap].sum().astype(int).tolist()),
        R_array=('mbre', lambda x: df.loc[x.index[x == 'D'], ordr_ceap].sum().astype(int).tolist())
    ).reset_index()
    print(f"met2 : df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    
    return df
    
# ----
# Inpu : Meth 3 : see 'GR04_zzzz_pand_ceap : pa43_ceap_unbi_deta()' : output from it
# ----
def inpu_met3():
    
    dire_inpu = "C:/tate01/grph01/gr05/keep_c3c6_full_abso"
    df = file_inpu(f"{dire_inpu}/ke43_ceap_unbi_deta_abso.c3c6_full.abso.csv") 
    
    # dire_path = "C:/tate01/grph01/gr05/keep_c3c6_full_abso"
    # file_path = "ke43_ceap_unbi_deta_abso.c3c6_full.abso.csv"
    dire_path = os.path.dirname(os.path.abspath(__file__))
    file_path = "ke43_ceap_unbi_deta_abso.c3c6_full.abso.csv"
    df = file_inpu(f"{dire_path}/../inpu/{file_path}", deli="|")
    print(f"df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    
    
    df = df[['doss', 'sexe', 'unbi','L_array', 'R_array']]
    print(f"met3 : df.size:{len(df)} df.type:{type(df)}\n{df}\n:{df.index}\n:{df.columns}")
    
    return df

def inpu_C0C6():
    df2 = inpu_met2()
    df3 = inpu_met3()
    #print ('***')
    #print(f"df2.size\n{len(df2)} df2.type:{type(df2)}\n{df2}\n:{df2.index}\n:{df2.columns}")
    #print(f"df3.size\n{len(df3)} df3.type:{type(df3)}\n{df3}\n:{df3.index}\n:{df3.columns}")
    #print ('***')
    df3['L_array'] = df3['L_array'].apply(ast.literal_eval)
    df3['R_array'] = df3['R_array'].apply(ast.literal_eval)
    df2 = df2.rename(columns={ 'L_array': 'lisL', 'R_array': 'lisR' })
    df3 = df3.rename(columns={ 'L_array': 'lisL', 'R_array': 'lisR' })
    if not df2.equals(df3):
        raise Exception()
    print(f"inpu_C0C6 : df3.size:{len(df3)} df3.type:{type(df3)}\n{df3}\n:{df3.index}\n:{df3.columns}")
    return df3

def inpu_NAC6(df3):

    df4 = df3.copy()
    # Prepend 'NA' iff lisL,R is all 0
    def prepend_value(lis):
        if all(x == 0 for x in lis):
            return [1] + lis  # Prepend 1 if all are 0
        else:
            return [0] + lis  # Prepend 0 otherwise
    df4['lisL'] = df4['lisL'].apply(prepend_value)
    df4['lisR'] = df4['lisR'].apply(prepend_value)
    print(f"inpu_NAC6 : df4.size:{len(df4)} df4.type:{type(df4)}\n{df4}\n:{df4.index}\n:{df4.columns}")
    return df4
# ----
# Veri
# ----
def inpu_C0C6_veri(df):
    
    sum_lisL = df['lisL'].apply(pd.Series).sum().tolist()
    sum_lisR = df['lisR'].apply(pd.Series).sum().tolist()
    sum_list = [l + r for l, r in zip(sum_lisL, sum_lisR)]
    print(f"veri_C0C6 : Total1 lisL:{sum_lisL} lisR:{sum_lisR} list:{sum_list}")
    print(f"veri_C0C6 : Total2 lisL:{sum(sum_lisL)} lisR:{sum(sum_lisR)} list:{sum(sum_list)}")
    # Count rows where all items in lisL are zero
    lisL_zero = df['lisL'].apply(lambda x: all(i == 0 for i in x)).sum()
    lisR_zero = df['lisR'].apply(lambda x: all(i == 0 for i in x)).sum()
    print(f"veri_C0C6 : Total3 lisL_zero:{lisL_zero} lisR_zero:{lisR_zero} tota:{lisL_zero+lisR_zero}")

def inpu_NAC6_veri(df):    

    sum_lisL = df['lisL'].apply(pd.Series).sum().tolist()
    sum_lisR = df['lisR'].apply(pd.Series).sum().tolist()
    sum_list = [l + r for l, r in zip(sum_lisL, sum_lisR)]
    print(f"veri_NAC6 : Total1 lisL:{sum_lisL} lisR:{sum_lisR} list:{sum_list}")
    print(f"veri_NAC6 : Total2 lisL:{sum(sum_lisL)} lisR:{sum(sum_lisR)} list:{sum(sum_list)}")
    
# ####
# Mtrx
# ####    
# ----
# Create a nxn df
# ----
def mtrx_nxn(df_inpu):
    print (df_inpu)
    
    # Prec
    # ----
    row_index = df_inpu.index[0]  # Assuming you want the first row
    # Get the lists from the specified row
    row_lisL = df_inpu.loc[row_index, 'lisL']
    row_lisR = df_inpu.loc[row_index, 'lisL']
    # Get the sizes
    size_lisL = len(row_lisL)
    size_lisR = len(row_lisR)
    if not ((size_lisL == size_lisR) & (size_lisR == 8)): # 'NAC6' listformat
        raise Exception()
    
    # Step 1: Create an 8x8 DataFrame
    # ----
    df_oupu = pd.DataFrame(np.zeros((8, 8), dtype=int))
    #
    pairs = []
    for index, row in df_inpu.head(10000).iterrows():
        lisL = row['lisL']
        lisR = row['lisR']
        for i in range(8):
            for j in range(8):
                if lisL[i] == 1 and lisR[j] == 1:
                    pairs.append((i, j))
                    df_oupu.iloc[i, j] += 1
    # print(f"Row {index}: {pairs}") # to list all pairs

    # Step 2
    # ----
    if True:
        lisL_labl = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        lisR_labl = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        df_oupu.index = lisL_labl
        df_oupu.columns = lisR_labl
    else:
        lisL_labl = ['LNA', 'LC0', 'LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'LC6']
        lisR_labl = ['RNA', 'RC0', 'RC1', 'RC2', 'RC3', 'RC4', 'RC5', 'RC6']
        df_oupu.index = lisL_labl
        df_oupu.columns = lisR_labl
    
    print(f"df_oupu.size\n{len(df_oupu)} df_oupu.type:{type(df_oupu)}\n{df_oupu}\n:{df_oupu.index}\n:{df_oupu.columns}\n:{df_oupu.sum().sum()}")
        
    # Exit
    return df_oupu

# ----
# Create a nx2 df
# ----
def mtrx_nx2(df_inpu):
    if False:
        data = {
            'doss': ['D10077', 'D10103', 'D10104'],
            'lisL': [[1, 1, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
            'lisR': [[1, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
        }
        df_inpu = pd.DataFrame(data)
    if False:
        data = {
            'doss': ['D10077', 'D10103'],
            'lisL': [[1, 1, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
            'lisR': [[1, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
        }
        df_inpu = pd.DataFrame(data)
    if False:
        data = {
            'doss': ['D10077'],
            'lisL': [[1, 1, 0, 1, 0, 0, 0, 0]],
            'lisR': [[1, 0, 0, 1, 1, 1, 1, 0]]
        }
        df_inpu = pd.DataFrame(data)
        
    # Exec
    # ----
    # Create dict
    ordr_ceap = ['NA', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    cont_tabl_dict = {}  
    for indx, ceap in enumerate(ordr_ceap):
        mbre_left = [arra[indx] for arra in df_inpu['lisL']]
        mbre_righ = [arra[indx] for arra in df_inpu['lisR']]
        #print (f"{ceap}:{indx} : mbre_left : {mbre_left} mbre_righ: {mbre_righ}") 
        # 0,0 ; 0, 1 : ceap classe [member G absent] : [membre D absent] ; [membre D presnt]
        # 1,0 ; 1, 1 : ceap classe [member G presnt] : [membre D absent] ; [membre D presnt]   
        cont_item = [
            [sum((l == 0) & (r == 0) for l, r in zip(mbre_left, mbre_righ)), sum((l == 0) & (r == 1) for l, r in zip(mbre_left, mbre_righ))],
            [sum((l == 1) & (r == 0) for l, r in zip(mbre_left, mbre_righ)), sum((l == 1) & (r == 1) for l, r in zip(mbre_left, mbre_righ))]
        ]
        # print (ceap, cont_item)
        cont_tabl_dict[ceap] = cont_item
    if False:
        print (df_inpu)
        for ceap, cont_item in cont_tabl_dict.items():
            print(f"cont_tabl_dict[{ceap}] = {cont_item}")
    # Create array and data frame
    ar_oupu = {key: np.array(value) for key, value in cont_tabl_dict.items()}
    df_oupu = pd.DataFrame({key: [value] for key, value in cont_tabl_dict.items()}).T
    df_oupu.columns = ['cont_tabl']
    df_oupu['pati_nmbr'] = df_oupu['cont_tabl'].apply(lambda x: sum(sum(sublist) for sublist in x))
    #df_oupu['cont_tab2'] = df_oupu['cont_tabl'].apply(lambda x: [[0] + x[0][1:], x[1]])
    #df_oupu['pair_nmbr'] = df_oupu['cont_tab2'].apply(lambda x: sum(sum(sublist) for sublist in x))
    print(ar_oupu)
    print(df_oupu)
    #print("Sum of pair_nmbr:", df_oupu['pair_nmbr'].sum())
    return df_oupu

# ####
# Selc
# ####
def mtrx_nxn_selc(df_inpu, x00, xnn): # eg : 'LC2', 'LC2'


    top_lef = (x00, x00)  # (row_name_1, column_name_1)
    bot_rig = (xnn, xnn)  # (row_name_2, column_name_2)
    df_oupu = df_inpu.copy()
    df_oupu = df_oupu.loc[top_lef[0]:bot_rig[0], top_lef[1]:bot_rig[1]]
    
    return df_oupu

def mtrx_nx2_selc(df_inpu, x0, xn): # eg : 'LC2', 'LC2'
    print(f"df_inpu.size\n{len(df_inpu)} df_inpu.type:{type(df_inpu)}\n{df_inpu}\n:{df_inpu.index}\n:{df_inpu.columns}")
    df_oupu = df_inpu.copy()
    df_oupu = df_oupu.loc[x0:xn, ['cont_tabl']]
    print(f"df_oupu.size\n{len(df_oupu)} df_oupu.type:{type(df_oupu)}\n{df_oupu}\n:{df_oupu.index}\n:{df_oupu.columns}")
    
    return df_oupu
# ####
# Main
# ####
def inpu_func():
        
    # Inpu
    # ----
    df3 = inpu_C0C6()
    inpu_C0C6_veri(df3)
    df4 = inpu_NAC6(df3)
    inpu_NAC6_veri(df4)
    
    # Sexe
    # ----
    df_A = df4.copy()
    df_M = df4[df4['sexe'] == 'M']
    df_F = df4[df4['sexe'] == 'F']
    if len(df_A) != len(df_M) + len(df_F):
        raise Exception()
    
    # Mtrx
    # ----
    df_A_nxn = mtrx_nxn(df_A)
    df_A_nx2 = mtrx_nx2(df_A)
    print(f"df_nxn.size:{len(df_A_nxn)}\ndf_nxn.type:{type(df_A_nxn)}\n{df_A_nxn}\n:{df_A_nxn.index}\n:{df_A_nxn.columns}\n:{df_A_nxn.sum().sum()}")
    print(f"df_nx2.size:{len(df_A_nx2)}\ndf_nx2.type:{type(df_A_nx2)}\n{df_A_nx2}\n:{df_A_nx2.index}\n:{df_A_nx2.columns}")
    df_M_nxn = mtrx_nxn(df_M)
    df_M_nx2 = mtrx_nx2(df_M)
    df_F_nxn = mtrx_nxn(df_F)
    df_F_nx2 = mtrx_nx2(df_F)
    
    # Visu
    # ----
    df_A_nxnv = df_A_nxn.copy()
    df_A_nxnv.index = ['G_' + str(i) for i in df_A_nxnv.index]
    df_A_nxnv.columns = ['D_' + str(col) for col in df_A_nxnv.columns]
    print (df_A_nxnv)

    # Exit
    return df_A_nxn, df_A_nx2, df_M_nxn, df_M_nx2, df_F_nxn, df_F_nx2

def main():
    
    # Exec
    df_A_nxn, df_A_nx2, df_M_nxn, df_M_nx2, df_F_nxn, df_F_nx2 = inpu_func()  

    # Selc
    # ----
    df_nxn_C0C0 = mtrx_nxn_selc(df_A_nxn, 'LC0', 'RC0')
    print(f"df_nxn_C0C0.size:{len(df_nxn_C0C0)}\ndf_nxn_C0C0.type:{type(df_nxn_C0C0)}\n{df_nxn_C0C0}\n:{df_nxn_C0C0.index}\n:{df_nxn_C0C0.columns}\n:{df_nxn_C0C0.sum().sum()}")

# ----
#
# ----
if __name__ == "__main__":
    main()