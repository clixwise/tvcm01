import pandas as pd
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
data = {
    'M': [8, 15, 23, 61, 102, 83, 70, 16, 0],
    'F': [4, 16, 64, 74, 105, 114, 92, 26, 3]
}
df = pd.DataFrame(data, index=['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'])

# Print the DataFrame
print("df:")
print(df)

# Calculate the sum of all elements in the DataFrame
df_sum = df.sum().sum()

# Print the sum
print("\ndf.sum():")
print(df_sum)

# Restructure cols as lists to be used by stat tests
df_stat = df.copy()
df_stat.columns = ['obsM', 'obsF']
obsM = df_stat['obsM'].tolist()
obsF = df_stat['obsF'].tolist()
