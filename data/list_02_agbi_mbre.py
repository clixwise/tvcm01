import pandas as pd
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
data = {
    'G': [6, 14, 43, 64, 101, 99, 80, 21, 1],
    'D': [6, 17, 44, 71, 106, 98, 82, 21, 2]
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